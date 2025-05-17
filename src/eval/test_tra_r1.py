from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import sys
import os
from Funtions import TrajectoryRewardCalculator
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re

from pprint import pprint
import random
import numpy as np
import similaritymeasures
from scipy.spatial.distance import directed_hausdorff
from similaritymeasures import frechet_dist
import torch.distributed as dist
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

parser = argparse.ArgumentParser(description="Set steps value and version.")
parser.add_argument("--steps", type=int, default=100, help="Number of steps")
parser.add_argument("--version", type=str, default="output", help="New or Old version")
parser.add_argument("--run_name", type=str, default="Qwen2.5-VL-3B-GRPO-REC_trajectory_ALL", help="Run name")
args = parser.parse_args()

steps = args.steps
version = args.version  # 修正变量名
RUN_NAME = args.run_name

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    return local_rank, world_size, rank

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")


if rank == 0:
    print("Steps: ", steps)



MODEL_PATH=f"ManipLVM-R1/src/open-r1-multimodal/{version}/{RUN_NAME}/checkpoint-{steps}"
OUTPUT_PATH="./logs/{RUN_NAME}/results_{DATASET}_{RUN_NAME}_{STEPS}.json"

BSZ=64
DATA_ROOT = "ManipLVM-R1/data/ShareRobot/trajectory"



TEST_DATASETS = ['test_trajectory']
IMAGE_ROOT = "ManipLVM-R1/data/ShareRobot/trajectory/images"


#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank},
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def extract_trajectory_answer(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    point_pattern = r'\[\s*(-?\d*\.?\d+(?:e[-+]?\d+)?)\s*,\s*(-?\d*\.?\d+(?:e[-+]?\d+)?)\s*\]'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        point_matches = re.findall(point_pattern, content_answer)
        if point_matches and 3 <= len(point_matches) <= 10:
            pred_pts = np.array([[float(x), float(y)] for x, y in point_matches])
            return pred_pts
    return None

# num_samples = 1000
for ds in TEST_DATASETS:
    if rank == 0:
        print(f"Processing {ds}...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))

    QUESTION_TEMPLATE = """You are a robot using joint control. The task is to "{Question}".
Please predict 3–10 trajectory points to complete the task.First output the thinking process in <think> </think> tags
The final answer should be a detailed trajectory in JSON format, containing 3-10 points: {{[x1, y1], [x2, y2], ...\}}, and please provide it within <answer>...</answer> tags without any additional text.
"""

    # Split data for distributed evaluation
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]

    messages = []

    for x in rank_data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        message = [
            # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                }
            ]
        }]
        messages.append(message)

    rank_outputs = [] # List to store answers for this rank
    all_outputs = []  # List to store all answers

    # Process data
    for i in tqdm(range(0, len(messages), BSZ), disable=rank != 0):
        batch_messages = messages[i:i + BSZ]

        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]

        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        rank_outputs.extend(batch_output_text)

    print(f"Rank {rank} has finished processing {len(rank_outputs)} examples")

    # Gather all outputs from all ranks
    all_outputs = [None] * len(data)
    rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)

    # Ensure all data is gathered correctly
    if rank == 0:
        last_gathered_index = -1
        for results in gathered_results:
            if results: # Check if the list is not empty
                last_gathered_index = max(last_gathered_index, results[-1][0])
        assert last_gathered_index == len(data) - 1, f"Expected last index {len(data) - 1}, but got {last_gathered_index}"


    # The main process will collect all results
    if rank == 0:
        for results in gathered_results:
            for idx, output in results:
                assert idx < len(all_outputs), f"Index {idx} out of bounds for all_outputs of size {len(all_outputs)}"
                all_outputs[idx] = output
        # Check if all elements have been filled
        assert all(item is not None for item in all_outputs), "Not all outputs were gathered"


        final_output = []
        average_dfd = 0.0 # Use float for clarity
        average_hd = 0.0
        average_rmse = 0.0
        valid_dfd_count = 0
        valid_hd_count = 0 # Track valid HD calculations
        valid_rmse_count = 0 # Track valid RMSE calculations

        for input_example, model_output in zip(data, all_outputs):
            original_output = model_output
            ground_truth = input_example['solution'] # This is already a list of lists (trajectory)
            model_answer = extract_trajectory_answer(original_output) # This returns np.array or None

            gt_traj = None
            pred_traj = model_answer # Already np.array or None

            # Ensure ground truth is a valid numpy array
            try:
                # Assuming ground_truth is a list of lists or similar structure
                gt_traj_list = ground_truth # Use the original list
                gt_traj = np.array(gt_traj_list, dtype=float)
                # Basic validation of shape
                if gt_traj.ndim != 2 or gt_traj.shape[1] != 2:
                    print(f"Warning: Ground truth trajectory has unexpected shape: {gt_traj.shape}. Skipping metrics.")
                    gt_traj = None # Invalidate for metric calculation
            except Exception as e:
                print(f"Warning: Could not convert ground truth trajectory to numpy array: {e}. Skipping metrics.")
                gt_traj = None # Invalidate for metric calculation

            dfd_distance = None # Initialize metrics as None
            hd_distance = None
            rmse_distance = None
            num_resample_points = 20

            # Check if both trajectories are valid numpy arrays for calculations
            if isinstance(pred_traj, np.ndarray) and isinstance(gt_traj, np.ndarray):
                # Check for minimum points required by some metrics
                if len(pred_traj) >= 3 and len(gt_traj) >= 3:
                    try:
                        # --- Calculate DFD (Discrete Frechet Distance) ---
                        try:
                            dfd_distance = frechet_dist(pred_traj, gt_traj)
                            if dfd_distance is not None and np.isfinite(dfd_distance):
                                average_dfd += dfd_distance
                                valid_dfd_count += 1
                            else:
                                dfd_distance = None # Ensure it's None if calculation failed or resulted in non-finite
                        except NameError:
                            pass # Silently skip if function not found
                        except Exception as e:
                            print(f"Error calculating DFD: {e}")
                            dfd_distance = None

                        # --- Calculate HD (Hausdorff Distance) ---
                        try:
                            dist_pred_to_gt = directed_hausdorff(pred_traj, gt_traj)[0]
                            dist_gt_to_pred = directed_hausdorff(gt_traj, pred_traj)[0]
                            hd_distance = max(dist_pred_to_gt, dist_gt_to_pred)
                            if hd_distance is not None and np.isfinite(hd_distance):
                                average_hd += hd_distance
                                valid_hd_count += 1
                            else:
                                hd_distance = None
                        except NameError:
                            pass # Silently skip if function not found
                        except Exception as e:
                            print(f"Error calculating Hausdorff distance: {e}")
                            hd_distance = None

                        # --- Calculate RMSE (Root Mean Square Error) after resampling ---
                        if len(pred_traj) >= 2 and len(gt_traj) >= 2:
                            try:
                                pred_resampled = TrajectoryRewardCalculator.resample_trajectory_by_arc_length(pred_traj, num_resample_points)
                                gt_resampled = TrajectoryRewardCalculator.resample_trajectory_by_arc_length(gt_traj, num_resample_points)

                                if pred_resampled is not None and gt_resampled is not None:
                                    squared_errors = np.sum((pred_resampled - gt_resampled)**2, axis=1)
                                    rmse_distance = np.sqrt(np.mean(squared_errors))
                                    if rmse_distance is not None and np.isfinite(rmse_distance):
                                        average_rmse += rmse_distance
                                        valid_rmse_count += 1
                                    else:
                                        rmse_distance = None
                                else:
                                    rmse_distance = None # Resampling failed
                            except NameError:
                                pass # Silently skip if class not found
                            except Exception as e:
                                print(f"Error calculating RMSE: {e}")
                                rmse_distance = None
                        else: # Trajectories too short for RMSE
                             rmse_distance = None

                    except Exception as e:
                        # Catch any unexpected error during the metric calculations for this item
                        print(f"General error during metric calculation for example: {e}")
                        # Ensure distances are reset if a general error occurred within the block
                        dfd_distance = None
                        hd_distance = None
                        rmse_distance = None
                # else: # Trajectories exist but are too short (handled within specific metric checks)
                #     pass # Metrics remain None

            # else: # pred_traj was None or gt_traj was None
            #     pass # Metrics remain None

            # Create a result dictionary for this example
            # Ground truth is stored as the original list, extracted_answer as np.array (or None)
            result = {
                'image': input_example['image'],
                'question': input_example['solution'], # This is the text question
                'ground_truth': ground_truth, # Store original ground truth (list of lists)
                'model_output': original_output,
                'extracted_answer': model_answer, # Store np.array or None
                'DFD': dfd_distance,
                'HD': hd_distance,
                'RMSE': rmse_distance
            }
            final_output.append(result)
            # Note: Averages are now accumulated within the metric calculation blocks

        # Calculate final averages
        average_dfd = average_dfd / valid_dfd_count if valid_dfd_count > 0 else float('nan')
        average_hd = average_hd / valid_hd_count if valid_hd_count > 0 else float('nan')
        average_rmse = average_rmse / valid_rmse_count if valid_rmse_count > 0 else float('nan')

        # --- Define conversion function ---
        def convert_numpy_to_python(data):
            if isinstance(data, np.ndarray):
                return data.tolist()
            if isinstance(data, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(data)
            elif isinstance(data, (np.float16, np.float32,
                                  np.float64)):
                # Handle NaN specifically for JSON compatibility
                return float('nan') if np.isnan(data) else float(data)
            elif isinstance(data, (np.complex64, np.complex128)):
                return {'real': data.real, 'imag': data.imag}
            elif isinstance(data, (np.bool_)):
                return bool(data)
            elif isinstance(data, (np.void)):
                return None
            elif isinstance(data, dict):
                return {key: convert_numpy_to_python(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [convert_numpy_to_python(item) for item in data]
            # Handle None explicitly if needed, though it's usually JSON serializable
            elif data is None:
                return None
            else:
                # Attempt to return data directly if it's already a standard Python type
                if isinstance(data, (str, int, float, bool, dict, list)):
                     return data
                # If it's some other type not handled, maybe return its string representation or raise error
                # For now, return as is, json.dump might raise error later if not serializable
                # print(f"Warning: Unhandled type {type(data)} in conversion.") # Optional warning
                return data


        # --- Convert results before saving ---
        converted_final_output = [convert_numpy_to_python(result) for result in final_output]

        # Convert average values
        average_dfd_py = convert_numpy_to_python(average_dfd)
        average_hd_py = convert_numpy_to_python(average_hd)
        average_rmse_py = convert_numpy_to_python(average_rmse)

        # Print using converted values (handle potential NaN for printing)
        dfd_to_print = f"{average_dfd_py:.2f}" if not np.isnan(average_dfd_py) else "NaN"
        hd_to_print = f"{average_hd_py:.2f}" if not np.isnan(average_hd_py) else "NaN"
        rmse_to_print = f"{average_rmse_py:.2f}" if not np.isnan(average_rmse_py) else "NaN"

        print(f"Average DFD: {dfd_to_print}, "
              f"Average HD: {hd_to_print}, "
              f"Average RMSE: {rmse_to_print}")


        # Save results to a JSON file using converted values
        output_path = OUTPUT_PATH.format(DATASET=ds, RUN_NAME=RUN_NAME, STEPS=steps)
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            json.dump({
                'average_dfd': average_dfd_py,
                'average_hd': average_hd_py,
                'average_rmse': average_rmse_py,
                'results': converted_final_output # 使用转换后的列表
            }, f, indent=2, allow_nan=True) # allow_nan=True is important if metrics can be nan

        print(f"Results saved to {output_path}")
        print("-"*100)

    # Synchronize all processes before starting the next dataset (if any)
    dist.barrier()




