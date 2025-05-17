# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "trajectory_comprehensive"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments, question_template: str):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.question_template = question_template

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        QUESTION_TEMPLATE = self.question_template
        def make_conversation_image(example):
            return {
                "prompt": [
                    # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                        ],
                    },
                ],
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if 'image' in example:
            image_path = os.path.join(image_root, example['image'])
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
            image = Image.open(image_path).convert("RGB")
        #     # 取原图尺寸
        #     w, h = image.size
        #     max_dim = max(w, h)

        #     # 计算填充颜色：根据图片内容计算平均颜色
        #     image_mean = np.array(image).mean(axis=(0, 1)).astype(int)
        #     bg_color = tuple(image_mean)

        #     # 创建正方形新图，填充为 image_mean 背景色
        #     new_image = Image.new("RGB", (max_dim, max_dim), bg_color)

        #     # 将原图居中贴在正方形上
        #     new_image.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))

            # 替换掉原图
            # image = new_image
        else:
            image = None
        

        return {
            'image': image,
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt'],
        }


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    elif "TrajectoryRewardCalculator" in model_name_or_path.lower():
        return TrajectoryRewardCalculator
    elif "robobrain" in model_name_or_path.lower():
        return None
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")
    

def interpolate_points(points, num=20):
    if len(points) < 2:
        return np.array(points * num)[:num]  # 如果只有1个点，重复
    points = np.array(points)
    dists = np.cumsum([0] + [np.linalg.norm(points[i] - points[i-1]) for i in range(1, len(points))])
    total_len = dists[-1]
    interp_dists = np.linspace(0, total_len, num)
    interp_x = np.interp(interp_dists, dists, points[:, 0])
    interp_y = np.interp(interp_dists, dists, points[:, 1])
    return np.stack([interp_x, interp_y], axis=1)



def trajectory_reward_interp(completions, solution, **kwargs):
    import re
    rewards = []
    point_pattern = r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    contents = [completion[0]["content"] for completion in completions]
    for content, sol in zip(contents, solution):
        # match = re.findall(pattern, pred[0]["content"])
        main_list_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if not main_list_match:
            rewards.append(0.0)
            continue
        try:
            points_str = main_list_match.group(1)
            point_matches = re.findall(point_pattern, points_str)
            if point_matches and 3 <= len(point_matches) <= 10:
                pred_pts = np.array([[float(x), float(y)] for x, y in point_matches])
                interp_pred = interpolate_points(pred_pts, num=20)
                interp_sol = interpolate_points(sol, num=20)
                mse = np.mean(np.sum((interp_pred - interp_sol) ** 2, axis=1))
                rewards.append(max(0.0, 1.0 - mse / 10000))  # 可根据经验调调 10000
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards

def trajectory_reward(completions, solution, **kwargs):
    import re
    import os
    from datetime import datetime
    pattern = r"\[(.*?)\]"
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    for pred, sol in zip(completions, solution):
        match = re.findall(pattern, pred[0]["content"])
        if not match:
            rewards.append(0.0)
            continue
        try:
            pred_pts = eval("[" + match[0] + "]")
            mse = sum((px - sx)**2 + (py - sy)**2 for (px, py), (sx, sy) in zip(pred_pts, sol)) / len(pred_pts)
            reward = max(0.0, 1.0 - mse / 10000)
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Trajectory reward: {reward} -------------\n")
                    f.write(f"Prediction: {pred_pts}\n")
                    f.write(f"Solution: {sol}\n")
        except:
            rewards.append(0.0)
    return rewards

# 这只是用来评价平滑度的
# def trajectory_smooth_reward(completions, solution, **kwargs):
#     import re
#     import os
#     from datetime import datetime
#     pattern = r"\[(.*?)\]"
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
#     for pred in completions:
#         match = re.findall(pattern, pred[0]["content"])
#         if not match:
#             rewards.append(0.0)
#             continue
#         try:
#             pred_pts = eval("[" + match[0] + "]")
#             smoothness = sum(
#                 abs((pred_pts[i+1][0] - pred_pts[i][0]) - (pred_pts[i][0] - pred_pts[i-1][0])) +
#                 abs((pred_pts[i+1][1] - pred_pts[i][1]) - (pred_pts[i][1] - pred_pts[i-1][1]))
#                 for i in range(1, len(pred_pts)-1)
#             )
#             reward = max(0.0, 1.0 - smoothness / 1000)
#             rewards.append(reward)
#             if os.getenv("DEBUG_MODE") == "true":
#                 log_path = os.getenv("LOG_PATH")
#                 with open(log_path, "a", encoding='utf-8') as f:
#                     f.write(f"------------- {current_time} Trajectory smooth reward: {reward} -------------\n")
#                     f.write(f"Prediction: {pred_pts}\n")
#         except:
#             rewards.append(0.0)
#     return rewards

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    vlm_module_tra = TrajectoryRewardCalculator
    print("using vlm module:", vlm_module_cls.__name__)

    # Load the reward functions
    reward_funcs_registry = {
        "accuracy": vlm_module_cls.iou_reward,
        "format": vlm_module_cls.format_reward_tra,
        "trajectory_interp":vlm_module_cls.trajectory_reward_interp,
        "trajectory":trajectory_reward_interp,
        "trajectory_comprehensive":vlm_module_tra.trajectory_reward_comprehensive
        # "trajectory_smooth":trajectory_smooth_reward
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)
    # QUESTION_TEMPLATE = vlm_module_cls.get_question_template(task_type="trajectory")
    # print("🚧 QUESTION_TEMPLATE =", QUESTION_TEMPLATE)  # ✅ 打印模板内容
    # # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args, question_template=vlm_module_cls.get_question_template(task_type="trajectory"))

    trainer_cls = VLMGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
