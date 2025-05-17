from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import re
import numpy as np
import os
import datetime
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
# Optional dependencies for trajectory reward calculation
euclidean_norm = lambda x, y: np.linalg.norm(x - y)

try:
    # similaritymeasures: For discrete Frechet distance and other metrics
    import similaritymeasures
except ImportError:
    print("Warning: 'similaritymeasures' library not found. Frechet distance-based reward functions will not be available.")
    # Define a placeholder class/function
    class SimilarityMeasuresPlaceholder:
        def frechet_dist(self, *args, **kwargs):
            raise NotImplementedError("Frechet distance calculation requires the 'similaritymeasures' library.")
    similaritymeasures = SimilarityMeasuresPlaceholder()

from open_r1.vlm_modules.vlm_module import VLMBaseModule

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "affordance":
                return "You are a robot using the joint control. The task is \"{Question}\". Please predict a possible affordance area of the end effector?"
            case "affordance_long":
                return "You are a robot using the joint control. The task is \"{Question}\". Please predict a possible affordance area of the end effector.First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "trajectory":
                return """You are a robot using joint control. The task is to "{Question}".  
Please predict 3–10 trajectory points to complete the task.First output the thinking process in <think> </think> tags
The final answer should be a detailed trajectory in JSON format, containing 3-10 points: {{[x1, y1], [x2, y2], ...\}}, and please provide it within <answer>...</answer> tags without any additional text.
"""

            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def format_reward_tra(completions, **kwargs):
        import re
        answer_pattern = r"<think>.*?</think>\s*<answer>(.*?)</answer>"
        point_pattern = r'\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]'
        
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content in completion_contents:
            match = re.search(answer_pattern, content, re.DOTALL)
            if match:
                answer_content = match.group(1)
                points = re.findall(point_pattern, answer_content)
                if 3 <= len(points) <= 10:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        
        return rewards

    
    @staticmethod
    def format_reward(completions, **kwargs):
        import re
        pattern = r"<think>.*?</think>\s*<answer>.*?\[.*?{\"bbox_2d\":\s*\[\s*\d+,\s*\d+,\s*\d+,\s*\d+\s*\]\s*,\s*\"label\":\s*\".*?\"\s*}.*?\].*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
        
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'(?:\[\s*\d+\s*,\s*\d+\s*\])(?:\s*,\s*\[\s*\d+\s*,\s*\d+\s*\]){2,9}'
        for content, sol in zip(contents, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards
    

    # --- main reward function ---
    @staticmethod
    def trajectory_reward_interp(completions, solution, **kwargs):
        """
        calculate reward based on predicted trajectory (completions) and ground truth trajectory (solution).

        Args:
            completions (list): model generated prediction results list, each element contains predicted text.
                             expected format: [{'content': '<answer>[[x,y],...]</answer>'}]
            solution (list): corresponding ground truth trajectory list. Each element should be a Numpy array.

        **kwargs: optional parameters for configuring reward function.
            reward_type (str): reward calculation method ('dtw_endpoint', 'frechet_endpoint', 'endpoint_only').
                              default is 'dtw_endpoint'.
            w_endpoint (float): weight of endpoint reward.
                w_similarity (float): weight of trajectory similarity reward (DTW or Frechet).
                k_endpoint (float): endpoint reward distance decay coefficient.
                k_dtw (float): DTW reward distance decay coefficient.
                k_frechet (float): Frechet reward distance decay coefficient.

        Returns:
            list: a list of reward scores for each prediction.
        """
        # --- Helper functions ---
        # Get current time for logging
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def parse_trajectory_2d(text):
            """
            parse 2D trajectory coordinates from text, support two formats:
            1. square brackets: [[x1, y1], [x2, y2],...]
            2. curly brackets: {[x1, y1], [x2, y2],...}
            
            return a Numpy array (N, 2), if parsing fails return None.
            """
            try:
                # use more general regex to match both square brackets and curly brackets
                main_list_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
                # match each coordinate point separately
                point_pattern = r'\[\s*(-?\d*\.?\d+(?:e[-+]?\d+)?)\s*,\s*(-?\d*\.?\d+(?:e[-+]?\d+)?)\s*\]'
                if main_list_match:
                    points_str = main_list_match.group(1)
                    # find all coordinate points
                    point_matches = re.findall(point_pattern, points_str)
                    # ensure there are 3-10 coordinate points
                    if point_matches and 3 <= len(point_matches) <= 10:
                        # convert extracted tuples to numerical list
                        pred_pts = np.array([[float(x), float(y)] for x, y in point_matches])
                        return pred_pts
                    else:
                        return None
                else:
                    return None

            except Exception as e:
                return None

        def calculate_endpoint_reward(pred_traj, gt_traj, k_endpoint=1.0):
            """
            calculate reward based on endpoint Euclidean distance.
            reward = exp(-k * distance^2), value range [0, 1], smaller distance means higher reward.
            """
            try:
                pred_end = pred_traj[-1]
                gt_end = gt_traj[-1]
                endpoint_dist = np.linalg.norm(pred_end - gt_end) # 计算欧氏距离
                reward = np.exp(-k_endpoint * endpoint_dist**2) # 指数衰减函数 [2]    
                return reward
            except Exception as e:
                print(f"error when calculating endpoint reward: {e}")
                return 0.0

        def calculate_dtw_reward(pred_traj, gt_traj, k_dtw=0.000001):
            """
            calculate reward based on DTW distance.
            reward = exp(-k * dtw_distance), value range [0, 1], smaller distance means higher reward.
            need dtw-python library.
            """
            # DTW 
            try:
                # 使用 dtw-python 
                # alignment.distance
                dtw_distance, _ = fastdtw(pred_traj, gt_traj, dist=euclidean)
                # reward = np.exp(-k_dtw * dtw_distance) 
                reward = max(0, 1 - k_dtw * dtw_distance)
                return reward
            except NotImplementedError:
                print("DTW calculation skipped: dependency library 'dtw-python' not installed.")
                return 0.0
            except Exception as e:
                print(f"error when calculating DTW reward: {e}")
                return 0.0

        def calculate_frechet_reward(pred_traj, gt_traj, k_frechet=0.1):
            """
            calculate reward based on discrete Frechet distance.
            reward = exp(-k * frechet_distance), value range [0, 1], smaller distance means higher reward.
            need similaritymeasures library.
            """
            if pred_traj is None or gt_traj is None or len(pred_traj) == 0 or len(gt_traj) == 0:
                return 0.0
            try:
                frechet_result = similaritymeasures.frechet_dist(pred_traj, gt_traj)
                if isinstance(frechet_result, tuple):
                    frechet_dist_val = frechet_result[0]
                else:
                    frechet_dist_val = frechet_result
                reward = np.exp(-k_frechet * frechet_dist_val)
                return reward
            except Exception as e:
                print(f"error when calculating Frechet reward: {e}")
                return 0.0
        
        rewards_list = []
        
        # extract predicted contents from completions
        contents = [completion[0]["content"] for completion in completions]
        # --- configure reward function parameters ---
        reward_type = kwargs.get('reward_type', 'dtw_endpoint') 
        w_endpoint = kwargs.get('w_endpoint', 0.1)     
        w_similarity = kwargs.get('w_similarity', 0.9) 

        k_endpoint = kwargs.get('k_endpoint', 1.0)      
        k_dtw = kwargs.get('k_dtw', 0.1)                
        k_frechet = kwargs.get('k_frechet', 0.1)          

        # --- iterate over each prediction and corresponding ground truth ---
        for content, sol_gt_traj in zip(contents, solution):
            reward = 0.0
            pred_traj = None
            
            try:
                pred_traj = parse_trajectory_2d(content)
                # 2. validate ground truth trajectory format
                if not isinstance(sol_gt_traj, np.ndarray):
                    # try to convert elements of solution to Numpy array
                    sol_gt_traj = np.array(sol_gt_traj, dtype=float)

                # 3. calculate reward (only when predicted and ground truth trajectories are valid)
                if pred_traj is not None and sol_gt_traj is not None:
                    # calculate basic reward components
                    r_end = calculate_endpoint_reward(pred_traj, sol_gt_traj, k_endpoint)
                    
                    # calculate final reward based on reward_type
                    if reward_type == 'dtw_endpoint':
                        r_sim = calculate_dtw_reward(pred_traj, sol_gt_traj, k_dtw)
                        reward = w_endpoint * r_end + w_similarity * r_sim
                    elif reward_type == 'frechet_endpoint':
                        r_sim = calculate_frechet_reward(pred_traj, sol_gt_traj, k_frechet)
                        reward = w_endpoint * r_end + w_similarity * r_sim
                    elif reward_type == 'endpoint_only':
                        reward = r_end # only consider endpoint accuracy
                    # can add more reward combinations, e.g. only DTW or only Frechet
                    elif reward_type == 'dtw_only':
                        reward = calculate_dtw_reward(pred_traj, sol_gt_traj, k_dtw)
                    elif reward_type == 'frechet_only':
                        reward = calculate_frechet_reward(pred_traj, sol_gt_traj, k_frechet)
                    else:
                        print(f"warning: unknown reward_type '{reward_type}'.")
                        r_sim = calculate_dtw_reward(pred_traj, sol_gt_traj, k_dtw)
                        reward = w_endpoint * r_end + w_similarity * r_sim
                        
                    # ensure reward value is in the range [0, 1] (for exponential decay and positive weights, usually this is the case, but just in case)
                    reward = max(0.0, min(1.0, reward)) 

                else:
                    reward = 0.0 # if failed to calculate, reward is 0

                # add debug log
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    if log_path:
                        with open(log_path, "a", encoding='utf-8') as f:
                            f.write(f"------------- {current_time} Trajectory reward: {reward} -------------\n")
                            f.write(f"Content: {content}\n")
                            f.write(f"Predicted trajectory: {pred_traj}\n")
                            f.write(f"Ground truth trajectory: {sol_gt_traj}\n")
                            if pred_traj is not None and sol_gt_traj is not None:
                                f.write(f"Reward type: {reward_type}\n")
                                f.write(f"Endpoint reward: {r_end}\n")
                                if reward_type in ['dtw_endpoint', 'dtw_only']:
                                    f.write(f"DTW reward: {r_sim if 'r_sim' in locals() else 'N/A'}\n")
                                elif reward_type in ['frechet_endpoint', 'frechet_only']:
                                    f.write(f"Frechet reward: {r_sim if 'r_sim' in locals() else 'N/A'}\n")
                            f.write("\n")

            except Exception as e:
                reward = 0.0 # if failed to calculate, reward is 0
                
                # add debug log
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    if log_path:
                        with open(log_path, "a", encoding='utf-8') as f:
                            f.write(f"------------- {current_time} Trajectory reward ERROR -------------\n")
                            f.write(f"Content: {content}\n")
                            f.write(f"Error: {str(e)}\n\n")

            rewards_list.append(reward)
        rewards_list = [0.0 if r is None else r for r in rewards_list]
        return rewards_list

import numpy as np
import re
import datetime
import os
from scipy.spatial.distance import directed_hausdorff
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import similaritymeasures # For Frechet Distance

class TrajectoryRewardCalculator:

    # --- Helper: Trajectory Parsing ---
    @staticmethod
    def parse_trajectory_2d(text):
        """
        parse 2D trajectory coordinates from <answer>...</answer> tag.
        support [[x, y], ...] format.
        return a Numpy array (N, 2) containing 3 to 10 points, if parsing fails or the number of points is invalid, return None.
        """
        try:
            # 提取 <answer> 标签内的内容
            main_list_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
            if not main_list_match:
                # print("Debug: No <answer> tag found.")
                return None
            
            content_inside_answer = main_list_match.group(1).strip()
            # print(f"Debug: Content inside answer: {content_inside_answer}")
            point_pattern = r'\[\s*(-?\d*\.?\d+(?:e[-+]?\d+)?)\s*,\s*(-?\d*\.?\d+(?:e[-+]?\d+)?)\s*\]'
            point_matches = re.findall(point_pattern, content_inside_answer)    
            if point_matches and 3 <= len(point_matches) <= 10:
                pred_pts = np.array([[float(x), float(y)] for x, y in point_matches])
                return pred_pts
            else:
                # print(f"Debug: Invalid number of points: {num_points}")
                return None

        except Exception as e:
            # print(f"Debug: General parsing error: {e}")
            return None

    # --- Helper: Arc Length Resampling for RMSE ---
    @staticmethod
    def resample_trajectory_by_arc_length(traj, num_points):
        """
        resample trajectory to fixed number of points by arc length parameterization.

        Args:
            traj (np.ndarray): original trajectory (N, 2).
            num_points (int): number of points after resampling.

        Returns:
            np.ndarray or None: resampled trajectory (num_points, 2), if failed return None.
        """
        if traj is None or len(traj) < 2:
            return None
        try:
            segment_lengths = np.linalg.norm(np.diff(traj, axis=0), axis=1)

            cumulative_arc_length = np.concatenate(([0], np.cumsum(segment_lengths)))
            total_length = cumulative_arc_length[-1]

            if total_length <= 1e-6: # avoid zero division
                return np.tile(traj[0], (num_points, 1))

            f_x = interp1d(cumulative_arc_length, traj[:, 0], kind='linear', fill_value="extrapolate")
            f_y = interp1d(cumulative_arc_length, traj[:, 1], kind='linear', fill_value="extrapolate")

            s_resampled = np.linspace(0, total_length, num_points)

            x_resampled = f_x(s_resampled)
            y_resampled = f_y(s_resampled)

            return np.vstack((x_resampled, y_resampled)).T
        except Exception as e:
            return None

    # --- Helper: Reward/Penalty Calculations ---
    @staticmethod
    def calculate_endpoint_reward(pred_traj, gt_traj, k_endpoint=1.0):
        """calculate endpoint reward"""
        if pred_traj is None or gt_traj is None or len(pred_traj) < 1 or len(gt_traj) < 1:
            return 0.0
        try:
            pred_end = pred_traj[-1]
            gt_end = gt_traj[-1]
            endpoint_dist = np.linalg.norm(pred_end - gt_end)
            reward = np.exp(-k_endpoint * endpoint_dist**2)
            return reward
        except Exception as e:
            return 0.0

    @staticmethod
    def _metric_to_reward(distance, k_factor):
        """convert distance to reward"""
        if distance is None or distance < 0:
            return 0.0

        reward = np.exp(-k_factor * distance)
        return reward

    @staticmethod
    def calculate_dtw_reward(pred_traj, gt_traj, k_dtw=0.1):
        """calculate dtw reward"""
        if pred_traj is None or gt_traj is None or len(pred_traj) < 2 or len(gt_traj) < 2:
             return 0.0
        try:
            dtw_distance, _ = fastdtw(pred_traj, gt_traj, dist=euclidean)
            return TrajectoryRewardCalculator._metric_to_reward(dtw_distance, k_dtw)
        except ImportError:
            print("DTW error")
            return 0.0
        except Exception as e:
            return 0.0

    @staticmethod
    def calculate_frechet_reward(pred_traj, gt_traj, k_frechet=0.1):
        """calculate frechet reward"""
        if pred_traj is None or gt_traj is None or len(pred_traj) < 1 or len(gt_traj) < 1:
            return 0.0
        try:
            frechet_dist_val, _ = similaritymeasures.frechet_dist(pred_traj, gt_traj)
            return TrajectoryRewardCalculator._metric_to_reward(frechet_dist_val, k_frechet)
        except ImportError:
            print("Frechet error")
            return 0.0
        except Exception as e:
            return 0.0

    @staticmethod
    def calculate_hausdorff_reward(pred_traj, gt_traj, k_hd=0.1):
        """calculate hausdorff reward"""
        if pred_traj is None or gt_traj is None or len(pred_traj) < 1 or len(gt_traj) < 1:
            return 0.0
        try:
            dist_pred_to_gt = directed_hausdorff(pred_traj, gt_traj)[0]
            dist_gt_to_pred = directed_hausdorff(gt_traj, pred_traj)[0]
            hausdorff_dist = max(dist_pred_to_gt, dist_gt_to_pred)
            return TrajectoryRewardCalculator._metric_to_reward(hausdorff_dist, k_hd)
        except ImportError:
            print("Hausdorff error")
            return 0.0
        except Exception as e:
            return 0.0

    @staticmethod
    def calculate_rmse_reward(pred_traj, gt_traj, k_rmse=0.1, num_resample_points=20):
        """calculate rmse reward"""
        if pred_traj is None or gt_traj is None or len(pred_traj) < 2 or len(gt_traj) < 2:
             return 0.0
        try:
            pred_resampled = TrajectoryRewardCalculator.resample_trajectory_by_arc_length(pred_traj, num_resample_points)
            gt_resampled = TrajectoryRewardCalculator.resample_trajectory_by_arc_length(gt_traj, num_resample_points)

            if pred_resampled is None or gt_resampled is None:
                 return 0.0
            
            squared_errors = np.sum((pred_resampled - gt_resampled)**2, axis=1)
            rmse = np.sqrt(np.mean(squared_errors))
            return TrajectoryRewardCalculator._metric_to_reward(rmse, k_rmse)
        except ImportError:
            print("RMSE error")
            return 0.0
        except Exception as e:
            return 0.0

    @staticmethod
    def calculate_jerk_penalty(pred_traj, dt=0.1):
        if pred_traj is None or len(pred_traj) < 4:
            return 0.0 
        try:
            velocity = np.diff(pred_traj, axis=0) / dt
            acceleration = np.diff(velocity, axis=0) / dt
            # Jerk j = diff(a) / dt = diff(p, 3) / dt^3
            jerk = np.diff(acceleration, axis=0) / dt

            squared_jerk_magnitude = np.sum(jerk**2, axis=1)
            mean_squared_jerk = np.mean(squared_jerk_magnitude)


            return mean_squared_jerk
        except Exception as e:
            return 0.0 

    @staticmethod
    def calculate_feasibility_penalty(pred_traj, joint_limits=None, collision_checker=None, dt=0.1):
        """
        计算可行性惩罚 (0 表示可行, >0 表示不可行)。
        检查关节限制和碰撞。
        """
        penalty = 0.0
        if pred_traj is None or len(pred_traj) < 1:
            return 1.0 

        try:
            if joint_limits:
                has_vel_limits = 'velocity' in joint_limits
                has_acc_limits = 'acceleration' in joint_limits
                has_jerk_limits = 'jerk' in joint_limits
                
                velocity = None
                acceleration = None
                jerk = None

                if len(pred_traj) > 1 and (has_vel_limits or has_acc_limits or has_jerk_limits):
                    velocity = np.diff(pred_traj, axis=0) / dt
                if len(pred_traj) > 2 and (has_acc_limits or has_jerk_limits):
                    acceleration = np.diff(velocity, axis=0) / dt
                if len(pred_traj) > 3 and has_jerk_limits:
                    jerk = np.diff(acceleration, axis=0) / dt

                for i, pos in enumerate(pred_traj):
                    if 'position' in joint_limits:
                        if np.any(pos < joint_limits['position']['min']) or np.any(pos > joint_limits['position']['max']):
                            penalty = 1.0
                            break
                    if has_vel_limits and velocity is not None and i > 0:

                         # vel_mag = np.linalg.norm(velocity[i-1])
                         # if vel_mag > joint_limits['velocity']['max_magnitude']:
                         if np.any(np.abs(velocity[i-1]) > joint_limits['velocity']['max']):
                             penalty = 1.0
                             break

                    if has_acc_limits and acceleration is not None and i > 1:
                         if np.any(np.abs(acceleration[i-2]) > joint_limits['acceleration']['max']):
                             penalty = 1.0
                             break
                    if has_jerk_limits and jerk is not None and i > 2:
                         if np.any(np.abs(jerk[i-3]) > joint_limits['jerk']['max']):
                             penalty = 1.0
                             break
                
                if penalty > 0:
                    return penalty 


            if collision_checker:
                for q in pred_traj:
                    if collision_checker(q): 
                        penalty = 1.0

                        break 
            return penalty

        except Exception as e:
            
            return 1.0 

    # --- 主奖励函数 ---
    @staticmethod
    def trajectory_reward_comprehensive(completions, solution, **kwargs):
        """
        Calculate comprehensive reward based on predicted trajectory (completions) and ground truth trajectory (solution).
        Includes DFD, HD, RMSE, Endpoint, Smoothness(Jerk), and Feasibility.

        Args:
            completions (list): List of model-generated predictions, e.g., [[{'content': '<answer>...</answer>'}]]
            solution (list): Corresponding ground truth trajectory list (numpy arrays or convertible to arrays).
            **kwargs: Configuration parameters:
                reward_type (str): Define which components to use ('all', 'similarity_endpoint', 'dtw_endpoint', 'frechet_endpoint', 'quality_only', ...)
                num_resample_points (int): Number of resampling points for RMSE, default 20.
                dt (float): Assumed time step for derivative calculations (Jerk etc.), default 0.1.
                # Similarity & Endpoint Weights and Scaling Factors
                w_dfd (float): DFD reward weight, default 0.25
                w_hd (float): HD reward weight, default 0.15
                w_rmse (float): RMSE reward weight, default 0.20
                w_endpoint (float): Endpoint reward weight, default 0.40
                k_dfd (float): DFD distance decay coefficient, default 0.1
                k_hd (float): HD distance decay coefficient, default 0.1
                k_rmse (float): RMSE distance decay coefficient, default 0.5
                k_endpoint (float): Endpoint distance decay coefficient, default 1.0
                # Quality Weights and Parameters
                w_jerk (float): Jerk penalty weight, default 0.1
                w_feas (float): Feasibility penalty weight, default 10.0 (strong penalty)
                joint_limits (dict): Joint limits dictionary, e.g., 
                                        {'position': {'min': [-1,-1], 'max': [1,1]}, 
                                        'velocity': {'max': [10,10]}}
                collision_checker (callable): Collision checking function checker(q) -> bool (True if collision)

        Returns:
            list: List containing comprehensive reward scores for each prediction.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rewards_list = []

        reward_type = kwargs.get('reward_type', 'all') 
        num_resample_points = kwargs.get('num_resample_points', 20)
        dt = kwargs.get('dt', 0.1) 

        w_dfd = kwargs.get('w_dfd', 0.25)
        w_hd = kwargs.get('w_hd', 0.15)
        w_rmse = kwargs.get('w_rmse', 0.20)
        w_endpoint = kwargs.get('w_endpoint', 0.40)

        k_dfd = kwargs.get('k_dfd', 0.1)
        k_hd = kwargs.get('k_hd', 0.1)
        k_rmse = kwargs.get('k_rmse', 0.5) 
        k_endpoint = kwargs.get('k_endpoint', 1.0) 

        w_jerk = kwargs.get('w_jerk', 0.0) 
        w_feas = kwargs.get('w_feas', 0.0) 

        
        joint_limits = kwargs.get('joint_limits', None)
        collision_checker = kwargs.get('collision_checker', None) 
        try:
             contents = [completion[0]["content"] for completion in completions]
        except (TypeError, IndexError, KeyError):
             print("error: 'completions' 格式不符合预期 [[{'content': ...}]]")
             return [0.0] * len(completions)


            # --- 遍历计算 ---
        for content, sol_gt_traj_orig in zip(contents, solution):
            pred_traj = None
            sol_gt_traj = None
            log_data = {"content": content, "error": None}

            try:

                pred_traj = TrajectoryRewardCalculator.parse_trajectory_2d(content)
                log_data["pred_traj"] = pred_traj


                if isinstance(sol_gt_traj_orig, np.ndarray):
                    sol_gt_traj = sol_gt_traj_orig.astype(float)
                else:
                    try:
                        sol_gt_traj = np.array(sol_gt_traj_orig, dtype=float)
                        if sol_gt_traj.ndim != 2 or sol_gt_traj.shape[1] != 2:
                             raise ValueError("error")
                    except Exception as e:
                        log_data["error"] = f"error: {e}"
                        sol_gt_traj = None 
                log_data["gt_traj"] = sol_gt_traj

                reward = 0.0
                positive_reward = 0.0
                negative_penalty = 0.0

                if pred_traj is not None and sol_gt_traj is not None:
                    r_dfd = TrajectoryRewardCalculator.calculate_frechet_reward(pred_traj, sol_gt_traj, k_dfd)
                    r_hd = TrajectoryRewardCalculator.calculate_hausdorff_reward(pred_traj, sol_gt_traj, k_hd)
                    r_rmse = TrajectoryRewardCalculator.calculate_rmse_reward(pred_traj, sol_gt_traj, k_rmse, num_resample_points)
                    r_endpoint = TrajectoryRewardCalculator.calculate_endpoint_reward(pred_traj, sol_gt_traj, k_endpoint)
                    log_data.update({"r_dfd": r_dfd, "r_hd": r_hd, "r_rmse": r_rmse, "r_endpoint": r_endpoint})

                    p_jerk = TrajectoryRewardCalculator.calculate_jerk_penalty(pred_traj, dt)
                    p_feas = TrajectoryRewardCalculator.calculate_feasibility_penalty(pred_traj, joint_limits, collision_checker, dt)
                    log_data.update({"p_jerk": p_jerk, "p_feas": p_feas})

                    if reward_type == 'all':
                        positive_reward = (w_dfd * r_dfd +
                                           w_hd * r_hd +
                                           w_rmse * r_rmse +
                                           w_endpoint * r_endpoint)
                        negative_penalty = 0
                    elif reward_type == 'similarity_endpoint': # DFD+HD+RMSE+Endpoint
                         positive_reward = (w_dfd * r_dfd + w_hd * r_hd + w_rmse * r_rmse + w_endpoint * r_endpoint)
                         negative_penalty = 0 
                    elif reward_type == 'dtw_endpoint':
                        k_dtw = kwargs.get('k_dtw', 0.1)
                        r_dtw = TrajectoryRewardCalculator.calculate_dtw_reward(pred_traj, sol_gt_traj, k_dtw) 
                        log_data["r_dtw"] = r_dtw
                        positive_reward = kwargs.get('w_similarity', 0.6) * r_dtw + kwargs.get('w_endpoint', 0.4) * r_endpoint
                        negative_penalty = (w_jerk * p_jerk + w_feas * p_feas) 
                    elif reward_type == 'frechet_endpoint': 
                         positive_reward = kwargs.get('w_similarity', 0.6) * r_dfd + kwargs.get('w_endpoint', 0.4) * r_endpoint
                         negative_penalty = (w_jerk * p_jerk + w_feas * p_feas) 
                    elif reward_type == 'quality_only':
                        positive_reward = 0 
                        negative_penalty = (w_jerk * p_jerk + w_feas * p_feas)
                    else:
                        print(f"error reward_type '{reward_type}'.")
                        positive_reward = (w_dfd * r_dfd +
                                           w_hd * r_hd +
                                           w_rmse * r_rmse +
                                           w_endpoint * r_endpoint)
                        negative_penalty = (w_jerk * p_jerk +
                                            w_feas * p_feas)

                    if p_feas > 0:
                         reward = -w_feas 
                    else:
                         reward = positive_reward - negative_penalty
                         reward = max(-1.0, min(1.0, reward)) 

                else:
                    reward = -1.0 
                    if pred_traj is None: log_data["error"] = log_data.get("error", "error")
                    if sol_gt_traj is None: log_data["error"] = log_data.get("error", "error")


            except Exception as e:
                reward = -1.0 
                log_data["error"] = f"error: {str(e)}"

            rewards_list.append(reward)
            log_data["final_reward"] = reward

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                if log_path:
                    try:
                        with open(log_path, "a", encoding='utf-8') as f:
                            f.write(f"------------- {current_time} Trajectory Reward Log -------------\n")
                            f.write(f"Content: {log_data.get('content', 'N/A')}\n")
                            pred_str = np.array2string(log_data['pred_traj'], precision=3, separator=',') if log_data.get('pred_traj') is not None else 'None'
                            gt_str = np.array2string(log_data['gt_traj'], precision=3, separator=',') if log_data.get('gt_traj') is not None else 'None'
                            f.write(f"Predicted: {pred_str}\n")
                            f.write(f"Ground Truth: {gt_str}\n")
                            if log_data.get("error"):
                                f.write(f"Error: {log_data['error']}\n")
                            else:
                                f.write(f"Reward Type: {reward_type}\n")
                                f.write(f"  R_DFD: {log_data.get('r_dfd', 'N/A'):.4f} (w={w_dfd if reward_type=='all' else 'N/A'})\n")
                                f.write(f"  R_HD:  {log_data.get('r_hd', 'N/A'):.4f} (w={w_hd if reward_type=='all' else 'N/A'})\n")
                                f.write(f"  R_RMSE:{log_data.get('r_rmse', 'N/A'):.4f} (w={w_rmse if reward_type=='all' else 'N/A'})\n")
                                f.write(f"  R_End: {log_data.get('r_endpoint', 'N/A'):.4f} (w={w_endpoint if reward_type=='all' else 'N/A'})\n")
                                f.write(f"  P_Jerk:{log_data.get('p_jerk', 'N/A'):.4f} (w={w_jerk if reward_type=='all' else 'N/A'})\n")
                                f.write(f"  P_Feas:{log_data.get('p_feas', 'N/A'):.4f} (w={w_feas if reward_type=='all' else 'N/A'})\n")
                                f.write(f"  -> Final Reward: {log_data['final_reward']:.4f}\n")
                            f.write("-" * 60 + "\n\n")
                    except Exception as log_e:
                        print(f"error: {log_e}") 
        return rewards_list
