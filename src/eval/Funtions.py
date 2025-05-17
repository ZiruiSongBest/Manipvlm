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
        # 首先匹配answer部分
        answer_pattern = r"<think>.*?</think>\s*<answer>(.*?)</answer>"
        # 然后在answer内容中匹配坐标点
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
    

    # --- 主奖励函数 ---
    @staticmethod
    def trajectory_reward_interp(completions, solution, **kwargs):
        """
        根据预测轨迹 (completions) 和真实轨迹 (solution) 计算奖励。

        Args:
            completions (list): 模型生成的预测结果列表，每个元素包含预测文本。
                            预期格式: [{'content': '<answer>[[x,y],...]</answer>'}]
            solution (list): 对应的真实轨迹列表。每个元素应为 Numpy 数组

            **kwargs: 可选参数，用于配置奖励函数。
                reward_type (str): 奖励计算方式 ('dtw_endpoint', 'frechet_endpoint', 'endpoint_only')。
                                    默认为 'dtw_endpoint'。
                w_endpoint (float): 终点奖励的权重。
                w_similarity (float): 轨迹相似性奖励 (DTW或Frechet) 的权重。
                k_endpoint (float): 终点奖励距离衰减系数。
                k_dtw (float): DTW奖励距离衰减系数。
                k_frechet (float): Frechet奖励距离衰减系数。

        Returns:
            list: 包含每个预测对应的奖励分数的列表。
        """
        # --- Helper functions ---
        # Get current time for logging
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def parse_trajectory_2d(text):
            """
            从文本中解析2D轨迹坐标，支持两种格式：
            1. 方括号包围: [[x1, y1], [x2, y2],...]
            2. 花括号包围: {[x1, y1], [x2, y2],...}
            
            返回一个Numpy数组 (N, 2)，如果解析失败则返回 None。
            """
            try:
                # 使用更通用的正则表达式，同时匹配方括号和花括号包围的列表
                main_list_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
                # 单独匹配每个坐标点
                point_pattern = r'\[\s*(-?\d*\.?\d+(?:e[-+]?\d+)?)\s*,\s*(-?\d*\.?\d+(?:e[-+]?\d+)?)\s*\]'
                if main_list_match:
                    points_str = main_list_match.group(1)
                    # 找出所有坐标点
                    point_matches = re.findall(point_pattern, points_str)
                    # 确保有3-10个坐标点
                    if point_matches and 3 <= len(point_matches) <= 10:
                        # 将提取的元组转换为数值列表
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
            计算基于终点欧几里得距离的奖励。
            奖励 = exp(-k * distance^2)，值域 ，距离越小奖励越高。
            """
            try:
                pred_end = pred_traj[-1]
                gt_end = gt_traj[-1]
                endpoint_dist = np.linalg.norm(pred_end - gt_end) # 计算欧氏距离
                reward = np.exp(-k_endpoint * endpoint_dist**2) # 指数衰减函数 [2]    
                return reward
            except Exception as e:
                print(f"计算终点奖励时发生错误: {e}")
                return 0.0

        def calculate_dtw_reward(pred_traj, gt_traj, k_dtw=0.1):
            """
            计算基于DTW距离的奖励。
            奖励 = exp(-k * dtw_distance)，值域 ，距离越小奖励越高。
            需要 dtw-python 库。
            """
            # DTW 至少需要两个点来计算
            try:
                # 使用 dtw-python 库计算DTW距离 [12, 13]
                # alignment.distance 是未归一化的DTW距离
                dtw_distance, _ = fastdtw(pred_traj, gt_traj, dist=euclidean)
                reward = np.exp(-k_dtw * dtw_distance)  # 指数衰减
                return reward
            except NotImplementedError:
                print("DTW 计算跳过: 依赖库 'dtw-python' 未安装。")
                return 0.0
            except Exception as e:
                print(f"计算DTW奖励时发生错误: {e}")
                return 0.0

        def calculate_frechet_reward(pred_traj, gt_traj, k_frechet=0.1):
            """
            计算基于离散Frechet距离的奖励。
            奖励 = exp(-k * frechet_distance)，值域 ，距离越小奖励越高。
            需要 similaritymeasures 库。
            """
            if pred_traj is None or gt_traj is None or len(pred_traj) == 0 or len(gt_traj) == 0:
                return 0.0
            try:
                # 使用 similaritymeasures 库计算离散Frechet距离
                # 该库支持N维数据
                frechet_dist_val, _ = similaritymeasures.frechet_dist(pred_traj, gt_traj) 
                reward = np.exp(-k_frechet * frechet_dist_val) # 指数衰减
                return reward
            except NotImplementedError:
                print("Frechet 计算跳过: 依赖库 'similaritymeasures' 未安装。")
                return 0.0
            except Exception as e:
                return 0.0
        
        rewards_list = []
        
        # 从 completions 提取预测内容
            # 假设 completions 是 [[{'content':...}], [{'content':...}],...] 的结构
        contents = [completion[0]["content"] for completion in completions]
        # --- 配置奖励函数参数 ---
        # 从kwargs获取参数，如果未提供则使用默认值
        reward_type = kwargs.get('reward_type', 'dtw_endpoint') 
        # 权重 (可以调整以平衡不同目标的重要性) [15, 2, 16, 17, 18, 19]
        w_endpoint = kwargs.get('w_endpoint', 0.4)      # 终点准确性权重
        w_similarity = kwargs.get('w_similarity', 0.6)  # 轨迹相似性权重 (DTW 或 Frechet)
        # 衰减系数 (控制奖励随距离下降的速度)
        k_endpoint = kwargs.get('k_endpoint', 1.0)      
        k_dtw = kwargs.get('k_dtw', 0.1)                
        k_frechet = kwargs.get('k_frechet', 0.1)          

        # 用于从预测内容中提取答案的正则表达式
        # answer_tag_pattern = r'<answer>(.*?)</answer>'

        # --- 遍历每个预测和对应的真实解 ---
        for content, sol_gt_traj in zip(contents, solution):
            reward = 0.0
            pred_traj = None
            
            try:
                pred_traj = parse_trajectory_2d(content)
                # 2. 验证真实轨迹格式
                if not isinstance(sol_gt_traj, np.ndarray):
                    # 尝试将 solution 中的元素转换为 Numpy 数组
                    sol_gt_traj = np.array(sol_gt_traj, dtype=float)

                # 3. 计算奖励 (仅当预测和真实轨迹都有效时)
                if pred_traj is not None and sol_gt_traj is not None:
                    # 计算基础奖励组件
                    r_end = calculate_endpoint_reward(pred_traj, sol_gt_traj, k_endpoint)
                    
                    # 根据 reward_type 计算最终奖励
                    if reward_type == 'dtw_endpoint':
                        r_sim = calculate_dtw_reward(pred_traj, sol_gt_traj, k_dtw)
                        reward = w_endpoint * r_end + w_similarity * r_sim
                    elif reward_type == 'frechet_endpoint':
                        r_sim = calculate_frechet_reward(pred_traj, sol_gt_traj, k_frechet)
                        reward = w_endpoint * r_end + w_similarity * r_sim
                    elif reward_type == 'endpoint_only':
                        reward = r_end # 只考虑终点准确性
                    # 可以添加更多奖励组合，例如仅DTW或仅Frechet
                    elif reward_type == 'dtw_only':
                        reward = calculate_dtw_reward(pred_traj, sol_gt_traj, k_dtw)
                    elif reward_type == 'frechet_only':
                        reward = calculate_frechet_reward(pred_traj, sol_gt_traj, k_frechet)
                    else:
                        print(f"警告: 未知的 reward_type '{reward_type}'。将默认使用 'dtw_endpoint'。")
                        r_sim = calculate_dtw_reward(pred_traj, sol_gt_traj, k_dtw)
                        reward = w_endpoint * r_end + w_similarity * r_sim
                        
                    # 确保奖励值在  范围内 (对于指数衰减和正权重，通常是这样，但以防万一)
                    reward = max(0.0, min(1.0, reward)) 

                else:
                    reward = 0.0 # 如果无法计算，则奖励为0

                # 添加调试日志
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
                reward = 0.0 # 出错时奖励为0
                
                # 记录错误到日志
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    if log_path:
                        with open(log_path, "a", encoding='utf-8') as f:
                            f.write(f"------------- {current_time} Trajectory reward ERROR -------------\n")
                            f.write(f"Content: {content}\n")
                            f.write(f"Error: {str(e)}\n\n")

            rewards_list.append(reward)
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
        从文本 <answer>...</answer> 标签中解析2D轨迹坐标。
        支持 [[x, y], ...] 格式。
        返回一个 Numpy 数组 (N, 2) 包含 3 到 10 个点，如果解析失败或点数无效则返回 None。
        """
        try:
            # 提取 <answer> 标签内的内容
            main_list_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
            if not main_list_match:
                # print("Debug: No <answer> tag found.")
                return None
            
            content_inside_answer = main_list_match.group(1).strip()
            # print(f"Debug: Content inside answer: {content_inside_answer}")

            # 使用正则表达式严格匹配 [[d+,d+],[d+,d+],...] 格式
            # 允许点与点之间以及括号内的数字之间有空格
            # 确保整个内容是方括号包围的列表
            point_pattern = r'\[\s*(-?\d*\.?\d+(?:e[-+]?\d+)?)\s*,\s*(-?\d*\.?\d+(?:e[-+]?\d+)?)\s*\]'
            point_matches = re.findall(point_pattern, content_inside_answer)    
            if point_matches and 3 <= len(point_matches) <= 10:
                # 将提取的元组转换为数值列表
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
        使用弧长参数化将轨迹重采样到固定数量的点。

        Args:
            traj (np.ndarray): 原始轨迹 (N, 2)。
            num_points (int): 重采样后的点数。

        Returns:
            np.ndarray or None: 重采样后的轨迹 (num_points, 2)，如果失败则返回 None。
        """
        if traj is None or len(traj) < 2:
            return None
        try:
            # 1. 计算段长度
            segment_lengths = np.linalg.norm(np.diff(traj, axis=0), axis=1)
            # 2. 计算累积弧长 (参数 s)
            cumulative_arc_length = np.concatenate(([0], np.cumsum(segment_lengths)))
            total_length = cumulative_arc_length[-1]

            if total_length <= 1e-6: # 避免除零或长度为零的情况
                # 如果轨迹长度几乎为零，返回重复的第一个点
                return np.tile(traj[0], (num_points, 1))

            # 3. 创建插值函数: s -> t (这里 t 是原始参数，近似为索引)
            # 我们需要的是 s -> (x, y)，可以通过 s -> t -> (x, y) 实现
            # 或者直接插值 x(s) 和 y(s)
            # 使用累积弧长作为插值点
            f_x = interp1d(cumulative_arc_length, traj[:, 0], kind='linear', fill_value="extrapolate")
            f_y = interp1d(cumulative_arc_length, traj[:, 1], kind='linear', fill_value="extrapolate")

            # 4. 生成均匀分布的弧长采样点
            s_resampled = np.linspace(0, total_length, num_points)

            # 5. 插值得到重采样后的坐标
            x_resampled = f_x(s_resampled)
            y_resampled = f_y(s_resampled)

            return np.vstack((x_resampled, y_resampled)).T
        except Exception as e:
            # print(f"重采样时出错: {e}")
            return None

    # --- Helper: Reward/Penalty Calculations ---
    @staticmethod
    def calculate_endpoint_reward(pred_traj, gt_traj, k_endpoint=1.0):
        """计算基于终点欧几里得距离的奖励 (越高越好)。"""
        if pred_traj is None or gt_traj is None or len(pred_traj) < 1 or len(gt_traj) < 1:
            return 0.0
        try:
            pred_end = pred_traj[-1]
            gt_end = gt_traj[-1]
            endpoint_dist = np.linalg.norm(pred_end - gt_end)
            # 使用指数衰减将距离转换为 [0, 1] 范围的奖励
            reward = np.exp(-k_endpoint * endpoint_dist**2)
            return reward
        except Exception as e:
            # print(f"计算终点奖励时出错: {e}")
            return 0.0

    @staticmethod
    def _metric_to_reward(distance, k_factor):
        """通用函数，将距离转换为 [0, 1] 的奖励。"""
        if distance is None or distance < 0:
            return 0.0
        # 指数衰减: exp(-k * distance)
        # k_factor 控制衰减速度，k 越大，对距离增加越敏感
        reward = np.exp(-k_factor * distance)
        return reward

    @staticmethod
    def calculate_dtw_reward(pred_traj, gt_traj, k_dtw=0.1):
        """计算基于 FastDTW 距离的奖励 (越高越好)。"""
        if pred_traj is None or gt_traj is None or len(pred_traj) < 2 or len(gt_traj) < 2:
             # print("DTW 跳过: 轨迹点数不足 (<2)。")
             return 0.0
        try:
            dtw_distance, _ = fastdtw(pred_traj, gt_traj, dist=euclidean)
            return TrajectoryRewardCalculator._metric_to_reward(dtw_distance, k_dtw)
        except ImportError:
            print("DTW 计算跳过: 依赖库 'fastdtw' 未安装。")
            return 0.0
        except Exception as e:
            # print(f"计算DTW奖励时出错: {e}")
            return 0.0

    @staticmethod
    def calculate_frechet_reward(pred_traj, gt_traj, k_frechet=0.1):
        """计算基于离散 Frechet 距离的奖励 (越高越好)。"""
        if pred_traj is None or gt_traj is None or len(pred_traj) < 3 or len(gt_traj) < 3:
            return 0.0
        try:
            frechet_dist_val = similaritymeasures.frechet_dist(pred_traj, gt_traj)
            return TrajectoryRewardCalculator._metric_to_reward(frechet_dist_val, k_frechet)
        except ImportError:
            print("Frechet 计算跳过: 依赖库 'similaritymeasures' 未安装。")
            return 0.0
        except Exception as e:
            # print(f"计算Frechet奖励时出错: {e}")
            return 0.0

    @staticmethod
    def calculate_hausdorff_reward(pred_traj, gt_traj, k_hd=0.1):
        """计算基于 Hausdorff 距离的奖励 (越高越好)。"""
        if pred_traj is None or gt_traj is None or len(pred_traj) < 1 or len(gt_traj) < 1:
            return 0.0
        try:
            # 使用 scipy 计算 directed Hausdorff distance, 然后取最大值
            dist_pred_to_gt = directed_hausdorff(pred_traj, gt_traj)[0]
            dist_gt_to_pred = directed_hausdorff(gt_traj, pred_traj)[0]
            hausdorff_dist = max(dist_pred_to_gt, dist_gt_to_pred)
            return TrajectoryRewardCalculator._metric_to_reward(hausdorff_dist, k_hd)
        except ImportError:
            print("Hausdorff 计算跳过: 依赖库 'scipy' 未安装或版本过低。")
            return 0.0
        except Exception as e:
            # print(f"计算Hausdorff奖励时出错: {e}")
            return 0.0

    @staticmethod
    def calculate_rmse_reward(pred_traj, gt_traj, k_rmse=0.1, num_resample_points=20):
        """计算基于重采样后 RMSE 的奖励 (越高越好)。"""
        if pred_traj is None or gt_traj is None or len(pred_traj) < 2 or len(gt_traj) < 2:
             # print("RMSE 跳过: 原始轨迹点数不足 (<2)。")
             return 0.0
        try:
            pred_resampled = TrajectoryRewardCalculator.resample_trajectory_by_arc_length(pred_traj, num_resample_points)
            gt_resampled = TrajectoryRewardCalculator.resample_trajectory_by_arc_length(gt_traj, num_resample_points)

            if pred_resampled is None or gt_resampled is None:
                 # print("RMSE 跳过: 重采样失败。")
                 return 0.0
            
            # 计算点对点欧氏距离的平方
            squared_errors = np.sum((pred_resampled - gt_resampled)**2, axis=1)
            # 计算均方根误差
            rmse = np.sqrt(np.mean(squared_errors))
            return TrajectoryRewardCalculator._metric_to_reward(rmse, k_rmse)
        except ImportError:
            print("RMSE 计算跳过: 依赖库 'scipy' 未安装或版本过低。")
            return 0.0
        except Exception as e:
            # print(f"计算RMSE奖励时出错: {e}")
            return 0.0

    @staticmethod
    def calculate_jerk_penalty(pred_traj, dt=0.1):
        """计算平均平方Jerk惩罚项 (越低越好)。"""
        if pred_traj is None or len(pred_traj) < 4:
            # Jerk需要至少4个点 (位置->速度->加速度->Jerk)
            return 0.0 # 或者返回一个大的惩罚值，如果轨迹太短是不希望的
        try:
            # 假设 dt 为恒定时间步长
            # 速度 v = diff(p) / dt
            velocity = np.diff(pred_traj, axis=0) / dt
            # 加速度 a = diff(v) / dt = diff(p, 2) / dt^2
            acceleration = np.diff(velocity, axis=0) / dt
            # Jerk j = diff(a) / dt = diff(p, 3) / dt^3
            jerk = np.diff(acceleration, axis=0) / dt

            # 计算平方Jerk的大小
            squared_jerk_magnitude = np.sum(jerk**2, axis=1)
            # 返回平均平方Jerk，作为惩罚值
            mean_squared_jerk = np.mean(squared_jerk_magnitude)

            # 可以考虑归一化，例如除以一个预期的最大Jerk平方值，使惩罚在 [0, ~1] 范围
            # normalized_penalty = mean_squared_jerk / (expected_max_jerk**2)
            # return normalized_penalty

            # 暂时返回未归一化的值，权重 w_jerk 将控制其影响
            return mean_squared_jerk
        except Exception as e:
            # print(f"计算Jerk惩罚时出错: {e}")
            return 0.0 # 或一个大惩罚值

    @staticmethod
    def calculate_feasibility_penalty(pred_traj, joint_limits=None, collision_checker=None, dt=0.1):
        """
        计算可行性惩罚 (0 表示可行, >0 表示不可行)。
        检查关节限制和碰撞。
        """
        penalty = 0.0
        if pred_traj is None or len(pred_traj) < 1:
            return 1.0 # 惩罚无效轨迹

        try:
            # --- 1. 检查关节限制 (需要计算速度/加速度/Jerk) ---
            if joint_limits:
                # 计算导数 (如果需要检查速度/加速度/Jerk限制)
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

                # 检查每个点的限制
                for i, pos in enumerate(pred_traj):
                    # 位置限制
                    if 'position' in joint_limits:
                        if np.any(pos < joint_limits['position']['min']) or np.any(pos > joint_limits['position']['max']):
                            penalty = 1.0
                            break
                    # 速度限制 (检查点 i 对应的速度段 i-1)
                    if has_vel_limits and velocity is not None and i > 0:
                         # 检查速度的大小，或者每个分量
                         # vel_mag = np.linalg.norm(velocity[i-1])
                         # if vel_mag > joint_limits['velocity']['max_magnitude']:
                         if np.any(np.abs(velocity[i-1]) > joint_limits['velocity']['max']):
                             penalty = 1.0
                             break
                    # 加速度限制 (检查点 i 对应的加速度段 i-2)
                    if has_acc_limits and acceleration is not None and i > 1:
                         if np.any(np.abs(acceleration[i-2]) > joint_limits['acceleration']['max']):
                             penalty = 1.0
                             break
                    # Jerk 限制 (检查点 i 对应的 Jerk 段 i-3)
                    if has_jerk_limits and jerk is not None and i > 2:
                         if np.any(np.abs(jerk[i-3]) > joint_limits['jerk']['max']):
                             penalty = 1.0
                             break
                
                if penalty > 0:
                    # print("Debug: 关节限制违规")
                    return penalty # 如果违反限制，直接返回惩罚

            # --- 2. 检查碰撞 ---
            if collision_checker:
                # 可以检查每个点，或者更常用的是检查段
                # 检查段更安全，但计算成本更高
                # 这里简化为检查每个点
                for q in pred_traj:
                    if collision_checker(q): # 假设 checker 返回 True 如果碰撞
                        penalty = 1.0
                        # print("Debug: 检测到碰撞")
                        break # 发现碰撞就停止检查

            return penalty

        except Exception as e:
            # print(f"检查可行性时出错: {e}")
            return 1.0 # 出错时假设不可行

    # --- 主奖励函数 ---
    @staticmethod
    def trajectory_reward_comprehensive(completions, solution, **kwargs):
        """
        根据预测轨迹 (completions) 和真实轨迹 (solution) 计算综合奖励。
        包含 DFD, HD, RMSE, Endpoint, Smoothness(Jerk), Feasibility。

        Args:
            completions (list): 模型生成的预测结果列表, e.g., [[{'content': '<answer>...</answer>'}]]
            solution (list): 对应的真实轨迹列表 (numpy 数组或可转换为数组的列表)。
            **kwargs: 配置参数:
                reward_type (str): 定义使用哪些组件 ('all', 'similarity_endpoint', 'dtw_endpoint', 'frechet_endpoint', 'quality_only', ...)
                num_resample_points (int): RMSE 重采样点数，默认 20。
                dt (float): 计算导数(Jerk等)时假设的时间步长，默认 0.1。
                # Similarity & Endpoint Weights and Scaling Factors
                w_dfd (float): DFD 奖励权重, default 0.25
                w_hd (float): HD 奖励权重, default 0.15
                w_rmse (float): RMSE 奖励权重, default 0.20
                w_endpoint (float): Endpoint 奖励权重, default 0.40
                k_dfd (float): DFD 距离衰减系数, default 0.1
                k_hd (float): HD 距离衰减系数, default 0.1
                k_rmse (float): RMSE 距离衰减系数, default 0.5
                k_endpoint (float): Endpoint 距离衰减系数, default 1.0
                # Quality Weights and Parameters
                w_jerk (float): Jerk 惩罚权重, default 0.1
                w_feas (float): Feasibility 惩罚权重, default 10.0 (使其成为强惩罚)
                joint_limits (dict): 关节限制字典, e.g., 
                                        {'position': {'min': [-1,-1], 'max': [1,1]}, 
                                        'velocity': {'max': [10,10]}}
                collision_checker (callable): 碰撞检查函数 checker(q) -> bool (True if collision)

        Returns:
            list: 包含每个预测对应的综合奖励分数的列表。
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rewards_list = []

        # --- 配置奖励函数参数 ---
        reward_type = kwargs.get('reward_type', 'all') # 默认使用所有指标
        num_resample_points = kwargs.get('num_resample_points', 20)
        dt = kwargs.get('dt', 0.1) # 假设的时间步长

        # 相似性与终点权重 (正向奖励) - 确保它们加起来 <= 1 (如果需要)
        w_dfd = kwargs.get('w_dfd', 0.25)
        w_hd = kwargs.get('w_hd', 0.15)
        w_rmse = kwargs.get('w_rmse', 0.20)
        w_endpoint = kwargs.get('w_endpoint', 0.40)
        # 检查权重和是否合理 (可选)
        # total_positive_weight = w_dfd + w_hd + w_rmse + w_endpoint
        # if not np.isclose(total_positive_weight, 1.0) and reward_type == 'all':
        #      print(f"警告: 正向奖励权重和 ({total_positive_weight}) 不为 1.0")
        
        # 距离衰减系数
        k_dfd = kwargs.get('k_dfd', 0.1)
        k_hd = kwargs.get('k_hd', 0.1)
        k_rmse = kwargs.get('k_rmse', 0.5) # RMSE 通常值域较大，衰减快些
        k_endpoint = kwargs.get('k_endpoint', 1.0) # 终点距离平方衰减，k可以大些

        # 质量指标权重 (负向惩罚)
        w_jerk = kwargs.get('w_jerk', 0.0) # 需要根据Jerk值的范围调整
        w_feas = kwargs.get('w_feas', 0.0) # 可行性是硬约束，给高权重

        # 可行性检查参数
        joint_limits = kwargs.get('joint_limits', None)
        collision_checker = kwargs.get('collision_checker', None) # 外部传入检查函数

        # 提取预测内容
        try:
             contents = [completion[0]["content"] for completion in completions]
        except (TypeError, IndexError, KeyError):
             print("错误: 'completions' 格式不符合预期 [[{'content': ...}]]")
             # 返回与 completions 同样数量的 0 奖励
             return [0.0] * len(completions)


        # --- 遍历计算 ---
        for content, sol_gt_traj_orig in zip(contents, solution):
            pred_traj = None
            sol_gt_traj = None
            log_data = {"content": content, "error": None}

            try:
                # 1. 解析预测轨迹
                pred_traj = TrajectoryRewardCalculator.parse_trajectory_2d(content)
                log_data["pred_traj"] = pred_traj

                # 2. 验证/转换真实轨迹
                if isinstance(sol_gt_traj_orig, np.ndarray):
                    sol_gt_traj = sol_gt_traj_orig.astype(float)
                else:
                    try:
                        sol_gt_traj = np.array(sol_gt_traj_orig, dtype=float)
                        if sol_gt_traj.ndim != 2 or sol_gt_traj.shape[1] != 2:
                             raise ValueError("真实轨迹必须是 (M, 2) 形状")
                    except Exception as e:
                        log_data["error"] = f"真实轨迹转换失败: {e}"
                        sol_gt_traj = None # 标记 GT 无效
                log_data["gt_traj"] = sol_gt_traj

                # 3. 计算奖励 (仅当预测和真实轨迹都有效时)
                reward = 0.0
                positive_reward = 0.0
                negative_penalty = 0.0

                if pred_traj is not None and sol_gt_traj is not None:
                    # --- 计算正向奖励组件 ---
                    r_dfd = TrajectoryRewardCalculator.calculate_frechet_reward(pred_traj, sol_gt_traj, k_dfd)
                    r_hd = TrajectoryRewardCalculator.calculate_hausdorff_reward(pred_traj, sol_gt_traj, k_hd)
                    r_rmse = TrajectoryRewardCalculator.calculate_rmse_reward(pred_traj, sol_gt_traj, k_rmse, num_resample_points)
                    r_endpoint = TrajectoryRewardCalculator.calculate_endpoint_reward(pred_traj, sol_gt_traj, k_endpoint)
                    log_data.update({"r_dfd": r_dfd, "r_hd": r_hd, "r_rmse": r_rmse, "r_endpoint": r_endpoint})

                    # --- 计算负向惩罚组件 ---
                    p_jerk = TrajectoryRewardCalculator.calculate_jerk_penalty(pred_traj, dt)
                    p_feas = TrajectoryRewardCalculator.calculate_feasibility_penalty(pred_traj, joint_limits, collision_checker, dt)
                    log_data.update({"p_jerk": p_jerk, "p_feas": p_feas})

                    # --- 根据 reward_type 组合 ---
                    if reward_type == 'all':
                        positive_reward = (w_dfd * r_dfd +
                                           w_hd * r_hd +
                                           w_rmse * r_rmse +
                                           w_endpoint * r_endpoint)
                        negative_penalty = (w_jerk * p_jerk +
                                            w_feas * p_feas)
                    elif reward_type == 'similarity_endpoint': # DFD+HD+RMSE+Endpoint
                         positive_reward = (w_dfd * r_dfd + w_hd * r_hd + w_rmse * r_rmse + w_endpoint * r_endpoint)
                         # 可能需要调整权重和
                         negative_penalty = 0 # 或保留质量惩罚
                    elif reward_type == 'dtw_endpoint': # 兼容旧版
                        r_dtw = TrajectoryRewardCalculator.calculate_dtw_reward(pred_traj, sol_gt_traj, k_dtw) # 使用旧的k_dtw
                        log_data["r_dtw"] = r_dtw
                        positive_reward = kwargs.get('w_similarity', 0.6) * r_dtw + kwargs.get('w_endpoint', 0.4) * r_endpoint
                        negative_penalty = (w_jerk * p_jerk + w_feas * p_feas) # 也加上质量惩罚
                    elif reward_type == 'frechet_endpoint': # 兼容旧版
                         positive_reward = kwargs.get('w_similarity', 0.6) * r_dfd + kwargs.get('w_endpoint', 0.4) * r_endpoint
                         negative_penalty = (w_jerk * p_jerk + w_feas * p_feas) # 也加上质量惩罚
                    elif reward_type == 'quality_only':
                        positive_reward = 0 # 或者给一个小的基础分?
                        negative_penalty = (w_jerk * p_jerk + w_feas * p_feas)
                    # ... 可以添加更多自定义组合 ...
                    else:
                        print(f"警告: 未知的 reward_type '{reward_type}'。将默认使用 'all'。")
                        positive_reward = (w_dfd * r_dfd +
                                           w_hd * r_hd +
                                           w_rmse * r_rmse +
                                           w_endpoint * r_endpoint)
                        negative_penalty = (w_jerk * p_jerk +
                                            w_feas * p_feas)

                    # 计算最终奖励
                    # 如果可行性检查失败 (p_feas > 0)，可以考虑将奖励直接设为负值或0
                    if p_feas > 0:
                         reward = -w_feas # 或者 -1.0, 0.0? 强惩罚
                    else:
                         reward = positive_reward - negative_penalty
                         # 可以选择将奖励裁剪到 [0, 1] 或 [-1, 1] 范围
                         # reward = max(0.0, min(1.0, reward)) # 裁剪到 [0, 1]
                         reward = max(-1.0, min(1.0, reward)) # 裁剪到 [-1, 1]

                else:
                    # 解析失败或GT无效，奖励为负值（或0）
                    reward = -1.0 # 表示失败
                    if pred_traj is None: log_data["error"] = log_data.get("error", "") + "预测轨迹解析失败. "
                    if sol_gt_traj is None: log_data["error"] = log_data.get("error", "") + "真实轨迹无效."


            except Exception as e:
                reward = -1.0 # 出错时给最低奖励
                log_data["error"] = f"奖励计算中发生意外错误: {str(e)}"

            rewards_list.append(reward)
            log_data["final_reward"] = reward

            # --- 调试日志 ---
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
                        print(f"写入日志时出错: {log_e}") # 避免日志错误中断主流程

        return rewards_list

# --- 示例用法 ---
if __name__ == '__main__':
    # 模拟数据
    completions_data = [
        [{'content': 'Some text before <answer>[[1, 1], [2, 2], [3, 3], [4, 4]]</answer> some text after'}],
        [{'content': '<answer>[[1, 0], [2, 1], [3, 0], [3, -1]]</answer>'}],
        [{'content': '<answer>[[1, 1], [10, 10]]</answer>'}], # 点数太少
        [{'content': '<answer>Not a trajectory</answer>'}] , # 格式错误
        [{'content': '<answer>[[1.1, 1.1], [2.1, 2.2], [3.5, 3.4], [4.1, 4.0]]</answer>'}] # 包含浮点数
    ]
    solution_data = [
        np.array([[1, 1], [2.1, 2.1], [3, 3.1], [4.2, 3.9]]), # 正常 GT
        np.array([[1, 0], [2, 1], [3, 0], [3, -1], [3, -2]]), # 正常 GT (长度不同)
        np.array([[1, 1], [10, 10], [11,11]]), # 正常 GT
        np.array([[0, 0], [1, 1], [2, 2]]), # 正常 GT
        np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), # 正常 GT for float test
    ]

    # 设置环境变量用于调试日志
    os.environ["DEBUG_MODE"] = "true"
    log_file = "trajectory_reward_log.txt"
    if os.path.exists(log_file): os.remove(log_file) # 清除旧日志
    os.environ["LOG_PATH"] = log_file

    # 定义碰撞检查器 (示例)
    def simple_collision_checker(q):
        # 简单的边界检查作为示例
        x, y = q
        if not (0 <= x <= 10 and 0 <= y <= 10):
             # print(f"Debug Collision: Point {q} out of bounds [0,10]")
             return True # 发生碰撞 (出界)
        return False # 无碰撞

    # 定义关节限制 (示例) - 假设2D位置没有速度/加速度限制
    example_joint_limits = {
        'position': {'min': [0, 0], 'max': [10, 10]}
        # 'velocity': {'max': [5.0, 5.0]}, # 可选
        # 'acceleration': {'max': [2.0, 2.0]}, # 可选
        # 'jerk': {'max': [10.0, 10.0]} # 可选
    }

    calculator = TrajectoryRewardCalculator()

    # 测试不同的 reward_type 和参数
    print("--- Testing reward_type='all' ---")
    rewards_all = calculator.trajectory_reward_comprehensive(
        completions_data,
        solution_data,
        reward_type='all',
        num_resample_points=15,
        dt=0.1,
        # 默认权重
        collision_checker=simple_collision_checker,
        joint_limits=example_joint_limits
    )
    print(f"Rewards ('all'): {rewards_all}")

    print("\n--- Testing reward_type='dtw_endpoint' (old style) ---")
    rewards_dtw_ep = calculator.trajectory_reward_comprehensive(
        completions_data,
        solution_data,
        reward_type='dtw_endpoint', # 使用旧的类型
        w_endpoint=0.3, # 传递旧的权重 (示例)
        w_similarity=0.7,
        k_dtw=0.05,
        k_endpoint=0.8,
         collision_checker=simple_collision_checker, # 仍然可以应用质量惩罚
         joint_limits=example_joint_limits,
         w_jerk=0.05, # 也可以为旧模式配置质量惩罚权重
         w_feas=5.0
    )
    print(f"Rewards ('dtw_endpoint'): {rewards_dtw_ep}")
    
    print("\n--- Testing reward_type='similarity_endpoint' (DFD+HD+RMSE+End) ---")
    rewards_sim_ep = calculator.trajectory_reward_comprehensive(
        completions_data,
        solution_data,
        reward_type='similarity_endpoint', 
        w_dfd=0.3, w_hd=0.1, w_rmse=0.2, w_endpoint=0.4, # 调整权重
        k_dfd=0.08, k_hd=0.08, k_rmse=0.4, k_endpoint=0.9
        # 不包括质量惩罚 (根据类型定义)
    )
    print(f"Rewards ('similarity_endpoint'): {rewards_sim_ep}")

    print(f"\nDebug logs written to: {log_file}")
