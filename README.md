# ManipLVM-R1: Reinforcement Learning for Reasoning in Embodied Manipulation with Large Vision-Language Model

<font size=4><div align='center' > [[🤗 Checkpoints](https://huggingface.co/JJJJinx/Qwen_SFT)]  </div></font>

## 🛠️ Setup

```bash
conda create -n ManipLVM-R1 python=3.10
conda activate ManipLVM-R1
bash setup.sh
```

## 💪🏻 Training

### Referring Expression Comprehension (Affordance)

#### 📚 GRPO

1. Download the Dataset (created from ShareRobot data) and we refer to the image dir as `<your_image_root>`.
2. Write the path of the annotation files in the `src/open-r1-multimodal/data_config/.yaml` file.

```bash
datasets:
    - json_path: /path/to/affordance_train.json
```

4. ``bash src/open-r1-multimodal/run_scripts/run_affordance.sh``

> [!NOTE]
> If you encounter 'CUDA out of memory' error, you can try to (1) set `gradient_checkpointing` as `true`, (2) reduce the `per_device_train_batch_size`, or (3) use lora.

```bash
cd src/open-r1-multimodal

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_affordance.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name data_config/affordance.yaml \
    --image_root <your_image_root> \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --freeze_vision_modules false # If you want to only finetune the language model, set this to true.
```
