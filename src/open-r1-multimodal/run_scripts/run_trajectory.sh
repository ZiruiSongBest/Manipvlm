cd /mnt/pfs_l2/jieti_team/MMGroup/lzz/mbzuai_data/VLM-R1/src/open-r1-multimodal
NPROC=${1:-2}

export DEBUG_MODE="true"

RUN_NAME="Qwen2.5-VL-3B-GRPO-trajectory_ALL_NEW"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="$NPROC" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12722" \
    src/open_r1/grpo_tra.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name data_config/trajectory.yaml \
    --image_root /mnt/pfs_l2/jieti_team/MMGroup/lzz/mbzuai_data/VLM-R1/data/ShareRobot/trajectory/images \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true
    