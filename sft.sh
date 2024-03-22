# 動かない

eval "$(/home/yuiseki/miniconda3/bin/conda shell.bash hook)"
export PATH="/home/yuiseki/miniconda3/bin:$PATH"
conda activate peft

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 --main_process_port 1234 ~/TinyLlama/sft/finetune.py \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T \
    --output_dir ./output/tinyllama-sft-v1 \
    --logging_steps 10 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 6 \
    --evaluation_strategy epoch \
    --eval_dataset_size 0.2 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length=False \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --warmup_ratio 0.05 \
    --lr_scheduler_type constant \
    --dataset kunishou/oasst1-89k-ja \
    --dataset_format oasst1-ja \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 4 \
    --max_steps 0 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --seed 0 \
    --trust_remote_code \
    --report_to wandb 
