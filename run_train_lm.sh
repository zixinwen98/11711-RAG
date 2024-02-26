export CUDA_VISIBLE_DEVICES=6,7,8,9


accelerate launch --config_file accelerate_config.yaml --main_process_port 29501 train_lm.py \
    --qa_model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --train_data_path data/augmented_data/generated_dataset_1000.json \
    --eval_data_path data/augmented_data/generated_dataset.json \
    --do_train \
    --do_eval \
    --output_dir saved_language_models/ \
    --num_train_epochs 1 \
    --max_steps 200 \
    --per_device_train_batch_size 1 \
    --model_max_length 512 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --logging_steps 1 \
    --fp16 \
    --weight_decay 0 \
    --seed 0 \
    --report_to wandb \
    --run_name "11711-RAG" \
    --warmup_steps 50 \
    #--do_eval \
    #--evaluation_strategy "steps" \
    #--per_device_eval_batch_size 1 \
    
    #--fsdp "full_shard auto_wrap" \
    #--fsdp_transformer_layer_cls_to_wrap PhiDecoderLayer \