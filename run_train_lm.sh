export CUDA_VISIBLE_DEVICES=5,6,7,8


accelerate launch --config_file accelerate_config2.yaml train_lm.py \
    --qa_model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --train_data_path data/augmented_data/generated_dataset_1000.json \
    --eval_data_path data/augmented_data/generated_dataset.json \
    --do_train \
    --do_eval \
    --evaluation_strategy "steps" \
    --output_dir saved_language_models/ \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --model_max_length 1024 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-5 \
    --logging_steps 1 \
    --fp16 \
    --weight_decay 0 \
    --seed 0 \
    --report_to wandb \
    --run_name "11711-RAG" \
    --warmup_steps 100 \
    #--fsdp "full_shard auto_wrap" \
    #--fsdp_transformer_layer_cls_to_wrap PhiDecoderLayer \