export CUDA_VISIBLE_DEVICES=8,9


accelerate launch train_lm.py \
    --qa_model_name_or_path microsoft/phi-2 \
    --train_data_path data/databrick_qa/databrick_train.json \
    --eval_data_path data/databrick_qa/databrick_test.json \
    --do_train \
    --do_eval \
    --evaluation_strategy "steps" \
    --output_dir saved_language_models/ \
    --num_train_epochs 1 \
    --max_steps 20 \
    --per_device_train_batch_size 2 \
    --per_device_train_batch_size 8 \
    --model_max_length 2048 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --logging_steps 5 \
    --warmup_steps 100 \
    --fp16 \
    --weight_decay 0 \
    --seed 0 \
    --report_to wandb \
    --run_name "11711-RAG" \
    #--fsdp "full_shard auto_wrap" \
    #--fsdp_transformer_layer_cls_to_wrap PhiDecoderLayer \