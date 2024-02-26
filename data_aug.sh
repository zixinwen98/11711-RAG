CUDA_VISIBLE_DEVICES=9 python RAG_data_aug.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --template_dataset_path data/questions.json \
    --source_dataset_path data/cmu/ \
    --extract_fact_from_context \
    --eval_f1_recall_of_facts_answers \
    --num_samples 5000 \
    --batch_size 2 \