export CUDA_VISIBLE_DEVICES=8,9 

python evaluate.py \
    --doc_encoder_model_name_or_path avsolatorio/GIST-large-Embedding-v0 \
    --qa_model_name_or_path saved_language_models/mistralai/Mistral-7B-Instruct-v0.2 \
    --tokenizer_path mistralai/Mistral-7B-Instruct-v0.2 \
    --test_data_path "data/questions.json" \
    --chunk_size 250 --overlap 50 --retriever_topk 3 \
    --document_path "data/cmu/" \
    --result_path experiment/ \
    --max_length 2048 \
    #--qa_model_name_or_path saved_language_models/mistralai/Mistral-7B-Instruct-v0.2

python evaluate.py \
    --doc_encoder_model_name_or_path avsolatorio/GIST-large-Embedding-v0 \
    --qa_model_name_or_path saved_language_models/mistralai/Mistral-7B-Instruct-v0.2 \
    --tokenizer_path mistralai/Mistral-7B-Instruct-v0.2 \
    --test_data_path "data/automated_questions.json" \
    --chunk_size 250 --overlap 50 --retriever_topk 3 \
    --document_path "data/cmu/" \
    --result_path experiment/ \
    --max_length 2048 \