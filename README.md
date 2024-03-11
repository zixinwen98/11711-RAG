## Script to run inference 

```
python inference.py \
    --doc_encoder_model_name_or_path avsolatorio/GIST-large-Embedding-v0\
    --qa_model_name_or_path microsoft/phi-2 \
    --test_data_path "data/questions.txt" \
    --chunk_size 250 --overlap 50 --retriever_topk 3 \
    --document_path "data/cmu_cleaned/" \
    --max_length 512 
```

This script generates a file corresponding to required system output, unless specified otherwise, a .txt file will be created under 

`result/[model_name]/[qa model_document embedding model_chunksize_overlap_topk.txt]`

## Script to run evaluate 

```
python evaluate.py \
    --doc_encoder_model_name_or_path avsolatorio/GIST-large-Embedding-v0\
    --qa_model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --test_data_path "data/automated_questions.json" \
    --chunk_size 250 --overlap 50 --retriever_topk 3 \
    --document_path "data/cmu_cleaned/" \
    --use_reranker True\
    --max_length 2048
```

This scripts evaluates on our created test set, unless specified otherwise, a .json file will be created under, with detailed analytical information for each sample, containing model answer, extracted context, sample f1, sample recall, sample exact match

`result/[model_name]/[qa model_document embedding model_chunksize_overlap_topk.json]`

## Training the LM 

```
accelerate launch --config_file accelerate_config_deepspeed.yaml --main_process_port 29501 train_lm.py \
    --qa_model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --train_data_path data/augmented_data/generated_dataset_with_facts_5000.json \
    --eval_data_path data/augmented_data/generated_dataset_1000.json \
    --do_train \
    --do_eval \
    --output_dir saved_language_models/ \
    --num_train_epochs 1 \
    --max_steps 300 \
    --per_device_train_batch_size 1 \
    --model_max_length 1024 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --logging_steps 1 \
    --fp16 \
    --weight_decay 0 \
    --seed 0 \
    --report_to wandb \
    --run_name "11711-RAG" \
    --warmup_steps 50 \
```
Note that training a Mistral is very compute-expensive, we used 4x48G GPU

## Few-shot train example generation 
```
python RAG_data_aug.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --template_dataset_path data/questions.json \
    --source_dataset_path data/cmu/ \
    --extract_fact_from_context \
    --eval_f1_recall_of_facts_answers \
    --num_samples 5000 \
    --batch_size 2 \
```

## Usage of the files and folders.

- `data/`: this contains all test sets/knowledge pile
- `experiments/`: this contains some evaluation results on our human-annotated datasets.
- `inference.py`: this file is used to generate our submission given the test questions.
- `evaluate.py`: this file is for the evaluation of our RAG pipeline, it reports f-measure for each evaluation.
- `model.py`: this file contains the retriever model class.
- `dataset.py`: this file contains the training dataset for instruction finetuning, based on the Stanford-Alpaca implementation.
- `train_lm.py`: this file contains the instruction finetuning pipeline, based on the Stanford-Alpaca implementation.
- `RAG_data_aug.py`: this file generate training dataset by doing few-shot prompting to mistral and formatting the outputs.
- `args.py`: this file contain arguments to all files that uses the RAG pipeline.
- `generate_dataset_from_tabular.py`: this file is used to generate dataset for tabular documents (see report for details).
- `paired-bootstrap.py`: this is a slightly modified version to do paired-boostrap on our own file format and f-1/recall/em score.
