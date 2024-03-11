## Example to run inference (i.e. generate a file corresponding to required system output)

```
python inference.py \
    --doc_encoder_model_name_or_path avsolatorio/GIST-large-Embedding-v0\
    --qa_model_name_or_path microsoft/phi-2 \
    --test_data_path "data/questions.txt" \
    --chunk_size 250 --overlap 50 --retriever_topk 3 \
    --document_path "data/cmu_cleaned/" \
    --max_length 512 
```

unless specified otherwise, a .txt file will be created under 

`result/[model_name]/[qa model_document embedding model_chunksize_overlap_topk.txt]`

## Example to run evaluate (i.e. on our created test set

```
python evaluate.py \
    --doc_encoder_model_name_or_path avsolatorio/GIST-large-Embedding-v0\
    --qa_model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --test_data_path "data/automated_questions.json" \
    --chunk_size 250 --overlap 50 --retriever_topk 3 \
    --document_path "data/cmu_cleaned/" \
    --max_length 2048
```

unless specified otherwise, a .json file will be created under, with detailed analytical information for each sample, containing model answer, extracted context, sample f1, sample recall, sample exact match

`result/[model_name]/[qa model_document embedding model_chunksize_overlap_topk.json]`

## Here we breifly describe the usage of each file/folder

- 'data/': this contains all test sets/knowledge pile
- 'RAG_data_aug.py': generate trainset by few-shot prompting to mistral
- 'args.py': contain argument to all inference/evaluate files
- 'generate_dataset_from_tabular.py': generate dataset for tabular documents (see report for details)
- 'paired-bootstrap.py': a slightly modified version to do paired-boostrap on our own file format and f-1/recall/em score
