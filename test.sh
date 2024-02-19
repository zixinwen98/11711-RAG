#will getting more answer help? 
CUDA_VISIBLE_DEVICES=7 python3 evaluate.py --chunk_size 500 --overlap 100 --retriever_topk 3
CUDA_VISIBLE_DEVICES=7 python3 evaluate.py --chunk_size 500 --overlap 100 --retriever_topk 5

#will smaller chunk but more answer help? 
CUDA_VISIBLE_DEVICES=7 python3 evaluate.py --chunk_size 250 --overlap 50 --retriever_topk 3
CUDA_VISIBLE_DEVICES=7 python3 evaluate.py --chunk_size 250 --overlap 50 --retriever_topk 10

#will larger chunk help? 
CUDA_VISIBLE_DEVICES=7 python3 evaluate.py --chunk_size 750 --overlap 150 --retriever_topk 3
CUDA_VISIBLE_DEVICES=7 python3 evaluate.py --chunk_size 750 --overlap 150 --retriever_topk 5