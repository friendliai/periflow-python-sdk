NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=localhost
NPROC_PER_NODE=8

torchrun --nnodes $NUM_NODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port 6000 --nproc_per_node $NPROC_PER_NODE \
run_mlm.py \
    --config_name roberta-base \
    --tokenizer_name roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./outputs \
    --cache_dir ../cache
