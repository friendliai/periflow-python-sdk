# Install gcc/g++
echo
echo "##### INSTALLING GCC / G++"
apt-get update -qq
apt-get install gcc -qq
apt-get install g++ -qq

pip -q install --upgrade pip

# Install detectron2
echo
echo "##### INSTALLING DETECTRON2"
pip -q install 'git+https://github.com/facebookresearch/detectron2.git'

# Run
export FILENAME="train.py"
echo
echo "##### RUNNING TRAINING PROGRAM ($FILENAME)"
# Add `--resume` to use checkpoint saved where `--output-dir` specifies.
torchrun --nnodes $NUM_NODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port 6000 --nproc_per_node $NPROC_PER_NODE $FILENAME \
    --config-file ./detectron_config/mask_rcnn_R_50_FPN_3x.yaml \
    --data-dir /workspace/data/coco_fruits_nuts \
    --output-dir /workspace/ckpt \
    --save-interval 50 \
    --resume
