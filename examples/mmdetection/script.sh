echo
echo "##### INSTALLING REQUIREMENTS"
apt-get update -qq
apt-get install ffmpeg libsm6 libxext6 -qq
apt-get install libgll-mesa-glx -qq
apt-get install gcc -qq
apt-get install g++ -qq
pip -q install -r requirements.txt

# Install mmdetection
echo
echo "##### INSTALLING MMDETECTION"
pip -q install --upgrade pip
pip -q install --upgrade openmim
mim install mmcv-full
pip -q install mmdet

# Run
export FILENAME="train.py"
echo
echo "##### RUNNING TRAINING PROGRAM ($FILENAME)"
# Add `--auto-resume` to resume training state from checkpoint
torchrun --nnodes $NUM_NODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port 6000 --nproc_per_node $NPROC_PER_NODE $FILENAME \
    --config configs/balloon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py \
    --work-dir /workspace/logs \
    --checkpoint-dir /workspace/ckpt
