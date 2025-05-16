#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-28508}

CONFIG="./projects/configs/diff_bevformer/layout_tiny.py"
UNET_CHECKPOINT_DIR="../../results/BEVDiffuser_BEVFormer_tiny/checkpoint-50000"
LOAD_FROM="../../ckpts/bevformer_tiny_epoch_24.pth"
RESUME_FROM=None
RUN_NAME="BEVFormer_tiny_with_BEVDiffuser"
WORK_DIR="/../../results"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work-dir=$WORK_DIR \
    --report-to='wandb' \
    --tracker-project-name='DiffBEVFormer' \
    --tracker-run-name=$RUN_NAME \
    --unet-checkpoint-dir=$UNET_CHECKPOINT_DIR \
    --load-from=$LOAD_FROM \
    --resume-from=$RESUME_FROM \
