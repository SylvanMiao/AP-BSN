#!/bin/bash
set -e

GPU=0
CONFIG=APBSN_CONFOCAL
PATCH_SIZE=512
OVERLAP=0
THREAD=8

echo "=== Step 1: Train ==="
mkdir -p ./dataset
python train.py -c $CONFIG -g $GPU --thread $THREAD

echo "=== Step 2: Test ==="
mkdir -p ckpt
CKPT=$(ls -t checkpoint/${CONFIG}_*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "No checkpoint found in checkpoint/. Skipping test."
else
    cp "$CKPT" ckpt/
    CKPT_NAME=$(basename "$CKPT")
    python test.py -c $CONFIG -g $GPU --pretrained "$CKPT_NAME" --thread $THREAD
fi

echo "=== Done ==="
