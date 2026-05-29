#!/bin/bash
set -e

GPU=0
CONFIG=APBSN_CONFOCAL
PATCH_SIZE=512
OVERLAP=0
THREAD=8

echo "=== Step 1: Prepare patches ==="
python prep.py -d Confocal -s $PATCH_SIZE -o $OVERLAP -p $THREAD

echo "=== Step 2: Train ==="
python train.py -c $CONFIG -g $GPU --thread $THREAD

echo "=== Step 3: Test ==="
CKPT=$(ls -t checkpoint/${CONFIG}_*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "No checkpoint found, trying ckpt folder..."
    CKPT=$(ls -t ckpt/${CONFIG}.pth 2>/dev/null | head -1)
fi

if [ -z "$CKPT" ]; then
    echo "No checkpoint found. Skipping test."
else
    python test.py -c $CONFIG -g $GPU --pretrained $(basename "$CKPT") --thread $THREAD
fi

echo "=== Done ==="
