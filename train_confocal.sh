#!/bin/bash
set -e

GPU=0
CONFIG=APBSN_CONFOCAL
THREAD=8

echo "=== Train AP_BSN CONFOCAL==="
mkdir -p ./dataset
python train.py -c $CONFIG -g $GPU --thread $THREAD

echo "=== Done ==="
