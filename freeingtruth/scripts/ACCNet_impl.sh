#!/bin/bash

echo "Starting the ACCNet training task..."
CUDA_VISIBLE_DEVICES=6 python -u ACCNet_impl.py -e 200 -T 0.8 -b 256

echo "Process finished at $(date)"