#!/bin/bash

echo "Starting the imagenet CNN training task..."
CUDA_VISIBLE_DEVICES=3 python -u imagenet_impl.py -e 50 -T 0.005 -v 0.04

echo "Process finished at $(date)"