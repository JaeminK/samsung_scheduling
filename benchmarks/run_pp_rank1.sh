#!/bin/bash

# Set environment variables for rank 1
export LOCAL_RANK=1
export WORLD_SIZE=2
export RANK=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Run rank 1
nsys profile --trace=cuda,nvtx -o tp_dist_rank1 \
    python ../main.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --cache-dir /workspace/cache \
        --output-dir ./results \
        --tensor-parallel-size 2 \
        --local-rank 1 \
        --world-size 2 \
        --seed 1234 \
        --min-output-length 1 \
        --max-output-length 512
