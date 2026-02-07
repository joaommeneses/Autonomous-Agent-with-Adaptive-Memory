#!/bin/bash
# Start vLLM server for Qwen2.5-1M-Instruct
# Usage: ./start_vllm.sh [port] [gpu_id]

PORT=${1:-8000}
GPU_ID=${2:-0}

echo "Starting vLLM server..."
echo "  Model: Qwen/Qwen2.5-7B-Instruct"
echo "  Port: $PORT"
echo "  GPU: $GPU_ID"
echo ""

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port $PORT \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --dtype=half