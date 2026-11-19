#!/usr/bin/env bash
set -euo pipefail

LLM_MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "google/gemma-4-E2B-it"
    "google/gemma-4-E4B-it"
)

for model in "${LLM_MODELS[@]}"; do
    echo "Generating response for $model"
    EXTRA_FLAGS=""
    if [[ "$model" == *"E4B"* ]] || [[ "$model" == *"e4b"* ]]; then
        EXTRA_FLAGS="--load_in_4bit"
    fi
    CUDA_VISIBLE_DEVICES=0 python generate_response.py --llm_model "$model" $EXTRA_FLAGS
done
