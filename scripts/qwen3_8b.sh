#!/bin/bash
# 2SSP+DLP Qwen3-8B 프루닝

python run_2ssp_dlp.py \
    --model "Qwen/Qwen3-8B" \
    --pruning_rate 0.5 \
    --alpha 1.5 \
    --alpha_dlp 0.15 \
    --nsamples 32 \
    --save_model "pruned/Qwen3-8B-2ssp-dlp-sparsity0.5"
