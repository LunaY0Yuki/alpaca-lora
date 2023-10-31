#!/bin/bash

python generate.py \
    --load_8bit True\
    --base_model meta-llama/Llama-2-7b-hf \
    --lora_weights './output/e=10_with-val' \
    --server_name '127.0.0.1'