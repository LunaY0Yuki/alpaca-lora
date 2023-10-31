#!/bin/bash

python evaluate.py \
    --load_8bit false\
    --base_model meta-llama/Llama-2-7b-hf \
    --lora_weights './output/e=10_with-val' \
    --test_data_path /share/portal/hw575/language_irl/alpaca_data/simple_exp1/alpaca_final_states_test.json