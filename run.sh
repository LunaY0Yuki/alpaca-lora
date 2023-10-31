#!/bin/bash

python finetune.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --data_path /share/portal/hw575/language_irl/alpaca_data/simple_exp1/alpaca_final_states_train.json \
    --val_data_path /share/portal/hw575/language_irl/alpaca_data/simple_exp1/alpaca_final_states_dev.json \
    --output_dir './output/e=10_with-val/' \
    --num_epochs 10 \
    --wandb_project language-irl-prelim