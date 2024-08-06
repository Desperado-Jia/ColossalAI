#!/bin/bash

# All raw dataset for training
declare -a INPUT_FILES=(
    "./demo.jsonl"
)
OUTPUT_DIR="./demo.tk"
MODEL_PATH="./models/llama3"

python prepare.py \
    --input_files ${INPUT_FILES[@]} \
    --output_dir $OUTPUT_DIR \
    --pretrained_model_name_or_path $MODEL_PATH \
    --max_sequence_length 4096
