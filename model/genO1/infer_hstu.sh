#!/bin/bash

# 设置环境变量
export EVAL_DATA_PATH="./data"
export EVAL_RESULT_PATH="./result"
export MODEL_OUTPUT_PATH="./output"

# 创建必要的目录
mkdir -p $EVAL_RESULT_PATH

# 运行HSTU模型推理
python infer.py \
  --model_type hstu \
  --batch_size 128 \
  --maxlen 101 \
  --hidden_units 64 \
  --num_blocks 4 \
  --num_heads 4 \
  --dropout_rate 0.03 \
  --norm_first \
  --device cuda \
  --use_amp \
  --use_torch_compile \
  --use_cudnn_benchmark \
  --use_tf32 \
  --mm_emb_id 81