#!/bin/bash

# 设置环境变量
export TRAIN_DATA_PATH="./data"
export TRAIN_LOG_PATH="./log"
export TRAIN_TF_EVENTS_PATH="./events"
export TRAIN_CKPT_PATH="./ckpt"
export MODEL_OUTPUT_PATH="./output"

# 创建必要的目录
mkdir -p $TRAIN_LOG_PATH
mkdir -p $TRAIN_TF_EVENTS_PATH
mkdir -p $TRAIN_CKPT_PATH
mkdir -p $MODEL_OUTPUT_PATH

# 运行HSTU模型训练
python main.py \
  --model_type hstu \
  --batch_size 128 \
  --lr 0.001 \
  --maxlen 101 \
  --hidden_units 64 \
  --num_blocks 4 \
  --num_epochs 5 \
  --num_heads 4 \
  --dropout_rate 0.03 \
  --l2_emb 0.001 \
  --device cuda \
  --norm_first \
  --use_amp \
  --use_torch_compile \
  --use_cudnn_benchmark \
  --use_tf32 \
  --mm_emb_id 81