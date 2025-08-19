#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

python -u main.py --hidden_units 64 --num_blocks 2 --num_heads 4 --batch_size 256 --lr 0.0005 --num_epochs 10 --dropout_rate 0.1 --l2_emb 0.001 --mm_emb_id 81 82 83 84