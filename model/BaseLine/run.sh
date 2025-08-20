#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

python -u main.py --use_amp --use_compile --enable_tf32 --cudnn_deterministic --num_workers 16