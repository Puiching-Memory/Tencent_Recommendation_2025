#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

python -u main.py --use_amp --use_torch_compile --use_cudnn_benchmark --use_tf32 --rq_num_codebooks 4 --rq_codebook_size 64 --rq_kmeans_method kmeans --rq_kmeans_iters 100 --rq_distances_method l2 --rq_loss_beta 0.25