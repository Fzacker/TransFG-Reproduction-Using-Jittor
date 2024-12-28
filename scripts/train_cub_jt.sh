#!/bin/bash
export DISABLE_MULTIPROCESSING=1
export JT_SAVE_MEM=1
export JT_SYNC=1
export trace_py_var=3
export device_mem_limit=40000000000
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_ALLOW_RUN_AS_ROOT=1

CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 2 python train.py \
    --dataset CUB_200_2011 \
    --split overlap \
    --num_steps 10000 \
    --train_batch_size 16 \
    --name cub_overlap_mar_0 \
    --data_root ./data/CUB_200_2011 \
    --model_type ViT-B_16 \
    --pretrained_dir ./data/models/ViT-B_16.npz \