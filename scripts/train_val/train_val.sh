#!/bin/bash

# Loading the required module
# source /etc/profile
# module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
# source activate th102_cu113_tgconda

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_val.py \
    --output-dir '/work/home/nextham/NextHAM_V1/res' \
    --model-name 'graph_attention_transformer_nonlinear_materials_ham_soc' \
    --input-irreps '64x0e' \
    --radius 8.0 \
    --is-accurate-label \
    --trace-out-len 81 \
    --epochs 100 \
    --lr 5e-4 \
    --batch-size 1 \
    --eval-batch-size 1 \
    --weight-decay 0 \
    --num-basis 64 \
    --with-trace \
    --test-interval 10000 \
    --target 'hamiltonian' \
    --target-blocks-type 'all' \
    --print-freq 100 \
    --checkpoint-path1 /work/home/nextham/NextHAM_V1/pretrained_models/model_range0_curr.pth.tar \
    --checkpoint-path2 /work/home/nextham/NextHAM_V1/pretrained_models/model_range1_curr.pth.tar \
    --checkpoint-path3 /work/home/nextham/NextHAM_V1/pretrained_models/model_range2_curr.pth.tar \
    --checkpoint-path4 /work/home/nextham/NextHAM_V1/pretrained_models/model_range3_curr.pth.tar