#!/bin/bash

# Loading the required module
# source /etc/profile
# module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
# source activate th102_cu113_tgconda

CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py \
    --output-dir '/your_path/NextHAM/test_res/' \
    --model-name 'graph_attention_transformer_nonlinear_materials_ham_soc' \
    --input-irreps '64x0e' \
    --radius 8.0 \
    --is-accurate-label \
    --trace-out-len 81 \
    --batch-size 1 \
    --eval-batch-size 1 \
    --weight-decay 0 \
    --num-basis 64 \
    --workers 0 \
    --with-trace \
    --energy-weight 1 \
    --force-weight 80 \
    --test-interval 10000 \
    --target 'hamiltonian' \
    --target-blocks-type 'all' \
    --checkpoint-path1 /your_path/NextHAM/res/model_range0_best.pth.tar \
    --checkpoint-path2 /your_path/NextHAM/res/model_range1_best.pth.tar \
    --checkpoint-path3 /your_path/NextHAM/res/model_range2_best.pth.tar \
    --checkpoint-path4 /your_path/NextHAM/res/model_range3_best.pth.tar