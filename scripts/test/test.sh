#!/bin/bash

# Loading the required module
# source /etc/profile
# module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
# source activate th102_cu113_tgconda

python test.py \
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
    --checkpoint-path1 /your_path/NextHAM/pretrained_models/model_range0_curr.pth.tar \
    --checkpoint-path2 /your_path/NextHAM/pretrained_models/model_range1_curr.pth.tar \
    --checkpoint-path3 /your_path/NextHAM/pretrained_models/model_range2_curr.pth.tar \
    --checkpoint-path4 /your_path/NextHAM/pretrained_models/model_range3_curr.pth.tar