#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J NERD_N5_K1_985
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R 'select[gpu32gb && !sxm2 ]'
#BSUB -R 'select[eth10g]'
#BSUB -R "rusage[mem=7GB]"
#BSUB -R "span[hosts=1]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
###BSUB -N$ 
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o NERD_N5_K1_985.out
#BSUB -e NERD_N5_K1_985.err
# -- end of LSF options --


module swap python3/3.8.1
source venv/bin/activate
pip3 install -r requirements.txt
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

SEEDS=(985)
N=5
K=1
mode=inter

for seed in ${SEEDS[@]}; do
    echo "Next seed"
    python3 main.py \
        --seed=${seed} \
        --mode=${mode} \
        --N=${N} \
        --K=${K} \
        --similar_k=10 \
        --eval_every_meta_steps=100 \
        --name=10-k_100_2_32_3_max_loss_2_5_BIOES \
        --train_mode=span \
        --inner_steps=2 \
        --inner_size=32 \
        --max_ft_steps=3 \
        --lambda_max_loss=2 \
        --inner_lambda_max_loss=5 \
        --tagging_scheme=BIOES \
        --viterbi=hard \
        --concat_types=None \
        --data_path data/episode-data \
        --ignore_eval_test
        

    python3 main.py \
        --seed=${seed} \
        --lr_inner=1e-4 \
        --lr_meta=1e-4 \
        --mode=${mode} \
        --N=${N} \
        --K=${K} \
        --similar_k=10 \
        --inner_similar_k=10 \
        --eval_every_meta_steps=100 \
        --name=10-k_100_type_2_32_3_10_10 \
        --train_mode=type \
        --inner_steps=2 \
        --inner_size=32 \
        --data_path data/episode-data \
        --max_ft_steps=3 \
        --concat_types=None \
        --lambda_max_loss=2.0

    cp models-${N}-${K}-${mode}/bert-base-uncased-innerSteps_2-innerSize_32-lrInner_0.0001-lrMeta_0.0001-maxSteps_5001-seed_${seed}-name_10-k_100_type_2_32_3_10_10/en_type_pytorch_model.bin models-${N}-${K}-${mode}/bert-base-uncased-innerSteps_2-innerSize_32-lrInner_3e-05-lrMeta_3e-05-maxSteps_5001-seed_${seed}-name_10-k_100_2_32_3_max_loss_2_5_BIOES

    python3 main.py \
        --seed=${seed} \
        --N=${N} \
        --K=${K} \
        --mode=${mode} \
        --similar_k=10 \
        --name=10-k_100_2_32_3_max_loss_2_5_BIOES \
        --concat_types=None \
        --test_only \
        --eval_mode=two-stage \
        --inner_steps=2 \
        --inner_size=32 \
        --max_ft_steps=3 \
        --max_type_ft_steps=3 \
        --lambda_max_loss=2.0 \
        --inner_lambda_max_loss=5.0 \
        --inner_similar_k=10 \
        --data_path data/episode-data \
        --viterbi=hard \
        --tagging_scheme=BIOES
done