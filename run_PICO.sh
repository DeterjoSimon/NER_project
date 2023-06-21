#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J PICO_7_5_13
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R 'select[gpu32gb && !sxm2 ]'
#BSUB -R "rusage[mem=8GB]"
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
#BSUB -o PICO_7_5_13.out
#BSUB -e PICO_7_5_13.err
# -- end of LSF options --


module swap python3/3.9.11
module load cuda/11.0
module load numpy/1.22.3-python-3.9.11-openblas-0.3.19
# /appl/cuda/11.4.0/samples/bin/x86_64/linux/release/deviceQuery
source venv/bin/activate
pip3 install -r /work3/s174450/requirements.txt
# pip3 install --upgrade torch
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

SEEDS=(13)
N=7
K=5
mode=inter
dataset=PICO

for seed in ${SEEDS[@]}; do
    echo "Next seed"
    python3 main.py \
        --dataset=${dataset} \
        --types_path /work3/s174450/data/entity_types_pico.json \
        --result_dir /work3/s174450 \
        --seed=${seed} \
        --mode=${mode} \
        --N=${N} \
        --K=${K} \
        --similar_k=10 \
        --max_meta_steps=601 \
        --eval_every_meta_steps=100 \
        --name=PICO_SPAN \
        --train_mode=span \
        --inner_steps=2 \
        --inner_size=32 \
        --max_ft_steps=3 \
        --lambda_max_loss=2 \
        --inner_lambda_max_loss=5 \
        --tagging_scheme=BIOES \
        --viterbi=hard \
        --concat_types=None \
        --data_path /work3/s174450/data/pico-episode-data \
        --ignore_eval_test
        

    python3 main.py \
        --dataset=${dataset} \
        --types_path /work3/s174450/data/entity_types_pico.json \
        --result_dir /work3/s174450 \
        --seed=${seed} \
        --lr_inner=1e-4 \
        --lr_meta=1e-4 \
        --mode=${mode} \
        --N=${N} \
        --K=${K} \
        --max_meta_steps=601 \
        --similar_k=10 \
        --inner_similar_k=10 \
        --eval_every_meta_steps=100 \
        --name=PICO_TYPE \
        --train_mode=type \
        --inner_steps=2 \
        --inner_size=32 \
        --data_path /work3/s174450/data/pico-episode-data \
        --max_ft_steps=3 \
        --concat_types=None \
        --lambda_max_loss=2.0

    cp models-${N}-${K}-${mode}/bert-base-uncased-innerSteps_2-innerSize_32-lrInner_0.0001-lrMeta_0.0001-maxSteps_601-seed_${seed}-name_PICO_TYPE/en_type_pytorch_model.bin models-${N}-${K}-${mode}/bert-base-uncased-innerSteps_2-innerSize_32-lrInner_3e-05-lrMeta_3e-05-maxSteps_601-seed_${seed}-name_PICO_SPAN

    python3 main.py \
        --dataset=${dataset} \
        --types_path /work3/s174450/data/entity_types_pico.json \
        --result_dir /work3/s174450 \
        --seed=${seed} \
        --N=${N} \
        --K=${K} \
        --mode=${mode} \
        --max_meta_steps=601 \
        --similar_k=10 \
        --name=PICO_SPAN \
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
        --data_path /work3/s174450/data/pico-episode-data \
        --viterbi=hard \
        --tagging_scheme=BIOES
done
