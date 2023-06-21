
#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J 5_1_To_5_5_s10
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R 'select[gpu32gb && !sxm2 ]'
#BSUB -R "rusage[mem=15GB]"
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
#BSUB -o 5_1_To_5_5_s10.out
#BSUB -e 5_1_To_5_5_s10.err
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

SEEDS=(171)
N=5
K=1
mode=inter
dataset=PICO

for seed in ${SEEDS[@]}; do
    python3 main.py \
        --seed=${seed} \
        --types_path /work3/s174450/data/entity_types_pico.json \
        --result_dir /work3/s174450 \
        --dataset=${dataset} \
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
        --data_path /work3/s174450/data/pico-episode-data \
        --inner_similar_k=10 \
        --viterbi=hard \
        --tagging_scheme=BIOES
done