#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Proto_PICO_7_5_13
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
# request 5GB of system-memory
#BSUB -R 'select[gpu32gb && !sxm2 ]'
#BSUB -R 'select[eth10g]'
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
###BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Proto_PICO_7_5_13.out
#BSUB -e Proto_PICO_7_5_13.err
# -- end of LSF options --


module swap python3/3.9.11
module load cuda/11.0
source venv/bin/activate
pip3 install -r requirements.txt
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

python3 train_demo.py  --mode inter \
--lr 1e-4 --batch_size 1 --trainN 7 --N 7 --K 5 --Q 5 \
--train_iter 20000 --val_iter 500 --test_iter 5000 --val_step 1000 --max_length 32 --model proto --seed 13 \
--use_sampled_data