#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J NER_model
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R 'select[gpu32gb && !sxm2 ]'
#BSUB -R 'select[eth10g]'
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
#BSUB -o NER_model.out
#BSUB -e NER_model.err
# -- end of LSF options --


module swap python3/3.8.1
source venv/bin/activate
pip3 install -r requirements.txt
python3 main.py --N 5 --K 1 --mode inter --tagging_scheme BIO --name Class_ON --data_path data/episode-data