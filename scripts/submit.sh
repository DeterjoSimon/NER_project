#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Exp_Transformer_noLayerNorm_100
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R 'select[localssd0 && gpu32gb && !sxm2 ]'
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
#BSUB -o gpu_Transformer_noLayerNorm_100.out
#BSUB -e gpu_Transformer_noLayerNorm_100.err
# -- end of LSF options --


module swap python3/3.9.11
source /dtu-compute/s174388-s174450/MasterProj/PowerMarketForecaster/src/data_proc/ve/bin/activate
#rm -r /localssd0/$USER
mkdir -p /localssd0/$USER
chmod go-rwx /localssd0/$USER
rsync -av /dtu-compute/s174388-s174450/MasterProj/skaae_amp_799aa5cddde8c8a5ad67925372c7e34f61f6437c_nov12 /localssd0/$USER
rsync -av /dtu-compute/s174388-s174450/MasterProj/PowerMarketForecaster/src/parameters.yaml /localssd0/$USER
rsync -av /dtu-compute/s174388-s174450/MasterProj/PowerMarketForecaster/src/data_proc/indices.pickle /localssd0/$USER
rsync -av /dtu-compute/s174388-s174450/MasterProj/PowerMarketForecaster/src/data_proc/scaler.pickle /localssd0/$USER
python3 /dtu-compute/s174388-s174450/MasterProj/PowerMarketForecaster/src/main.py experiment=Transformerensemble3