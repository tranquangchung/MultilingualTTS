#!/bin/bash

#PBS -q GPU-LA
#PBS -N 14L_3500V2
#PBS -l select=2:ngpus=1
#PBS -j oe
#PBS -M bktranquangchung1@gmail.com -m be

source /etc/profile.d/modules.sh
module load singularity/3.9.5
image_path="/home/s2220411/singularity/multilingual-tts-v003.sif"
batch_script="/home/s2220411/Code/FastSpeech2_multilingual/script/run_multi.sh"
singularity exec --nv $image_path bash $batch_script