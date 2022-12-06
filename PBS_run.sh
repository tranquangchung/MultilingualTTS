#!/bin/bash

#PBS -q GPU-S
#PBS -N tts_diffusion_style_AdaIN
#PBS -l select=2:ngpus=1
#PBS -j oe
#PBS -M bktranquangchung1@gmail.com -m be

source /etc/profile.d/modules.sh
module load singularity/3.9.5
image_path="/home/s2220411/singularity/multilingual-tts-v003.sif"
batch_script="/home/s2220411/Code/FastSpeech2_multilingual/script/run_multi.sh"
singularity exec --nv $image_path bash $batch_script