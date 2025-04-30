#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --mem=16GB
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

module purge
module load discovery
module load python/3.8.1 
module load anaconda3/3.7 
module load ffmpeg/20190305 
source activate /work/van-speech-nlp/jindaznb/slamenv/
which python


python overall_score_weighted.py /work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/ellmat/evaluation/results