#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --mem=16GB
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export HUGGING_FACE_HUB_TOKEN="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"  # Add your Hugging Face token here

module purge
module load discovery
module load python/3.8.1 
module load anaconda3/3.7 
module load ffmpeg/20190305 
source activate /work/van-speech-nlp/jindaznb/mmenv/
which python

# Train from scratch
# python level_estimator.py --model bert-base-cased --lm_layer 11 --seed 935 --num_labels 6 --batch 128 --warmup 0 --with_loss_weight --num_prototypes 3 --type contrastive --init_lr 1.0e-5 --alpha 0.2 --data ../CEFR-SP/SCoRE/CEFR-SP_SCoRE_ --test ../CEFR-SP/SCoRE/CEFR-SP_SCoRE_ --out ../out/

# Test a pretrained model
python level_estimator.py \
 --model bert-base-cased --lm_layer 11 --seed 935 --num_labels 6 --batch 128 --warmup 0 --with_loss_weight --num_prototypes 3 --type contrastive --init_lr 1.0e-5 --alpha 0.2 --data ../CEFR-SP/SCoRE/CEFR-SP_SCoRE --test ../CEFR-SP/SCoRE/CEFR-SP_SCoRE --out ../out/ --pretrained ../j/level_estimator.ckpt