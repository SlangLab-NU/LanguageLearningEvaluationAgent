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

module purge
module load discovery
module load python/3.8.1 
module load anaconda3/3.7 
module load ffmpeg/20190305 
source activate /work/van-speech-nlp/jindaznb/slamenv/
which python

# Set paths
RECORDINGS_DIR="/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/ellmat/data/recordings_wav"
OUTPUT_DIR="/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/ellmat/data/recordings_wav_output"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PYTHON_SCRIPT="${SCRIPT_DIR}/batch_process_recordings.py"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Run the Python script
python "${PYTHON_SCRIPT}" "${RECORDINGS_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --model "large-v2" \
    --device "cuda" \
    --compute_type "float16" \
    --batch_size 16 \
    --min_speakers 2 \
    --max_speakers 2 \
    --hf_token "${HF_TOKEN}" \
    --language "en" \
    --user_speaker "SPEAKER_00" \
    --npc_speaker "SPEAKER_01"

# Print completion message
echo "Job completed at $(date)" 