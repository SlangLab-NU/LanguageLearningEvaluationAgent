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

# Set the directory containing the audio files
INPUT_DIR="../data/recordings_wav_processed"

# Set the output file path
OUTPUT_FILE="../evaluation/audio_results/speech_metrics.json"

# Set the name filter (default is "USER")
NAME_FILTER="USER"

# Create the results directory if it doesn't exist
mkdir -p ../evaluation/audio_results

# Run the audio analysis script
python audio_features.py --input_dir "$INPUT_DIR" --output_file "$OUTPUT_FILE" --name_filter "$NAME_FILTER"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Audio analysis completed successfully."
    echo "Results saved to $OUTPUT_FILE"
else
    echo "Error: Audio analysis failed."
    exit 1
fi 