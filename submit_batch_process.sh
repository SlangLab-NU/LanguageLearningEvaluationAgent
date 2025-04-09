#!/bin/bash
# Wrapper script to submit batch_process_recordings.py to SLURM

# Default values
RECORDINGS_DIR=""
OUTPUT_DIR=""
MODEL="large-v2"
DEVICE="cuda"
COMPUTE_TYPE="float16"
BATCH_SIZE=16
MIN_SPEAKERS=2
MAX_SPEAKERS=2
HF_TOKEN=""
LANGUAGE="en"
USER_SPEAKER="SPEAKER_00"
NPC_SPEAKER="SPEAKER_01"
JOB_NAME="whisperx_diarization"
TIME_LIMIT="24:00:00"
MEMORY="32G"
CPUS=4
GPUS=1
PARTITION="gpu"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --recordings_dir)
      RECORDINGS_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --compute_type)
      COMPUTE_TYPE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --min_speakers)
      MIN_SPEAKERS="$2"
      shift 2
      ;;
    --max_speakers)
      MAX_SPEAKERS="$2"
      shift 2
      ;;
    --hf_token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --language)
      LANGUAGE="$2"
      shift 2
      ;;
    --user_speaker)
      USER_SPEAKER="$2"
      shift 2
      ;;
    --npc_speaker)
      NPC_SPEAKER="$2"
      shift 2
      ;;
    --job_name)
      JOB_NAME="$2"
      shift 2
      ;;
    --time_limit)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --memory)
      MEMORY="$2"
      shift 2
      ;;
    --cpus)
      CPUS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$RECORDINGS_DIR" ]; then
  echo "Error: --recordings_dir is required"
  exit 1
fi

# Set output directory to recordings directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$RECORDINGS_DIR"
fi

# Create a temporary SLURM script
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${JOB_NAME}_%j.out
#SBATCH --error=${JOB_NAME}_%j.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --partition=${PARTITION}

# Load necessary modules (adjust based on your HPC system)
module load cuda/11.8
module load anaconda3

# Activate the appropriate conda environment
# Replace 'your_env_name' with your actual environment name
source activate your_env_name

# Set environment variables
export HF_TOKEN="${HF_TOKEN}"

# Set paths
SCRIPT_DIR="\$(dirname "\$(readlink -f "\$0")")"
PYTHON_SCRIPT="\${SCRIPT_DIR}/src/batch_process_recordings.py"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Run the Python script
python "\${PYTHON_SCRIPT}" "${RECORDINGS_DIR}" \\
    --output_dir "${OUTPUT_DIR}" \\
    --model "${MODEL}" \\
    --device "${DEVICE}" \\
    --compute_type "${COMPUTE_TYPE}" \\
    --batch_size ${BATCH_SIZE} \\
    --min_speakers ${MIN_SPEAKERS} \\
    --max_speakers ${MAX_SPEAKERS} \\
    --hf_token "\${HF_TOKEN}" \\
    --language "${LANGUAGE}" \\
    --user_speaker "${USER_SPEAKER}" \\
    --npc_speaker "${NPC_SPEAKER}"

# Print completion message
echo "Job completed at \$(date)"
EOF

# Submit the job
echo "Submitting job to SLURM..."
sbatch "$TMP_SCRIPT"

# Clean up
rm "$TMP_SCRIPT"

echo "Job submitted successfully!" 