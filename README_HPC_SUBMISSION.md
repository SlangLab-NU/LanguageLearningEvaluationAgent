# HPC Submission for WhisperX Speaker Diarization

This document explains how to submit the WhisperX speaker diarization job to an HPC system using SLURM.

## Prerequisites

Before using these scripts, ensure you have:

1. Access to an HPC system with SLURM job scheduler
2. The necessary Python environment with WhisperX and its dependencies installed
3. A Hugging Face token for accessing PyAnnote models

## Available Scripts

### 1. `batch_process_recordings.sh`

This is a basic SLURM submission script with hardcoded parameters. It's suitable for simple use cases where you don't need to change parameters frequently.

### 2. `submit_batch_process.sh`

This is a more flexible wrapper script that allows you to specify parameters via command-line arguments. It creates a temporary SLURM script with your specified parameters and submits it to the queue.

## Usage

### Basic Usage

To submit a job using the basic script:

```bash
# Make the script executable
chmod +x batch_process_recordings.sh

# Edit the script to set your parameters
nano batch_process_recordings.sh

# Submit the job
sbatch batch_process_recordings.sh
```

### Advanced Usage

To submit a job using the flexible wrapper script:

```bash
# Make the script executable
chmod +x submit_batch_process.sh

# Submit a job with custom parameters
./submit_batch_process.sh \
  --recordings_dir "/path/to/your/recordings" \
  --output_dir "/path/to/your/output" \
  --model "large-v2" \
  --device "cuda" \
  --compute_type "float16" \
  --batch_size 16 \
  --min_speakers 2 \
  --max_speakers 2 \
  --hf_token "your_huggingface_token" \
  --language "en" \
  --user_speaker "SPEAKER_00" \
  --npc_speaker "SPEAKER_01" \
  --job_name "my_whisperx_job" \
  --time_limit "24:00:00" \
  --memory "32G" \
  --cpus 4 \
  --gpus 1 \
  --partition "gpu"
```

### Required Parameters

- `--recordings_dir`: Directory containing audio recordings (required)

### Optional Parameters

- `--output_dir`: Directory to save output files (default: same as recordings_dir)
- `--model`: Whisper model to use (default: "large-v2")
- `--device`: Device to run inference on (default: "cuda")
- `--compute_type`: Compute type for inference (default: "float16")
- `--batch_size`: Batch size for inference (default: 16)
- `--min_speakers`: Minimum number of speakers (default: 2)
- `--max_speakers`: Maximum number of speakers (default: 2)
- `--hf_token`: Hugging Face token for accessing PyAnnote models
- `--language`: Language code for transcription (default: "en")
- `--user_speaker`: Speaker ID for USER (default: "SPEAKER_00")
- `--npc_speaker`: Speaker ID for NPC (default: "SPEAKER_01")
- `--job_name`: Name of the SLURM job (default: "whisperx_diarization")
- `--time_limit`: Time limit for the job (default: "24:00:00")
- `--memory`: Memory allocation for the job (default: "32G")
- `--cpus`: Number of CPUs to allocate (default: 4)
- `--gpus`: Number of GPUs to allocate (default: 1)
- `--partition`: SLURM partition to use (default: "gpu")

## Customizing for Your HPC System

You may need to customize the scripts based on your specific HPC system:

1. **Module Loading**: Adjust the `module load` commands to match the available modules on your system.
2. **Environment Activation**: Replace `your_env_name` with the name of your conda environment.
3. **Partition Names**: Change the `--partition` parameter to match the available partitions on your system.
4. **Resource Limits**: Adjust memory, CPU, and GPU allocations based on your system's capabilities and policies.

## Monitoring Jobs

To monitor your submitted jobs:

```bash
# Check job status
squeue -u $USER

# Check job details
scontrol show job <job_id>

# Cancel a job
scancel <job_id>
```

## Troubleshooting

If your job fails, check the error log file (`whisperx_diarization_<job_id>.err`) for details. Common issues include:

1. **Environment Issues**: Ensure your conda environment has all required packages installed.
2. **Resource Issues**: If the job is killed due to resource limits, try reducing batch size or using a smaller model.
3. **Path Issues**: Ensure all paths are correct and accessible from the compute nodes.
4. **Permission Issues**: Ensure you have write permissions to the output directory. 