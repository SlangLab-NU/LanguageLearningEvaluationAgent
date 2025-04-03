#!/bin/bash
# Script to run the entire speaker diarization workflow

# Check if Hugging Face token is provided
if [ -z "$HF_TOKEN" ]; then
    echo "Error: Hugging Face token not found in environment variables."
    echo "Please set your token with: export HF_TOKEN=your_token_here"
    echo "Or provide it with the --hf_token option."
    exit 1
fi

# Default values
AUDIO_FILE=""
OUTPUT_DIR=""
MODEL="large-v2"
DEVICE="cuda"
COMPUTE_TYPE="float16"
BATCH_SIZE=16
MIN_SPEAKERS=2
MAX_SPEAKERS=2
LANGUAGE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --audio_file)
            AUDIO_FILE="$2"
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
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if audio file is provided
if [ -z "$AUDIO_FILE" ]; then
    echo "Error: Audio file not provided."
    echo "Usage: $0 --audio_file /path/to/audio.wav [options]"
    echo ""
    echo "Options:"
    echo "  --output_dir DIR       Directory to save output files"
    echo "  --model MODEL          Whisper model to use (default: large-v2)"
    echo "  --device DEVICE        Device to run inference on (default: cuda)"
    echo "  --compute_type TYPE    Compute type for inference (default: float16)"
    echo "  --batch_size SIZE      Batch size for inference (default: 16)"
    echo "  --min_speakers NUM     Minimum number of speakers (default: 2)"
    echo "  --max_speakers NUM     Maximum number of speakers (default: 2)"
    echo "  --language LANG        Language code for transcription"
    exit 1
fi

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    exit 1
fi

# Set output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR=$(dirname "$AUDIO_FILE")
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get base filename without extension
BASE_FILENAME=$(basename "$AUDIO_FILE" | sed 's/\.[^.]*$//')

echo "=== Running Speaker Diarization ==="
echo "Audio file: $AUDIO_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Compute type: $COMPUTE_TYPE"
echo "Batch size: $BATCH_SIZE"
echo "Min speakers: $MIN_SPEAKERS"
echo "Max speakers: $MAX_SPEAKERS"
echo "Language: $LANGUAGE"
echo ""

# Build command for speaker diarization
DIARIZATION_CMD="python speaker_diarization.py \"$AUDIO_FILE\" --output_dir \"$OUTPUT_DIR\" --model $MODEL --device $DEVICE --compute_type $COMPUTE_TYPE --batch_size $BATCH_SIZE --min_speakers $MIN_SPEAKERS --max_speakers $MAX_SPEAKERS --hf_token $HF_TOKEN"

if [ ! -z "$LANGUAGE" ]; then
    DIARIZATION_CMD="$DIARIZATION_CMD --language $LANGUAGE"
fi

echo "Running speaker diarization..."
eval $DIARIZATION_CMD

# Check if diarization was successful
DIARIZATION_FILE="$OUTPUT_DIR/${BASE_FILENAME}_diarization.txt"
if [ ! -f "$DIARIZATION_FILE" ]; then
    echo "Error: Diarization failed. Diarization file not found: $DIARIZATION_FILE"
    exit 1
fi

echo ""
echo "=== Extracting Speaker Audio ==="

# Build command for extracting speaker audio
EXTRACT_CMD="python extract_speaker_audio.py \"$AUDIO_FILE\" \"$DIARIZATION_FILE\" --output_dir \"$OUTPUT_DIR\""

echo "Extracting speaker audio..."
eval $EXTRACT_CMD

echo ""
echo "=== Workflow Completed ==="
echo "Transcription file: $OUTPUT_DIR/${BASE_FILENAME}_transcript.txt"
echo "Diarization file: $DIARIZATION_FILE"
echo "Speaker audio files:"
echo "  - $OUTPUT_DIR/${BASE_FILENAME}_USER.wav"
echo "  - $OUTPUT_DIR/${BASE_FILENAME}_NPC.wav" 