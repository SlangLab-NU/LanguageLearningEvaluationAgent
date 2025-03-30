#!/bin/bash

# Set the input and output directories
INPUT_DIR="/home/osx/Documents/GitHub/LLEvalAgent/data/recordings"
OUTPUT_DIR="/home/osx/Documents/GitHub/LLEvalAgent/data/recordings_wav"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script
python src/convert_to_wav.py "$INPUT_DIR" "$OUTPUT_DIR"

echo "Conversion completed. Check $OUTPUT_DIR for WAV files."
