#!/bin/bash

# Set the input and output directories
INPUT_DIR="../data/recordings"
OUTPUT_DIR="../data/recordings_wav"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script
python convert_to_wav.py "$INPUT_DIR" "$OUTPUT_DIR"

echo "Conversion completed. Check $OUTPUT_DIR for WAV files."