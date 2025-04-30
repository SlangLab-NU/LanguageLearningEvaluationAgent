#!/usr/bin/env python3
"""
Extract USER utterances from transcript files.
The first speaker in each transcript file is considered the USER.

Usage:
    python3 extract_user_transcripts.py /path/to/recordings/directory
"""

import os
import glob
import sys
from pathlib import Path

def extract_user_utterances(transcript_file: str) -> list:
    """Extract all utterances from the USER (first speaker) in the transcript."""
    user_utterances = []
    user_speaker = None
    
    with open(transcript_file, 'r') as f:
        lines = f.readlines()
        
        # Find the first speaker (USER)
        for line in lines:
            if '] SPEAKER_' in line:
                # Extract speaker ID from first line
                user_speaker = line.split('] ')[1].split(':')[0].strip()
                break
        
        if not user_speaker:
            print(f"Warning: No speaker found in {transcript_file}")
            return []
            
        # Extract all utterances from the user speaker
        for line in lines:
            if f'] {user_speaker}:' in line:
                # Extract only the text, ignoring timestamp
                text = line.split(':', 1)[1].strip()
                user_utterances.append(text)
    
    return user_utterances

def process_all_transcripts(input_dir: str):
    """Process all transcript files in the directory."""
    # Find all transcript files
    transcript_files = glob.glob(os.path.join(input_dir, "*_transcript.txt"))
    
    if not transcript_files:
        print(f"No transcript files found in {input_dir}")
        return
        
    print(f"Found {len(transcript_files)} transcript files")
    
    for transcript_file in transcript_files:
        print(f"Processing {transcript_file}...")
        
        # Get user utterances
        user_utterances = extract_user_utterances(transcript_file)
        
        if not user_utterances:
            print(f"No user utterances found in {transcript_file}")
            continue
            
        # Create output filename
        output_file = transcript_file.replace("_transcript.txt", "_transcript_USER.txt")
        
        # Write user utterances to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(user_utterances))
            
        print(f"Saved user utterances to {output_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 extract_user_transcripts.py /path/to/recordings/directory")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    
    if not os.path.exists(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)
        
    process_all_transcripts(input_dir)
    print("Done!")

if __name__ == "__main__":
    main() 