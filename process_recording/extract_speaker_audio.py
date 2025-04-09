#!/usr/bin/env python3
"""
Extract Speaker Audio Segments

This script extracts audio segments for each speaker (USER and NPC) from the original audio file
based on the diarization results from WhisperX.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def extract_speaker_audio(
    audio_path: str,
    diarization_path: str,
    output_dir: Optional[str] = None,
    speaker_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Extract audio segments for each speaker from the original audio file.
    
    Args:
        audio_path: Path to the original audio file
        diarization_path: Path to the diarization results file
        output_dir: Directory to save output audio files
        speaker_mapping: Dictionary mapping speaker IDs to custom names (e.g., {"SPEAKER_00": "USER", "SPEAKER_01": "NPC"})
                       If not provided, will automatically determine based on speaking order (first speaker = USER, second = NPC)
        
    Returns:
        Dictionary mapping speaker names to output audio file paths
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(audio_path)
    
    # Load audio file
    print(f"Loading audio file: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Parse diarization results
    print(f"Parsing diarization results: {diarization_path}")
    segments = []
    
    with open(diarization_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # Parse line like "[0.00s -> 5.23s] SPEAKER_00"
                parts = line.strip().split('] ')
                if len(parts) == 2:
                    time_part = parts[0].strip('[')
                    speaker = parts[1]
                    
                    start_str, end_str = time_part.split(' -> ')
                    start = float(start_str.replace('s', ''))
                    end = float(end_str.replace('s', ''))
                    
                    segments.append({
                        'start': start,
                        'end': end,
                        'speaker': speaker
                    })
    
    # Convert to DataFrame
    segments_df = pd.DataFrame(segments)
    
    # If speaker_mapping is not provided, determine based on speaking order
    if not speaker_mapping:
        # Get unique speakers in order of appearance
        unique_speakers = segments_df['speaker'].unique()
        
        if len(unique_speakers) >= 2:
            # First speaker is USER, second is NPC
            speaker_mapping = {
                unique_speakers[0]: "USER",
                unique_speakers[1]: "NPC"
            }
            print(f"Automatically mapped speakers: {speaker_mapping}")
        else:
            # If only one speaker, map to USER
            speaker_mapping = {unique_speakers[0]: "USER"}
            print(f"Only one speaker detected, mapped to USER: {speaker_mapping}")
    
    # Map speaker IDs to custom names
    segments_df['speaker_name'] = segments_df['speaker'].map(speaker_mapping)
    
    # Extract audio for each speaker
    output_files = {}
    base_filename = Path(audio_path).stem
    
    for speaker_name in segments_df['speaker_name'].unique():
        print(f"Extracting audio for speaker: {speaker_name}")
        
        # Get segments for this speaker
        speaker_segments = segments_df[segments_df['speaker_name'] == speaker_name]
        
        # Create a list to store all audio segments for this speaker
        speaker_audio_segments = []
        
        for _, segment in speaker_segments.iterrows():
            start_idx = int(segment['start'] * sample_rate)
            end_idx = int(segment['end'] * sample_rate)
            
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, waveform.shape[1]))
            end_idx = max(0, min(end_idx, waveform.shape[1]))
            
            if start_idx < end_idx:
                # Extract the segment of audio for this speaker
                segment_audio = waveform[:, start_idx:end_idx]
                speaker_audio_segments.append(segment_audio)
        
        if speaker_audio_segments:
            # Concatenate all segments for this speaker
            speaker_audio = torch.cat(speaker_audio_segments, dim=1)
            
            # Calculate total duration
            total_duration = speaker_audio.shape[1] / sample_rate
            print(f"Total speaking time for {speaker_name}: {total_duration:.2f} seconds")
            
            # Save to file
            output_file = os.path.join(output_dir, f"{base_filename}_{speaker_name}.wav")
            torchaudio.save(output_file, speaker_audio, sample_rate)
            
            output_files[speaker_name] = output_file
            print(f"Saved audio for {speaker_name} to {output_file}")
        else:
            print(f"No audio segments found for {speaker_name}")
    
    return output_files

def main():
    parser = argparse.ArgumentParser(description="Extract Speaker Audio Segments")
    parser.add_argument("audio_path", type=str, help="Path to the original audio file")
    parser.add_argument("diarization_path", type=str, help="Path to the diarization results file")
    parser.add_argument("--output_dir", type=str, help="Directory to save output audio files")
    parser.add_argument("--user_speaker", type=str, 
                        help="Speaker ID for USER (if not specified, first speaker will be USER)")
    parser.add_argument("--npc_speaker", type=str, 
                        help="Speaker ID for NPC (if not specified, second speaker will be NPC)")
    
    args = parser.parse_args()
    
    # Create speaker mapping if specified
    speaker_mapping = None
    if args.user_speaker and args.npc_speaker:
        speaker_mapping = {
            args.user_speaker: "USER",
            args.npc_speaker: "NPC"
        }
        print(f"Using manual speaker mapping: {speaker_mapping}")
    else:
        print("No speaker mapping provided. Will automatically determine USER and NPC based on speaking order.")
    
    output_files = extract_speaker_audio(
        audio_path=args.audio_path,
        diarization_path=args.diarization_path,
        output_dir=args.output_dir,
        speaker_mapping=speaker_mapping,
    )
    
    print("\nExtracted Audio Files:")
    for speaker, file_path in output_files.items():
        print(f"- {speaker}: {file_path}")

if __name__ == "__main__":
    main() 