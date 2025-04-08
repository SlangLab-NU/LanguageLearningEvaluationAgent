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
    
    # Map speaker IDs to custom names if provided
    if speaker_mapping:
        segments_df['speaker_name'] = segments_df['speaker'].map(speaker_mapping)
    else:
        segments_df['speaker_name'] = segments_df['speaker']
    
    # Extract audio for each speaker
    output_files = {}
    base_filename = Path(audio_path).stem
    
    for speaker_name in segments_df['speaker_name'].unique():
        print(f"Extracting audio for speaker: {speaker_name}")
        
        # Get segments for this speaker
        speaker_segments = segments_df[segments_df['speaker_name'] == speaker_name]
        
        # Create a mask for this speaker
        mask = torch.zeros_like(waveform)
        
        for _, segment in speaker_segments.iterrows():
            start_idx = int(segment['start'] * sample_rate)
            end_idx = int(segment['end'] * sample_rate)
            
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, waveform.shape[1]))
            end_idx = max(0, min(end_idx, waveform.shape[1]))
            
            if start_idx < end_idx:
                mask[:, start_idx:end_idx] = 1
        
        # Apply mask to get speaker audio
        speaker_audio = waveform * mask
        
        # Save to file
        output_file = os.path.join(output_dir, f"{base_filename}_{speaker_name}.wav")
        torchaudio.save(output_file, speaker_audio, sample_rate)
        
        output_files[speaker_name] = output_file
        print(f"Saved audio for {speaker_name} to {output_file}")
    
    return output_files

def main():
    parser = argparse.ArgumentParser(description="Extract Speaker Audio Segments")
    parser.add_argument("audio_path", type=str, help="Path to the original audio file")
    parser.add_argument("diarization_path", type=str, help="Path to the diarization results file")
    parser.add_argument("--output_dir", type=str, help="Directory to save output audio files")
    parser.add_argument("--user_speaker", type=str, default="SPEAKER_00", 
                        help="Speaker ID for USER (default: SPEAKER_00)")
    parser.add_argument("--npc_speaker", type=str, default="SPEAKER_01", 
                        help="Speaker ID for NPC (default: SPEAKER_01)")
    
    args = parser.parse_args()
    
    # Create speaker mapping
    speaker_mapping = {
        args.user_speaker: "USER",
        args.npc_speaker: "NPC"
    }
    
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