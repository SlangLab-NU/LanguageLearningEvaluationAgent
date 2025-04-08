#!/usr/bin/env python3
"""
Batch Process Recordings

This script processes all audio recordings in a specified folder using WhisperX for transcription
and speaker diarization, then extracts audio segments for each speaker.
It automatically skips recordings that have already been processed.
"""

import os
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Optional
from speaker_diarization import process_audio
from extract_speaker_audio import extract_speaker_audio

def get_processed_recordings(output_dir: str) -> List[str]:
    """
    Get a list of recordings that have already been processed.
    
    Args:
        output_dir: Directory containing processed recordings
        
    Returns:
        List of base filenames that have already been processed
    """
    processed = []
    
    # Check for transcript files which indicate processing is complete
    transcript_files = glob.glob(os.path.join(output_dir, "*_transcript.txt"))
    for transcript_file in transcript_files:
        base_filename = os.path.basename(transcript_file).replace("_transcript.txt", "")
        processed.append(base_filename)
    
    return processed

def process_recordings_folder(
    recordings_dir: str,
    output_dir: Optional[str] = None,
    model_name: str = "large-v2",
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
    min_speakers: int = 2,
    max_speakers: int = 2,
    hf_token: Optional[str] = None,
    language: Optional[str] = None,
    user_speaker: str = "SPEAKER_00",
    npc_speaker: str = "SPEAKER_01",
) -> Dict[str, Dict[str, str]]:
    """
    Process all audio recordings in a folder.
    
    Args:
        recordings_dir: Directory containing audio recordings
        output_dir: Directory to save output files (default: same as recordings_dir)
        model_name: Whisper model to use
        device: Device to run inference on
        compute_type: Compute type for inference
        batch_size: Batch size for inference
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        hf_token: Hugging Face token for accessing PyAnnote models
        language: Language code for transcription
        user_speaker: Speaker ID for USER
        npc_speaker: Speaker ID for NPC
        
    Returns:
        Dictionary mapping recording filenames to their processed output files
    """
    # Set output directory to recordings directory if not specified
    if output_dir is None:
        output_dir = recordings_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of already processed recordings
    processed_recordings = get_processed_recordings(output_dir)
    print(f"Found {len(processed_recordings)} already processed recordings")
    
    # Get all WAV files in the recordings directory
    wav_files = glob.glob(os.path.join(recordings_dir, "*.wav"))
    print(f"Found {len(wav_files)} WAV files in {recordings_dir}")
    
    # Process each recording
    results = {}
    for wav_file in wav_files:
        base_filename = os.path.basename(wav_file).replace(".wav", "")
        
        # Skip if already processed
        if base_filename in processed_recordings:
            print(f"Skipping {base_filename} - already processed")
            continue
        
        print(f"\nProcessing {base_filename}...")
        
        # Step 1: Perform speaker diarization
        try:
            diarization_result = process_audio(
                audio_path=wav_file,
                output_dir=output_dir,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                batch_size=batch_size,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                hf_token=hf_token,
                language=language,
            )
            
            # Step 2: Extract speaker audio
            diarization_file = os.path.join(output_dir, f"{base_filename}_diarization.txt")
            
            # Create speaker mapping
            speaker_mapping = {
                user_speaker: "USER",
                npc_speaker: "NPC"
            }
            
            extracted_files = extract_speaker_audio(
                audio_path=wav_file,
                diarization_path=diarization_file,
                output_dir=output_dir,
                speaker_mapping=speaker_mapping,
            )
            
            results[base_filename] = extracted_files
            print(f"Successfully processed {base_filename}")
            
        except Exception as e:
            print(f"Error processing {base_filename}: {str(e)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Batch Process Recordings with WhisperX")
    parser.add_argument("recordings_dir", type=str, help="Directory containing audio recordings")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--model", type=str, default="large-v2", help="Whisper model to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--compute_type", type=str, default="float16", 
                        choices=["float16", "float32", "int8"], 
                        help="Compute type for inference")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--min_speakers", type=int, default=2, help="Minimum number of speakers")
    parser.add_argument("--max_speakers", type=int, default=2, help="Maximum number of speakers")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token for accessing PyAnnote models")
    parser.add_argument("--language", type=str, help="Language code for transcription")
    parser.add_argument("--user_speaker", type=str, default="SPEAKER_00", 
                        help="Speaker ID for USER (default: SPEAKER_00)")
    parser.add_argument("--npc_speaker", type=str, default="SPEAKER_01", 
                        help="Speaker ID for NPC (default: SPEAKER_01)")
    
    args = parser.parse_args()
    
    results = process_recordings_folder(
        recordings_dir=args.recordings_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        hf_token=args.hf_token,
        language=args.language,
        user_speaker=args.user_speaker,
        npc_speaker=args.npc_speaker,
    )
    
    print("\nProcessing Summary:")
    print(f"Total recordings processed: {len(results)}")
    for recording, files in results.items():
        print(f"- {recording}:")
        for speaker, file_path in files.items():
            print(f"  - {speaker}: {file_path}")

if __name__ == "__main__":
    main() 