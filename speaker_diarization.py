#!/usr/bin/env python3
"""
Speaker Diarization Script using WhisperX

This script uses WhisperX to transcribe audio and perform speaker diarization,
specifically identifying USER and NPC speakers in the audio file.
"""

import os
import gc
import torch
import whisperx
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def process_audio(
    audio_path: str,
    output_dir: Optional[str] = None,
    model_name: str = "large-v2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    compute_type: str = "float16",
    batch_size: int = 16,
    min_speakers: int = 2,
    max_speakers: int = 2,
    hf_token: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict:
    """
    Process audio file with WhisperX for transcription and speaker diarization.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save output files
        model_name: Whisper model to use
        device: Device to run inference on
        compute_type: Compute type for inference
        batch_size: Batch size for inference
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        hf_token: Hugging Face token for accessing PyAnnote models
        language: Language code for transcription
        
    Returns:
        Dictionary containing transcription and diarization results
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(audio_path)
    
    print(f"Processing audio file: {audio_path}")
    print(f"Using device: {device}")
    
    # 1. Load Whisper model and transcribe
    print("Loading Whisper model and transcribing audio...")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    
    # Load audio
    audio = whisperx.load_audio(audio_path)
    
    # Transcribe
    result = model.transcribe(audio, batch_size=batch_size)
    print(f"Detected language: {result['language']}")
    
    # Free up GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Align whisper output
    print("Aligning transcription with audio...")
    model_a, metadata = whisperx.load_align_model(
        language_code=language or result["language"], 
        device=device
    )
    result = whisperx.align(
        result["segments"], 
        model_a, 
        metadata, 
        audio, 
        device, 
        return_char_alignments=False
    )
    
    # Free up GPU memory
    del model_a
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Perform speaker diarization
    print("Performing speaker diarization...")
    if hf_token is None:
        print("Warning: No Hugging Face token provided. Diarization may fail.")
        print("Please visit https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("to accept the user agreement and get your token from https://huggingface.co/settings/tokens")
    
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token, 
        device=device
    )
    
    # Perform diarization with specified number of speakers
    diarize_segments = diarize_model(
        audio, 
        min_speakers=min_speakers, 
        max_speakers=max_speakers
    )
    
    # Assign speaker labels to transcription
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # 4. Save results
    output_path = Path(output_dir)
    base_filename = Path(audio_path).stem
    
    # Save transcription with speaker labels
    with open(output_path / f"{base_filename}_transcript.txt", "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            f.write(f"[{start:.2f}s -> {end:.2f}s] {speaker}: {text}\n")
    
    # Save diarization segments
    with open(output_path / f"{base_filename}_diarization.txt", "w", encoding="utf-8") as f:
        for _, row in diarize_segments.iterrows():
            speaker = row["speaker"]
            start = row["start"]
            end = row["end"]
            f.write(f"[{start:.2f}s -> {end:.2f}s] {speaker}\n")
    
    print(f"Results saved to {output_dir}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Speaker Diarization with WhisperX")
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--model", type=str, default="large-v2", help="Whisper model to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on")
    parser.add_argument("--compute_type", type=str, default="float16", 
                        choices=["float16", "float32", "int8"], 
                        help="Compute type for inference")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--min_speakers", type=int, default=2, help="Minimum number of speakers")
    parser.add_argument("--max_speakers", type=int, default=2, help="Maximum number of speakers")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token for accessing PyAnnote models")
    parser.add_argument("--language", type=str, help="Language code for transcription")
    
    args = parser.parse_args()
    
    # Get Hugging Face token from environment if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    result = process_audio(
        audio_path=args.audio_path,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        hf_token=hf_token,
        language=args.language,
    )
    
    # Print summary of speakers
    speakers = set()
    for segment in result["segments"]:
        if "speaker" in segment:
            speakers.add(segment["speaker"])
    
    print("\nSpeaker Summary:")
    for speaker in sorted(speakers):
        print(f"- {speaker}")

if __name__ == "__main__":
    main() 