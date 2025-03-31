"""Speaker diarization module for separating human and AI agent voices from audio files.

This module provides functionality to split audio files containing conversations between
human speakers and AI agents into separate audio files for each speaker.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torchaudio
from pyannote.audio import Pipeline
from pydantic import BaseModel, Field
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeakerSegment(BaseModel):
    """Model representing a speaker segment in the audio."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    speaker: str = Field(..., description="Speaker identifier")
    confidence: float = Field(..., description="Confidence score for the segment")

class AudioDiarizer:
    """Class for performing speaker diarization on audio files."""
    
    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: Optional[str] = None
    ) -> None:
        """Initialize the AudioDiarizer.
        
        Args:
            hf_token: HuggingFace token for accessing pyannote.audio
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=hf_token
        ).to(self.device)
        
    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and convert to mono.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple containing the audio tensor and sample rate
        """
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform, sample_rate
    
    def diarize(
        self,
        audio_path: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, List[SpeakerSegment]]:
        """Perform speaker diarization on the audio file.
        
        Args:
            audio_path: Path to the input audio file
            output_dir: Directory to save the output audio files
            
        Returns:
            Dictionary mapping speaker labels to their segments
        """
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(audio_path).parent
            
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        waveform, sample_rate = self.load_audio(audio_path)
        
        # Perform diarization
        diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
        
        # Extract segments
        segments: Dict[str, List[SpeakerSegment]] = {}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in segments:
                segments[speaker] = []
            segments[speaker].append(
                SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                    confidence=float(turn.confidence)
                )
            )
            
        # Save individual speaker audio files
        self._save_speaker_audio(
            waveform,
            sample_rate,
            segments,
            output_path,
            Path(audio_path).stem
        )
        
        return segments
    
    def _save_speaker_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        segments: Dict[str, List[SpeakerSegment]],
        output_dir: Path,
        base_filename: str
    ) -> None:
        """Save audio segments for each speaker.
        
        Args:
            waveform: Input audio waveform
            sample_rate: Audio sample rate
            segments: Dictionary of speaker segments
            output_dir: Directory to save output files
            base_filename: Base filename for output files
        """
        for speaker, speaker_segments in segments.items():
            # Create mask for this speaker
            mask = torch.zeros_like(waveform)
            for segment in speaker_segments:
                start_idx = int(segment.start * sample_rate)
                end_idx = int(segment.end * sample_rate)
                mask[:, start_idx:end_idx] = 1
                
            # Apply mask to get speaker audio
            speaker_audio = waveform * mask
            
            # Save to file
            output_file = output_dir / f"{base_filename}_{speaker}.wav"
            torchaudio.save(
                output_file,
                speaker_audio,
                sample_rate
            )
            logger.info(f"Saved audio for speaker {speaker} to {output_file}")

def main() -> None:
    """Main function to demonstrate usage."""
    # Example usage
    audio_path = "data/recordings_wav/P001-com.oculus.vrshell-20240807-093454.wav"
    
    # Initialize diarizer (requires HuggingFace token)
    diarizer = AudioDiarizer(hf_token="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")
    
    # Perform diarization
    segments = diarizer.diarize(audio_path)
    
    # Print results
    for speaker, speaker_segments in segments.items():
        print(f"\nSpeaker {speaker}:")
        for segment in speaker_segments:
            print(f"  {segment.start:.2f}s - {segment.end:.2f}s (confidence: {segment.confidence:.2f})")

if __name__ == "__main__":
    main() 