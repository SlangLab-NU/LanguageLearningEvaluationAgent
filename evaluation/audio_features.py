import os
import json
import librosa
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """
    Analyzes audio files to extract speech metrics:
    - Pauses per minute
    - Average pause duration
    - Words per minute (estimated)
    """
    
    def __init__(self, sample_rate: int = 22050, name_filter: str = ""):
        """
        Initialize the analyzer with parameters.
        
        Args:
            sample_rate: Sample rate for audio processing
            name_filter: String to filter filenames (only files containing this string will be processed)
        """
        self.sample_rate = sample_rate
        self.name_filter = name_filter
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data and sample rate.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio_data, sr
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            return np.array([]), self.sample_rate
    
    def extract_speech_metrics(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Extract speech metrics from audio data.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary of speech metrics
        """
        try:
            # Calculate energy
            energy = librosa.feature.rms(y=audio_data)[0]
            
            # Find speech segments (high energy)
            speech_threshold = np.mean(energy) * 0.5
            speech_segments = energy > speech_threshold
            
            # Calculate speech duration
            speech_duration = np.sum(speech_segments) / (self.sample_rate / 512)  # Convert to seconds
            
            # Calculate total duration
            total_duration = len(audio_data) / self.sample_rate  # Total audio duration in seconds
            
            # Find pauses (low energy)
            pauses = ~speech_segments
            
            # Calculate pause statistics
            pause_changes = np.diff(pauses.astype(int))
            pause_starts = np.where(pause_changes == 1)[0]
            pause_ends = np.where(pause_changes == -1)[0]
            
            # Handle edge cases
            if len(pause_starts) == 0 or len(pause_ends) == 0:
                return {
                    "pause_frequency": 0,
                    "avg_pause_duration": 0,
                    "words_per_minute": 0
                }
            
            # Ensure we have matching starts and ends
            if pause_starts[0] > pause_ends[0]:
                pause_starts = np.insert(pause_starts, 0, 0)
            if pause_starts[-1] > pause_ends[-1]:
                pause_ends = np.append(pause_ends, len(pauses) - 1)
            
            # Calculate pause durations
            pause_durations = pause_ends - pause_starts
            
            # Filter out very short pauses (less than 0.25 seconds)
            min_pause_frames = int(0.25 * self.sample_rate / 512)  # Assuming hop_length=512 in librosa.feature.rms
            long_pauses = pause_durations[pause_durations > min_pause_frames]
            
            # Calculate pause statistics
            num_pauses = len(long_pauses)
            total_pause_duration = np.sum(long_pauses) / (self.sample_rate / 512)  # Convert to seconds
            avg_pause_duration = total_pause_duration / max(1, num_pauses)
            
            # Calculate pause frequency (pauses per minute)
            pause_frequency = num_pauses / (total_duration / 60) if total_duration > 0 else 0
            
            # Estimate words per minute
            # Average speaking rate is about 150 words per minute
            # We'll use this as a baseline and adjust based on speech duration
            words_per_minute = (speech_duration / total_duration) * 150  # Estimate based on speech proportion
            
            return {
                "pause_frequency": float(pause_frequency),  # pauses per minute
                "avg_pause_duration": float(avg_pause_duration),  # in seconds
                "words_per_minute": float(words_per_minute)  # estimated words per minute
            }
        except Exception as e:
            logger.error(f"Error extracting speech metrics: {e}")
            return {
                "pause_frequency": 0,
                "avg_pause_duration": 0,
                "words_per_minute": 0
            }
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze an audio file and extract speech metrics.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary of speech metrics
        """
        # Load audio
        audio_data, sr = self.load_audio(audio_path)
        
        if len(audio_data) == 0:
            logger.warning(f"Empty audio data for {audio_path}")
            return {
                "file": os.path.basename(audio_path),
                "error": "Failed to load audio file",
                "pause_frequency": 0,
                "avg_pause_duration": 0,
                "words_per_minute": 0
            }
        
        # Extract speech metrics
        metrics = self.extract_speech_metrics(audio_data)
        
        # Return results
        return {
            "file": os.path.basename(audio_path),
            "pause_frequency": metrics["pause_frequency"],
            "avg_pause_duration": metrics["avg_pause_duration"],
            "words_per_minute": metrics["words_per_minute"]
        }
    
    def analyze_directory(self, directory_path: str, output_path: str) -> None:
        """
        Analyze all audio files in a directory and save results to a JSON file.
        Only processes files that have the name_filter in their filename.
        
        Args:
            directory_path: Path to the directory containing audio files
            output_path: Path to save the JSON results
        """
        # Get all audio files with the name_filter in the filename
        audio_files = []
        for file in os.listdir(directory_path):
            if file.endswith(('.wav', '.mp3', '.ogg', '.flac')) and (self.name_filter == "" or self.name_filter in file):
                audio_files.append(os.path.join(directory_path, file))
        
        if not audio_files:
            logger.warning(f"No audio files found in {directory_path}")
            return
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Analyze each audio file
        results = []
        for audio_file in tqdm(audio_files, desc="Analyzing audio files"):
            logger.info(f"Processing file: {audio_file}")
            result = self.analyze_audio(audio_file)
            results.append(result)
            
            # Print metrics for this file
            logger.info(f"Metrics for {os.path.basename(audio_file)}:")
            logger.info(f"  - Pause frequency: {result['pause_frequency']:.2f} pauses per minute")
            logger.info(f"  - Average pause duration: {result['avg_pause_duration']:.2f} seconds")
            logger.info(f"  - Words per minute: {result['words_per_minute']:.2f}")
        
        # Save results to JSON file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract speech metrics from audio files')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save JSON results')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate for audio processing')
    parser.add_argument('--name_filter', type=str, default='', help='String to filter filenames (only files containing this string will be processed)')
    
    args = parser.parse_args()
    
    # Get absolute paths for debugging
    input_dir_abs = os.path.abspath(args.input_dir)
    output_file_abs = os.path.abspath(args.output_file)
    
    logger.info(f"Starting audio analysis with the following parameters:")
    logger.info(f"  - Input directory (absolute): {input_dir_abs}")
    logger.info(f"  - Output file (absolute): {output_file_abs}")
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Check if input directory is a directory
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input path is not a directory: {args.input_dir}")
        return
    
    analyzer = AudioAnalyzer(sample_rate=args.sample_rate, name_filter=args.name_filter)
    analyzer.analyze_directory(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
