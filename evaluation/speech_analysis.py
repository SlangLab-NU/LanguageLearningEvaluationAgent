#!/usr/bin/env python3
"""
Speech Analysis Script

This script combines audio feature extraction and fluency evaluation:
1. Extracts speech metrics from audio files (pause frequency, duration, speaking rate)
2. Evaluates fluency using the extracted metrics and transcripts
"""

import os
import sys
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import librosa
import numpy as np
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

from evaluator.evaluators import FluencyEvaluator
from utils.llm import OpenAIClientLLM


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
        self.sample_rate = sample_rate
        self.name_filter = name_filter
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        try:
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio_data, sr
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            return np.array([]), self.sample_rate
    
    def extract_speech_metrics(self, audio_data: np.ndarray) -> Dict[str, Any]:
        try:
            # Calculate energy
            energy = librosa.feature.rms(y=audio_data)[0]
            
            # Find speech segments (high energy)
            speech_threshold = np.mean(energy) * 0.5
            speech_segments = energy > speech_threshold
            
            # Calculate speech duration
            speech_duration = np.sum(speech_segments) / (self.sample_rate / 512)
            total_duration = len(audio_data) / self.sample_rate
            
            # Find pauses (low energy)
            pauses = ~speech_segments
            pause_changes = np.diff(pauses.astype(int))
            pause_starts = np.where(pause_changes == 1)[0]
            pause_ends = np.where(pause_changes == -1)[0]
            
            if len(pause_starts) == 0 or len(pause_ends) == 0:
                return {
                    "pause_frequency": 0,
                    "avg_pause_duration": 0,
                    "words_per_minute": 0
                }
            
            # Ensure matching starts and ends
            if pause_starts[0] > pause_ends[0]:
                pause_starts = np.insert(pause_starts, 0, 0)
            if pause_starts[-1] > pause_ends[-1]:
                pause_ends = np.append(pause_ends, len(pauses) - 1)
            
            pause_durations = pause_ends - pause_starts
            min_pause_frames = int(0.25 * self.sample_rate / 512)
            long_pauses = pause_durations[pause_durations > min_pause_frames]
            
            num_pauses = len(long_pauses)
            total_pause_duration = np.sum(long_pauses) / (self.sample_rate / 512)
            avg_pause_duration = total_pause_duration / max(1, num_pauses)
            pause_frequency = num_pauses / (total_duration / 60) if total_duration > 0 else 0
            words_per_minute = (speech_duration / total_duration) * 150
            
            return {
                "pause_frequency": float(pause_frequency),
                "avg_pause_duration": float(avg_pause_duration),
                "words_per_minute": float(words_per_minute)
            }
        except Exception as e:
            logger.error(f"Error extracting speech metrics: {e}")
            return {
                "pause_frequency": 0,
                "avg_pause_duration": 0,
                "words_per_minute": 0
            }
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
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
        
        metrics = self.extract_speech_metrics(audio_data)
        return {
            "file": os.path.basename(audio_path),
            "pause_frequency": metrics["pause_frequency"],
            "avg_pause_duration": metrics["avg_pause_duration"],
            "words_per_minute": metrics["words_per_minute"]
        }
    
    def analyze_directory(self, directory_path: str, output_path: str) -> List[Dict[str, Any]]:
        audio_files = []
        for file in os.listdir(directory_path):
            if file.endswith(('.wav', '.mp3', '.ogg', '.flac')) and (self.name_filter == "" or self.name_filter in file):
                audio_files.append(os.path.join(directory_path, file))
        
        if not audio_files:
            logger.warning(f"No audio files found in {directory_path}")
            return []
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        results = []
        for audio_file in tqdm(audio_files, desc="Analyzing audio files"):
            logger.info(f"Processing file: {audio_file}")
            result = self.analyze_audio(audio_file)
            results.append(result)
            
            logger.info(f"Metrics for {os.path.basename(audio_file)}:")
            logger.info(f"  - Pause frequency: {result['pause_frequency']:.2f} pauses per minute")
            logger.info(f"  - Average pause duration: {result['avg_pause_duration']:.2f} seconds")
            logger.info(f"  - Words per minute: {result['words_per_minute']:.2f}")
        
        # Save results to JSON file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to {output_path}")
        return results

class FluencyAnalyzer:
    """
    Analyzes fluency based on speech metrics and transcripts.
    """
    
    def __init__(self, recordings_dir: str):
        self.recordings_dir = recordings_dir
        self.evaluator = FluencyEvaluator(llm_class=OpenAIClientLLM)
    
    def get_transcript(self, wav_file: str) -> str:
        base_name = wav_file.replace("_USER.wav", "")
        transcript_file = f"{base_name}_transcript.txt"
        transcript_path = os.path.join(self.recordings_dir, transcript_file)
        
        if not os.path.exists(transcript_path):
            logger.warning(f"Transcript file not found: {transcript_path}")
            return ""
            
        try:
            with open(transcript_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading transcript: {e}")
            return ""
    
    def evaluate_fluency(
        self,
        transcript: str,
        pause_frequency: float,
        avg_pause_duration: float,
        speaking_rate: float
    ) -> Dict[str, Any]:
        try:
            result = self.evaluator.evaluate(
                script=transcript,
                pause_frequency=pause_frequency,
                avg_pause_duration=avg_pause_duration,
                speaking_rate=speaking_rate
            )
            return result
        except Exception as e:
            logger.error(f"Error evaluating fluency: {e}")
            return {
                "cefr_level": "A1",
                "reasoning": f"Error during evaluation: {str(e)}",
                "fluency_features": [],
                "summary": "Evaluation failed"
            }
    
    def analyze_metrics(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        
        for metric in metrics:
            wav_file = metric["file"]
            
            # Skip if not a USER recording
            if "_USER.wav" not in wav_file:
                continue
            
            logger.info(f"Processing {wav_file}...")
            
            transcript = self.get_transcript(wav_file)
            if not transcript:
                continue
            
            evaluation = self.evaluate_fluency(
                transcript=transcript,
                pause_frequency=metric["pause_frequency"],
                avg_pause_duration=metric["avg_pause_duration"],
                speaking_rate=metric["words_per_minute"]
            )
            
            result = {
                "file": wav_file,
                "pause_frequency": metric["pause_frequency"],
                "avg_pause_duration": metric["avg_pause_duration"],
                "words_per_minute": metric["words_per_minute"],
                "cefr_level": evaluation.get("cefr_level", "A1"),
                "reasoning": evaluation.get("reasoning", ""),
                "fluency_features": evaluation.get("fluency_features", []),
                "summary": evaluation.get("summary", "")
            }
            
            results.append(result)
            logger.info(f"Evaluated {wav_file} as {result['cefr_level']}")
        
        return results

def merge_fluency_results(output_dir: str, fluency_results: List[Dict[str, Any]]):
    """
    Merges fluency evaluation results into individual result files.
    
    Args:
        output_dir: Directory containing individual result files
        fluency_results: List of fluency evaluation results
    """
    for result in fluency_results:
        # Get the base filename without _USER.wav
        base_filename = result["file"].replace("_USER.wav", "_result.json")
        result_file = os.path.join(output_dir, base_filename)
        
        if not os.path.exists(result_file):
            logger.warning(f"Result file not found: {result_file}")
            continue
            
        try:
            # Read existing result file
            with open(result_file, 'r') as f:
                existing_result = json.load(f)
            
            # Check if fluency evaluation already exists
            if "fluency" not in existing_result:
                # Create fluency section
                existing_result["fluency"] = {
                    "cefr_level": result["cefr_level"],
                    "pause_frequency": result["pause_frequency"],
                    "avg_pause_duration": result["avg_pause_duration"],
                    "words_per_minute": result["words_per_minute"],
                    "reasoning": result["reasoning"],
                    "fluency_features": result["fluency_features"]
                }
                
                # Write back the updated result
                with open(result_file, 'w') as f:
                    json.dump(existing_result, f, indent=2)
                logger.info(f"Merged fluency results into {base_filename}")
            else:
                logger.info(f"Fluency results already exist in {base_filename}")
                
        except Exception as e:
            logger.error(f"Error merging fluency results for {base_filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Extract speech metrics and evaluate fluency from audio files')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate for audio processing')
    parser.add_argument('--name_filter', type=str, default='', help='String to filter filenames')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output paths
    metrics_file = os.path.join(args.output_dir, "speech_metrics.json")
    evaluation_file = os.path.join(args.output_dir, "fluency_evaluation_results.json")
    
    if os.path.exists(evaluation_file):
        logger.info(f"Found existing fluency evaluation results at {evaluation_file}")
        # Load existing evaluation results
        with open(evaluation_file, 'r') as f:
            evaluation_results = json.load(f)
    else:
        # Step 1: Extract audio features
        logger.info("Step 1: Extracting audio features...")
        audio_analyzer = AudioAnalyzer(sample_rate=args.sample_rate, name_filter=args.name_filter)
        metrics = audio_analyzer.analyze_directory(args.input_dir, metrics_file)
        
        # Step 2: Evaluate fluency
        logger.info("Step 2: Evaluating fluency...")
        fluency_analyzer = FluencyAnalyzer(recordings_dir=args.input_dir)
        evaluation_results = fluency_analyzer.analyze_metrics(metrics)
        
        # Save evaluation results
        with open(evaluation_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
    
    # Step 3: Merge fluency results into individual result files
    logger.info("Step 3: Merging fluency results into individual files...")
    merge_fluency_results(args.output_dir, evaluation_results)
    
    logger.info(f"Analysis complete. Results saved to:")
    logger.info(f"  - Speech metrics: {metrics_file}")
    logger.info(f"  - Fluency evaluation: {evaluation_file}")
    logger.info(f"  - Individual results updated with fluency data")

if __name__ == "__main__":
    main() 