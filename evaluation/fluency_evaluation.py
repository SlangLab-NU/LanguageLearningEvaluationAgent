#!/usr/bin/env python3
"""
Fluency Evaluation Script

This script reads speech metrics from a JSON file, uses the FluencyEvaluator
to evaluate all recordings with "USER" in their filenames, and saves the results
to the audio_results directory.
"""
import os
import sys
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from evaluator.evaluators import FluencyEvaluator
from utils.llm import OpenAIClientLLM


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_speech_metrics(metrics_file: str) -> List[Dict[str, Any]]:
    """
    Load speech metrics from a JSON file.
    
    Args:
        metrics_file: Path to the JSON file containing speech metrics
        
    Returns:
        List of dictionaries containing speech metrics
    """
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        logger.info(f"Loaded {len(metrics)} speech metrics from {metrics_file}")
        return metrics
    except Exception as e:
        logger.error(f"Error loading speech metrics: {e}")
        return []

def get_transcript_file(wav_file: str, recordings_dir: str) -> str:
    """
    Get the corresponding transcript file for a WAV file.
    
    Args:
        wav_file: Name of the WAV file
        recordings_dir: Directory containing the recordings
        
    Returns:
        Path to the transcript file
    """
    # Extract the base name without the _USER.wav suffix
    base_name = wav_file.replace("_USER.wav", "")
    transcript_file = f"{base_name}_transcript.txt"
    transcript_path = os.path.join(recordings_dir, transcript_file)
    
    if os.path.exists(transcript_path):
        return transcript_path
    else:
        logger.warning(f"Transcript file not found: {transcript_path}")
        return None

def read_transcript(transcript_file: str) -> str:
    """
    Read the transcript from a file.
    
    Args:
        transcript_file: Path to the transcript file
        
    Returns:
        Transcript text
    """
    try:
        with open(transcript_file, 'r') as f:
            transcript = f.read()
        return transcript
    except Exception as e:
        logger.error(f"Error reading transcript: {e}")
        return ""

def evaluate_fluency(
    transcript: str,
    pause_frequency: float,
    avg_pause_duration: float,
    speaking_rate: float
) -> Dict[str, Any]:
    """
    Evaluate the fluency of a transcript using the FluencyEvaluator.
    
    Args:
        transcript: Transcript text
        pause_frequency: Number of pauses per minute
        avg_pause_duration: Average duration of pauses in seconds
        speaking_rate: Words per minute
        
    Returns:
        Dictionary containing fluency evaluation results
    """
    try:
        # Initialize the FluencyEvaluator with OpenAI
        evaluator = FluencyEvaluator(llm_class=OpenAIClientLLM)
        
        # Evaluate the transcript
        result = evaluator.evaluate(
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

def main():
    """Main function to run the fluency evaluation."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    metrics_file = os.path.join(base_dir, "evaluation", "audio_results", "speech_metrics.json")
    recordings_dir = os.path.join(base_dir, "data", "recordings_wav_processed")
    results_dir = os.path.join(base_dir, "evaluation", "audio_results")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load speech metrics
    metrics = load_speech_metrics(metrics_file)
    if not metrics:
        logger.error("No speech metrics found. Exiting.")
        return
    
    # Initialize results list
    results = []
    
    # Process each recording
    for metric in metrics:
        wav_file = metric["file"]
        
        # Skip if not a USER recording
        if "_USER.wav" not in wav_file:
            continue
        
        logger.info(f"Processing {wav_file}...")
        
        # Get transcript file
        transcript_file = get_transcript_file(wav_file, recordings_dir)
        if not transcript_file:
            continue
        
        # Read transcript
        transcript = read_transcript(transcript_file)
        if not transcript:
            continue
        
        # Evaluate fluency
        evaluation = evaluate_fluency(
            transcript=transcript,
            pause_frequency=metric["pause_frequency"],
            avg_pause_duration=metric["avg_pause_duration"],
            speaking_rate=metric["words_per_minute"]
        )
        
        # Add to results
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
    
    # Save results
    results_file = os.path.join(results_dir, "fluency_evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved {len(results)} fluency evaluation results to {results_file}")

if __name__ == "__main__":
    main()
