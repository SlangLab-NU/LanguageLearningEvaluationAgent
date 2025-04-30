import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, Any

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator.evaluators import (
    GrammarEvaluator,
    CoherenceEvaluator,
    RangeEvaluator,
    InteractionEvaluator,
    FluencyEvaluator
)

def read_transcript(file_path: str) -> str:
    """Read the transcript file and return its contents."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to a file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def evaluate_transcript(transcript_path: str) -> Dict[str, Any]:
    """Run all evaluators on a transcript and return combined results."""
    # Read transcript
    transcript = read_transcript(transcript_path)
    
    # Initialize evaluators
    evaluators = {
        'grammar': GrammarEvaluator(),
        'coherence': CoherenceEvaluator(),
        'range': RangeEvaluator(),
        'interaction': InteractionEvaluator(),
        'fluency': FluencyEvaluator()
    }
    
    # Run evaluations
    results = {}
    all_failed = True  # Track if all evaluations failed
    
    for eval_name, evaluator in evaluators.items():
        try:
            # Run evaluation
            eval_results = evaluator.evaluate(transcript)
            results[eval_name] = eval_results
            all_failed = False  # At least one evaluation succeeded
            
        except Exception as e:
            print(f"Error in {eval_name} evaluation: {str(e)}")
            results[eval_name] = {
                "error": str(e),
                "cefr_level": "A1",
                "reasoning": "Evaluation failed"
            }
    
    return results, all_failed

def main():
    # Get all transcript files
    transcript_dir = Path("data/recordings_wav_processed")
    transcript_files = glob.glob(str(transcript_dir / "*_transcript.txt"))
    
    # Create results directory if it doesn't exist
    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each transcript
    for transcript_path in transcript_files:
        # Get base filename without extension
        base_name = Path(transcript_path).stem.replace("_transcript", "")
        
        # Check if results already exist (both txt and json)
        txt_output_path = results_dir / f"{base_name}_result.txt"
        json_output_path = results_dir / f"{base_name}_result.json"
        
        if txt_output_path.exists() or json_output_path.exists():
            print(f"Skipping {transcript_path} - already processed")
            continue
            
        print(f"Processing {transcript_path}...")
        
        # Run evaluation
        results, all_failed = evaluate_transcript(transcript_path)
        
        # Only save results if not all evaluations failed
        if not all_failed:
            save_results(results, json_output_path)  # Save as JSON by default
            print(f"Results saved to {json_output_path}")
        else:
            print(f"Skipping saving results for {transcript_path} - all evaluations failed")

if __name__ == "__main__":
    main()
