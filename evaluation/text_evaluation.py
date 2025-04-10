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
    for eval_name, evaluator in evaluators.items():
        try:
            # Run evaluation
            eval_results = evaluator.evaluate(transcript)
            results[eval_name] = eval_results
            
        except Exception as e:
            print(f"Error in {eval_name} evaluation: {str(e)}")
            results[eval_name] = {
                "error": str(e),
                "cefr_level": "A1",
                "reasoning": "Evaluation failed"
            }
    
    return results

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
        
        # Check if results already exist
        output_path = results_dir / f"{base_name}_result.txt"
        if output_path.exists():
            print(f"Skipping {transcript_path} - already processed")
            continue
            
        print(f"Processing {transcript_path}...")
        
        # Run evaluation
        results = evaluate_transcript(transcript_path)
        
        # Save results
        save_results(results, output_path)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
