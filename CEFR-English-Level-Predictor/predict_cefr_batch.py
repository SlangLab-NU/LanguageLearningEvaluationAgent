import os
import json
import argparse
from pathlib import Path
from cefr_predictor.inference import Model

def process_transcript_file(file_path, model):
    """Process a single transcript file and return its CEFR level."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    levels, _ = model.predict_decode([text])
    return levels[0]

def process_folder(input_folder, output_folder):
    # Initialize the model once for all predictions
    model = Model("cefr_predictor/models/xgboost.joblib")
    
    # Create results directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # Process all transcript files
    for file_name in os.listdir(input_folder):
        if file_name.endswith('_transcript_USER.txt'):
            file_path = os.path.join(input_folder, file_name)
            participant_id = file_name.split('-')[0]  # Extract P001, P002, etc.
            
            # Get predictions
            level = process_transcript_file(file_path, model)
            
            # Store results
            all_results[participant_id] = {
                'file_name': file_name,
                'cefr_level': level
            }
    
    # Save results to JSON file
    output_file = os.path.join(output_folder, 'cefr_scores.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    
    # Also create a human-readable summary
    summary_file = os.path.join(output_folder, 'cefr_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("CEFR Level Predictions Summary\n")
        f.write("===========================\n\n")
        for participant_id, result in sorted(all_results.items()):
            f.write(f"Participant: {participant_id}\n")
            f.write(f"CEFR Level: {result['cefr_level']}\n")
            f.write("-" * 50 + "\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict CEFR levels from transcript files.')
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='Input folder containing transcript files')
    parser.add_argument('--output', '-o', type=str, default='results',
                      help='Output folder for results (default: results)')
    
    args = parser.parse_args()
    process_folder(args.input, args.output) 