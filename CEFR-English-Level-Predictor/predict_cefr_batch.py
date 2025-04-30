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

def process_folder(input_folder, output_folder, use_plus_levels=False):
    """Process all transcript files in a folder and save CEFR level predictions."""
    # Initialize model
    model = Model("cefr_predictor/models/xgboost.joblib", use_plus_levels=use_plus_levels)
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store results
    results = {}
    
    # Process each file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('_transcript_USER.txt'):
            # Extract participant ID from filename
            participant_id = file_name.split('-')[0]
            
            # Process file
            file_path = os.path.join(input_folder, file_name)
            cefr_level = process_transcript_file(file_path, model)
            
            # Store results
            results[participant_id] = {
                'file_name': file_name,
                'cefr_level': cefr_level
            }
    
    # Save results to JSON file
    output_json = os.path.join(output_folder, 'cefr_scores.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    # Generate summary text file
    output_txt = os.path.join(output_folder, 'cefr_summary.txt')
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('CEFR Level Predictions Summary\n')
        f.write('===========================\n\n')
        for participant_id, data in results.items():
            f.write(f"Participant: {participant_id}\n")
            f.write(f"CEFR Level: {data['cefr_level']}\n")
            f.write('-' * 50 + '\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict CEFR levels from transcript files.')
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='Input folder containing transcript files')
    parser.add_argument('--output', '-o', type=str, default='results',
                      help='Output folder for results (default: results)')
    parser.add_argument('--use-plus-levels', action='store_true',
                      help='Use the original plus-level system instead of standard 6 levels')
    
    args = parser.parse_args()
    process_folder(args.input, args.output, use_plus_levels=args.use_plus_levels)