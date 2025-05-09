import json
import os
import glob

# CEFR level to numeric score mapping
CEFR_TO_SCORE = {
    'A1': 1,
    'A2': 2,
    'B1': 3,
    'B2': 4,
    'C1': 5,
    'C2': 6
}

# Numeric score to CEFR level mapping
SCORE_TO_CEFR = {
    1: 'A1',
    2: 'A2',
    3: 'B1',
    4: 'B2',
    5: 'C1',
    6: 'C2'
}

# Weights for different evaluation aspects
WEIGHTS = {
    'grammar': 0.2,
    'coherence': 0.2,
    'range': 0.2,
    'interaction': 0.2,
    'fluency': 0.2
}

def get_numeric_score(cefr_level):
    """Convert CEFR level to numeric score."""
    return CEFR_TO_SCORE.get(cefr_level.upper(), 0)

def get_cefr_level(score):
    """Convert numeric score to CEFR level."""
    # Round to nearest integer
    rounded_score = round(score)
    # Ensure score is within valid range
    clamped_score = max(1, min(6, rounded_score))
    return SCORE_TO_CEFR[clamped_score]

def calculate_weighted_score(evaluation_result):
    """Calculate weighted average score from evaluation results."""
    total_weight = 0
    weighted_sum = 0
    
    for field, weight in WEIGHTS.items():
        if field in evaluation_result:
            cefr_level = evaluation_result[field].get('cefr_level')
            if cefr_level:
                numeric_score = get_numeric_score(cefr_level)
                weighted_sum += numeric_score * weight
                total_weight += weight
    
    if total_weight == 0:
        return 0
        
    return weighted_sum / total_weight

def evaluate_overall(json_file_path):
    """Calculate overall CEFR level from evaluation results JSON file."""
    try:
        with open(json_file_path, 'r') as f:
            evaluation_result = json.load(f)
            
        weighted_score = calculate_weighted_score(evaluation_result)
        overall_cefr = get_cefr_level(weighted_score)
        
        return {
            'weighted_score': round(weighted_score, 2),
            'overall_cefr_level': overall_cefr
        }
        
    except Exception as e:
        return {
            'error': f"Error processing evaluation: {str(e)}"
        }

def evaluate_directory(directory_path):
    """Calculate overall CEFR level for all JSON files in a directory."""
    results = {}
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        result = evaluate_overall(json_file)
        results[file_name] = result
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python overall_score_weighted.py <path>")
        print("Path can be either a JSON file or a directory containing JSON files")
        sys.exit(1)
        
    path = sys.argv[1]
    if os.path.isdir(path):
        # Process directory
        results = evaluate_directory(path)
        print(json.dumps(results, indent=2))
    else:
        # Process single file
        result = evaluate_overall(path)
        print(json.dumps(result, indent=2))

