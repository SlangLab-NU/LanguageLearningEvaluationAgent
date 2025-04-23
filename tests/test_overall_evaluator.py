import sys
import json
sys.path.append("..")

from evaluator.evaluators import CEFROverallEvaluator
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Create evaluator instance
    evaluator = CEFROverallEvaluator()
    
    # Load sample evaluation results from JSON file
    json_path = "../evaluation/results/P001-com.oculus.vrshell-20240807-093454_result.json"
    try:
        with open(json_path, 'r') as f:
            evaluation_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {json_path}")
        return
    
    print("Testing with evaluation results:")
    print("-" * 50)
    print("Individual Scores:")
    print(f"Grammar: {evaluation_results['grammar']['cefr_level']}")
    print(f"Coherence: {evaluation_results['coherence']['cefr_level']}")
    print(f"Range: {evaluation_results['range']['cefr_level']}")
    print(f"Interaction: {evaluation_results['interaction']['cefr_level']}")
    print(f"Fluency: {evaluation_results['fluency']['cefr_level']}")
    print("-" * 50)
    
    # Run overall evaluation
    result = evaluator.evaluate(evaluation_results)
    
    # Print results
    print("\nFinal CEFR Assessment:")
    print("-" * 50)
    print(f"Overall CEFR Level: {result['cefr_level']}")
    print("\nReasoning:")
    print(result['reasoning'])

if __name__ == "__main__":
    main() 