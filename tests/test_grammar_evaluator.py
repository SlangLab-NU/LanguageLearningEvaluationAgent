import sys
sys.path.append("..")

from evaluator.evaluators import GrammarEvaluator
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Create evaluator instance
    evaluator = GrammarEvaluator()
    
    # Sample text with grammar errors
    sample_text = """
    The team are playing well in the tournament. 
    I saw elephant at zoo yesterday. 
    She likes reading, writing, and to dance. 
    If I would have known, I would have told you.
    The book was being read by me.
    """
    
    print("Evaluating text with grammar errors:")
    print("-" * 50)
    print(sample_text)
    print("-" * 50)
    
    # Run evaluation
    result = evaluator.evaluate(sample_text)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"Grammar Score: {result['grammar_score']}")
    print(f"Number of Errors: {result['num_errors']}")
    print("\nErrors Found:")
    for error in result['errors']:
        print(f"\nCategory: {error['category']}")
        print(f"Location: {error['location']}")
        print(f"Correction: {error['correction']}")
        print(f"Explanation: {error['explanation']}")
    print("\nSummary:")
    print(result['summary'])

if __name__ == "__main__":
    main()