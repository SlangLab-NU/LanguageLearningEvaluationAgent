import sys
sys.path.append("..")

from evaluator.evaluators import CoherenceEvaluator
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Create evaluator instance
    evaluator = CoherenceEvaluator()
    
    # Sample text with different coherence scenarios
    sample_text = """
    The process of photosynthesis is essential for life on Earth. 
    During this process, plants convert light energy into chemical energy. 
    This energy is stored in the form of glucose, which serves as food for the plant. 
    Additionally, photosynthesis produces oxygen as a byproduct, which is crucial for most living organisms.
    The weather was nice yesterday, and I went for a walk in the park.
    """
    
    print("Evaluating text for coherence:")
    print("-" * 50)
    print(sample_text)
    print("-" * 50)
    
    # Run evaluation
    result = evaluator.evaluate(sample_text)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"Overall Coherence Score: {result['overall_score']}")
    print("\nCriterion Scores:")
    for criterion, score in result['criterion_scores'].items():
        print(f"- {criterion}: {'Pass' if score else 'Fail'}")
    print("\nReasoning:")
    for criterion, reasoning in result['reasoning'].items():
        print(f"- {criterion}: {reasoning}")
    print("\nSummary:")
    print(result['summary'])

if __name__ == "__main__":
    main() 