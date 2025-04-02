import sys
sys.path.append("..")

from evaluator.evaluators import GrammarEvaluator
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Create evaluator instance
    evaluator = GrammarEvaluator()
    
    # Sample text for CEFR testing with grammar errors
    sample_text = """
    User: I've been working on implementing a sophisticated machine learning algorithm that demonstrates comprehensive understanding of neural networks. The analytical approach to data processing has yielded remarkable results while maintaining computational efficiency.
    
    NPC: That's fascinating! Could you tell me more about the specific techniques you're using?
    
    User: Certainly! The model's performance are consistently robust across various datasets, showcasing it's adaptability and reliability. I've developed an innovative methodology that has significantly advanced the field, while the theoretical framework provide a solid foundation for future developments.
    
    NPC: How do you handle different types of data inputs?
    
    User: We've implemented a flexible preprocessing pipeline that can handle diverse data formats. The system employ advanced feature extraction techniques and adaptive learning rates, which have proven particularly effective in real-world applications. The team are working on improving the algorithm further.
    """
    
    # Test with the sample text
    print("Evaluating text:")
    print("-" * 50)
    print(sample_text)
    print("-" * 50)
    
    # Run evaluation
    result = evaluator.evaluate(sample_text)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"CEFR Level: {result['cefr_level']}")
    print(f"Number of Errors: {result['num_errors']}")
    print("\nErrors Found:")
    for error in result['errors']:
        print(f"\nCategory: {error['category']}")
        print(f"Location: {error['location']}")
        print(f"Correction: {error['correction']}")
        print(f"Explanation: {error['explanation']}")
    print("\nReasoning:")
    print(result['reasoning'])

if __name__ == "__main__":
    main()