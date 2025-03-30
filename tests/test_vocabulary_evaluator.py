import sys
sys.path.append("..")

from evaluator.evaluators import VocabularyEvaluator
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Create evaluator instance
    evaluator = VocabularyEvaluator()
    
    # Sample text with different vocabulary scenarios
    sample_text = """
    The sophisticated implementation of the machine learning algorithm demonstrates 
    comprehensive understanding of neural networks. The analytical approach to 
    data processing yields remarkable results, while maintaining computational 
    efficiency. The model's performance is consistently robust across various 
    datasets, showcasing its adaptability and reliability.
    """
    
    print("Evaluating text for vocabulary usage:")
    print("-" * 50)
    print(sample_text)
    print("-" * 50)
    
    # Run evaluation
    result = evaluator.evaluate(sample_text)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"Overall Vocabulary Score: {result['overall_score']}")
    
    print("\nCriterion Scores:")
    for criterion, score in result['criterion_scores'].items():
        print(f"- {criterion}: {score:.2f}")
    
    print("\nReasoning:")
    for criterion, reasoning in result['reasoning'].items():
        print(f"- {criterion}: {reasoning}")
    
    print("\nVocabulary Features:")
    features = result['vocabulary_features']
    print(f"- Unique Words: {features['unique_words']}")
    print(f"- Total Words: {features['total_words']}")
    print(f"- Advanced Words: {', '.join(features['advanced_words'])}")
    print(f"- Repeated Words: {', '.join(features['repeated_words'])}")
    
    print("\nSummary:")
    print(result['summary'])

if __name__ == "__main__":
    main()