import sys
sys.path.append("..")

from evaluator.evaluators import FluencyEvaluator
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Create evaluator instance
    evaluator = FluencyEvaluator()
    
    # Sample text for CEFR testing with varying fluency levels
    sample_text = """
    User: I've been working on implementing a sophisticated machine learning algorithm that demonstrates comprehensive understanding of neural networks. The analytical approach to data processing has yielded remarkable results while maintaining computational efficiency.
    
    NPC: That's fascinating! Could you tell me more about the specific techniques you're using?
    
    User: Certainly! The model's performance is consistently robust across various datasets, showcasing its adaptability and reliability. I've developed an innovative methodology that has significantly advanced the field, while the theoretical framework provides a solid foundation for future developments.
    
    NPC: How do you handle different types of data inputs?
    
    User: We've implemented a flexible preprocessing pipeline that can handle diverse data formats. The system employs advanced feature extraction techniques and adaptive learning rates, which have proven particularly effective in real-world applications. The team is working on improving the algorithm further.
    """
    
    # Optional fluency metrics
    fluency_metrics = {
        "pause_frequency": "Low",
        "avg_pause_duration": "0.5 seconds",
        "speaking_rate": "150 words per minute"
    }
    
    # Test with the sample text
    print("Evaluating text:")
    print("-" * 50)
    print(sample_text)
    print("-" * 50)
    
    # Run evaluation with fluency metrics
    result = evaluator.evaluate(sample_text, **fluency_metrics)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"CEFR Level: {result['cefr_level']}")
    print("\nReasoning:")
    print(result['reasoning'])

if __name__ == "__main__":
    main() 