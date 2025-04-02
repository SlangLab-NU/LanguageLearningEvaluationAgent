import sys
sys.path.append("..")

from evaluator.evaluators import InteractionEvaluator
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Create evaluator instance
    evaluator = InteractionEvaluator()
    
    # Sample conversation
    sample_conversation = """
    User: What inspired you to become a writer and what kind of novels do you enjoy writing the most?
    NPC: Becoming a writer was a natural calling for me. I love weaving stories that transport readers to new worlds. As for my favorite genre, I enjoy writing historical fiction the most. It's like time travel with words! What about you, Tony? If you had a superpower, what would it be?
    User: How has growing up in Brentwood influenced your life and your work?
    NPC: Growing up in Brentwood has been a joy! The town's charm and delicious food have influenced my imagination and writing. How about you, Tony? What's your favorite superpower?
    """
    
    print("Evaluating conversation for interaction skills:")
    print("-" * 50)
    print(sample_conversation)
    print("-" * 50)
    
    # Run evaluation
    result = evaluator.evaluate(sample_conversation)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"CEFR Level: {result['cefr_level']}")
    print(f"Confidence Score: {result['confidence_score']}")
    print("\nKey Features Observed:")
    for feature in result['key_features']:
        print(f"- {feature}")
    print("\nReasoning:")
    print(result['reasoning'])
    print("\nSummary:")
    print(result['summary'])

if __name__ == "__main__":
    main() 