from __future__ import annotations
from enum import Enum, auto
from typing import Dict, Any
from utils.base import BasePrompt
import logging
logger = logging.getLogger(__name__)

class EvaluationType(BasePrompt):
    """Enumeration of different evaluation prompt types with JSON formatting"""
    GRAMMAR_EVALUATION = {
        'template': (
            "Evaluate the given text for grammatical errors using the following error categories and examples as reference.\n"
            "Text to evaluate: {text}\n\n"
            "Error Categories and Examples:\n{criteria}\n\n"
            "Please analyze the text and identify any grammatical errors, categorizing them according to the provided categories.\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Subject-Verb Agreement\n"
            "   Incorrect: The team are playing well.\n"
            "   Correct: The team is playing well.\n\n"
            "2. Tense Usage\n"
            "   Incorrect: I have been to Paris last year.\n"
            "   Correct: I went to Paris last year.\n\n"
            "3. Article Usage\n"
            "   Incorrect: I saw elephant at zoo.\n"
            "   Correct: I saw an elephant at the zoo.\n\n"
            "4. Preposition Usage\n"
            "   Incorrect: I'm looking forward meeting you.\n"
            "   Correct: I'm looking forward to meeting you.\n\n"
            "5. Pronoun Reference\n"
            "   Incorrect: When John met Peter, he was happy.\n"
            "   Correct: When John met Peter, John was happy.\n\n"
            "6. Modifier Placement\n"
            "   Incorrect: I only ate the sandwich.\n"
            "   Correct: I ate only the sandwich.\n\n"
            "7. Parallel Structure\n"
            "   Incorrect: She likes reading, writing, and to dance.\n"
            "   Correct: She likes reading, writing, and dancing.\n\n"
            "8. Countable/Uncountable Nouns\n"
            "   Incorrect: I have many informations.\n"
            "   Correct: I have much information.\n\n"
            "9. Conditional Sentences\n"
            "   Incorrect: If I would have known, I would have told you.\n"
            "   Correct: If I had known, I would have told you.\n\n"
            "10. Passive Voice\n"
            "    Incorrect: The book was being read by me.\n"
            "    Correct: I was reading the book.\n\n"
            "11. Gerund vs Infinitive\n"
            "    Incorrect: I enjoy to swim.\n"
            "    Correct: I enjoy swimming.\n\n"
            "12. Word Order\n"
            "    Incorrect: I yesterday went to the store.\n"
            "    Correct: I went to the store yesterday.\n\n"
            "13. Relative Clauses\n"
            "    Incorrect: The man which I met was friendly.\n"
            "    Correct: The man who I met was friendly.\n\n"
            "14. Comparatives and Superlatives\n"
            "    Incorrect: This is more better than that.\n"
            "    Correct: This is better than that.\n\n"
            "15. Modal Verbs\n"
            "    Incorrect: I must to go now.\n"
            "    Correct: I must go now.\n\n"
            "16. Phrasal Verbs\n"
            "    Incorrect: I look forward meeting you.\n"
            "    Correct: I look forward to meeting you.\n\n"
            "17. Reported Speech\n"
            "    Incorrect: He said he will come tomorrow.\n"
            "    Correct: He said he would come tomorrow.\n\n"
            "18. Question Formation\n"
            "    Incorrect: Where you are going?\n"
            "    Correct: Where are you going?\n\n"
            "19. Negation\n"
            "    Incorrect: I don't have no money.\n"
            "    Correct: I don't have any money.\n\n"
            "20. Punctuation\n"
            "    Incorrect: The cat sat on the mat it was warm.\n"
            "    Correct: The cat sat on the mat. It was warm."
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- errors (array of objects, each containing):\n"
            "  - category (string): The error category number and name\n"
            "  - location (string): The specific text segment containing the error\n"
            "  - correction (string): The corrected version\n"
            "  - explanation (string): Brief explanation of the error\n"
            "- summary (string): Overall assessment of the text's grammatical quality\n"
            "Example:\n"
            "```json\n"
            '{"errors": [{"category": "1. Subject-Verb Agreement", "location": "The team are playing", "correction": "The team is playing", "explanation": "Collective noun requires singular verb"}], "summary": "The text has one major grammatical error related to subject-verb agreement."}\n'
            "```"
        )
    }
    
    COHERENCE_EVALUATION = {
        'template': (
            "Evaluate the coherence of the given text based on the following criteria:\n"
            "Text to evaluate: {text}\n\n"
            "Evaluation Criteria:\n{criteria}\n\n"
            "Please analyze the text and provide scores and reasoning for each criterion.\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Completeness\n"
            "   - Does the text fully address the topic or question?\n"
            "   - Are all necessary components present?\n"
            "   - Is there any missing information?\n\n"
            "2. Relevance\n"
            "   - Are all parts of the text related to the main topic?\n"
            "   - Is there any irrelevant information?\n"
            "   - Does each part contribute meaningfully?\n\n"
            "3. Logical Flow\n"
            "   - Is the information organized in a logical sequence?\n"
            "   - Are transitions between ideas smooth?\n"
            "   - Does the text progress naturally?\n"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- overall_score (float): A score between 0 and 1 representing overall coherence\n"
            "- criterion_scores (object): Boolean scores for each criterion\n"
            "  - completeness (bool): Whether the text is complete\n"
            "  - relevance (bool): Whether the text is relevant\n"
            "  - logical_flow (bool): Whether the text has logical flow\n"
            "- reasoning (object): Detailed reasoning for each criterion\n"
            "  - completeness_reasoning (string): Explanation of completeness score\n"
            "  - relevance_reasoning (string): Explanation of relevance score\n"
            "  - logical_flow_reasoning (string): Explanation of logical flow score\n"
            "- summary (string): Overall assessment of the text's coherence\n"
            "Example:\n"
            "```json\n"
            '{"overall_score": 0.85, "criterion_scores": {"completeness": true, "relevance": true, "logical_flow": false}, "reasoning": {"completeness_reasoning": "The text fully addresses the topic and includes all necessary components", "relevance_reasoning": "All parts are related to the main topic", "logical_flow_reasoning": "The text lacks clear transitions between ideas"}, "summary": "The text is mostly coherent but could benefit from better organization and transitions."}\n'
            "```"
        )
    }

    VOCABULARY_EVALUATION = {
        'template': (
            "Evaluate the vocabulary usage in the given text based on the following criteria:\n"
            "Text to evaluate: {text}\n\n"
            "Evaluation Criteria:\n{criteria}\n\n"
            "Please analyze the text and provide scores and reasoning for each criterion.\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Word Variety\n"
            "   - Is there a good mix of different words?\n"
            "   - Are words repeated unnecessarily?\n"
            "   - Is the vocabulary appropriate for the context?\n\n"
            "2. Word Level\n"
            "   - Are advanced or sophisticated words used appropriately?\n"
            "   - Is the vocabulary level consistent throughout?\n"
            "   - Are words used in their correct context?\n\n"
            "3. Word Choice\n"
            "   - Are words chosen for precision and clarity?\n"
            "   - Are there any inappropriate or awkward word choices?\n"
            "   - Do the words effectively convey the intended meaning?\n\n"
            "4. Collocations and Phrases\n"
            "   - Are common word combinations used correctly?\n"
            "   - Are idiomatic expressions used appropriately?\n"
            "   - Are there any unnatural word combinations?\n\n"
            "5. Academic/Technical Vocabulary\n"
            "   - Is domain-specific vocabulary used correctly?\n"
            "   - Are technical terms explained when needed?\n"
            "   - Is the vocabulary level appropriate for the audience?"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- overall_score (float): A score between 0 and 1 representing overall vocabulary quality\n"
            "- criterion_scores (object): Scores for each criterion\n"
            "  - word_variety (float): Score for word variety (0-1)\n"
            "  - word_level (float): Score for word level (0-1)\n"
            "  - word_choice (float): Score for word choice (0-1)\n"
            "  - collocations (float): Score for collocations and phrases (0-1)\n"
            "  - academic_vocab (float): Score for academic/technical vocabulary (0-1)\n"
            "- reasoning (object): Detailed reasoning for each criterion\n"
            "  - word_variety_reasoning (string): Explanation of word variety score\n"
            "  - word_level_reasoning (string): Explanation of word level score\n"
            "  - word_choice_reasoning (string): Explanation of word choice score\n"
            "  - collocations_reasoning (string): Explanation of collocations score\n"
            "  - academic_vocab_reasoning (string): Explanation of academic vocabulary score\n"
            "- vocabulary_features (object): Analysis of vocabulary characteristics\n"
            "  - unique_words (int): Number of unique words\n"
            "  - total_words (int): Total number of words\n"
            "  - advanced_words (list): List of advanced/sophisticated words used\n"
            "  - repeated_words (list): List of words that might be overused\n"
            "- summary (string): Overall assessment of the text's vocabulary quality\n"
            "Example:\n"
            "```json\n"
            '{"overall_score": 0.85, "criterion_scores": {"word_variety": 0.9, "word_level": 0.8, "word_choice": 0.85, "collocations": 0.8, "academic_vocab": 0.85}, "reasoning": {"word_variety_reasoning": "Good mix of vocabulary with minimal repetition", "word_level_reasoning": "Appropriate use of advanced vocabulary", "word_choice_reasoning": "Words chosen with precision", "collocations_reasoning": "Natural word combinations", "academic_vocab_reasoning": "Domain-specific terms used appropriately"}, "vocabulary_features": {"unique_words": 150, "total_words": 200, "advanced_words": ["sophisticated", "comprehensive", "analytical"], "repeated_words": ["important"]}, "summary": "The text demonstrates strong vocabulary usage with good variety and appropriate word choices."}\n'
            "```"
        )
    }

class EvalPromptManager:
    """Manages prompt construction with JSON output formatting"""

    def __init__(self, default_type: EvaluationType = EvaluationType.GRAMMAR_EVALUATION):
        self.default_type = default_type

    def build_prompt(
        self,
        script: str = None,
        eval_type: EvaluationType = None,
        **kwargs
    ) -> str:
        """
        Construct an evaluation prompt with JSON formatting instructions

        Args:
            script: User's script to evaluate
            eval_type: Type of evaluation to perform
            kwargs: Additional template parameters

        Returns:
            Formatted evaluation prompt with JSON instructions
        """
        eval_type = eval_type or self.default_type

        return eval_type.template.format(
            script=script,
            criteria=eval_type.criteria,
            formatter=eval_type.formatter,
            **kwargs
        )
    

# Example usage
if __name__ == "__main__":
    # Create prompt manager with default evaluation type
    pm = EvalPromptManager(default_type=EvaluationType.GRAMMAR_EVALUATION)
    
    # Build a grammar evaluation prompt
    script = "The team are playing well in the tournament."
    
    prompt = pm.build_prompt(
        script=script,
        eval_type=EvaluationType.GRAMMAR_EVALUATION
    )
    
    logger.info("Grammar Evaluation Prompt:")
    logger.info(prompt)
