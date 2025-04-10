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
            "Evaluate grammatical errors of the User using the following error categories and examples as reference.\n"
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
            "- cefr_level (string): The assessed CEFR level (C2, C1, B2, B1, A2, A1)\n"
            "- reasoning (string): Detailed explanation of why this CEFR level was chosen\n"
            "Example:\n"
            "```json\n"
            '{"errors": [{"category": "1. Subject-Verb Agreement", "location": "The team are playing", "correction": "The team is playing", "explanation": "Collective noun requires singular verb"}], "cefr_level": "B2", "reasoning": "Shows a relatively high degree of grammatical control. Does not make errors which cause misunderstanding, and can correct most of his/her mistakes."}\n'
            "```"
        )
    }
    
    COHERENCE_EVALUATION = {
        'template': (
            "Evaluate the coherence of the User and determine its CEFR level based on the following criteria:\n"
            "Text to evaluate: {text}\n\n"
            "CEFR Coherence Criteria:\n{criteria}\n\n"
            "Please analyze the text and determine which CEFR level best describes the coherence demonstrated.\n"
            "{formatter}"
        ),
        'criteria': (
            "C2 Level:\n"
            "- Can create coherent and cohesive discourse making full and appropriate use of a variety of organisational patterns\n"
            "- Can use a wide range of connectors and other cohesive devices effectively\n"
            "- Demonstrates sophisticated control over discourse structure\n\n"
            "C1 Level:\n"
            "- Can produce clear, smoothly-flowing, well-structured speech\n"
            "- Shows controlled use of organisational patterns\n"
            "- Uses connectors and cohesive devices effectively\n\n"
            "B2 Level:\n"
            "- Can use a limited number of cohesive devices to link utterances into clear, coherent discourse\n"
            "- May show some \"jumpiness\" in longer contributions\n"
            "- Maintains basic coherence across the text\n\n"
            "B1 Level:\n"
            "- Can link a series of shorter, discrete simple elements into a connected sequence\n"
            "- Uses basic connectors to create linear sequences of points\n"
            "- Shows basic coherence in shorter texts\n\n"
            "A2 Level:\n"
            "- Can link groups of words with simple connectors like \"and\", \"but\" and \"because\"\n"
            "- Shows limited coherence in very short texts\n"
            "- Uses basic connectors appropriately\n\n"
            "A1 Level:\n"
            "- Can link words or groups of words with very basic linear connectors like \"and\" or \"then\"\n"
            "- Shows minimal coherence\n"
            "- Limited use of connectors"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- cefr_level (string): The assessed CEFR level (A1, A2, B1, B2, C1, C2)\n"
            "- reasoning (string): Detailed explanation of why this CEFR level was chosen, focusing on coherence features\n"
            "Example:\n"
            "```json\n"
            '{"cefr_level": "B2", "reasoning": "The text demonstrates B2-level coherence with appropriate use of cohesive devices, though there are some minor inconsistencies in longer sections. The overall structure is clear but could benefit from more sophisticated connectors."}\n'
            "```"
        )
    }

    VOCABULARY_EVALUATION = {
        'template': (
            "Evaluate the vocabulary of the User based on the following criteria:\n"
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

    INTERACTION_EVALUATION = {
        'template': (
            "Evaluate the interaction skills of the User and determine its CEFR level:\n"
            "Conversation to evaluate: {text}\n\n"
            "CEFR Interaction Criteria:\n{criteria}\n\n"
            "Please analyze the conversation and determine which CEFR level best describes the interaction skills demonstrated.\n"
            "{formatter}"
        ),
        'criteria': (
            "C2 Level:\n"
            "- Can interact with ease and skill\n"
            "- Picks up and uses non-verbal and intonational cues effortlessly\n"
            "- Interweaves contribution into joint discourse naturally\n"
            "- Demonstrates fully natural turn-taking\n"
            "- Shows skillful referencing and allusion making\n\n"
            "C1 Level:\n"
            "- Can select suitable phrases for discourse functions\n"
            "- Prefaces remarks appropriately to get/keep the floor\n"
            "- Relates contributions skillfully to other speakers\n"
            "- Shows good awareness of conversation flow\n\n"
            "B2 Level:\n"
            "- Can initiate discourse and take turns appropriately\n"
            "- Can end conversations when needed\n"
            "- Helps discussion along on familiar topics\n"
            "- Confirms comprehension and invites others in\n\n"
            "B1 Level:\n"
            "- Can initiate, maintain and close simple face-to-face conversations\n"
            "- Handles familiar or personally interesting topics\n"
            "- Can repeat back to confirm mutual understanding\n"
            "- Shows basic conversation management skills\n\n"
            "A2 Level:\n"
            "- Can answer questions and respond to simple statements\n"
            "- Can indicate when following the conversation\n"
            "- Limited ability to keep conversation going independently\n"
            "- Basic interaction skills\n\n"
            "A1 Level:\n"
            "- Can ask and answer questions about personal details\n"
            "- Can interact in a simple way\n"
            "- Communication dependent on repetition and rephrasing\n"
            "- Basic question-answer interaction"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- cefr_level (string): The assessed CEFR level (A1, A2, B1, B2, C1, C2)\n"
            "- confidence_score (float): Confidence in the assessment (0-1)\n"
            "- reasoning (string): Detailed explanation of why this CEFR level was chosen\n"
            "- key_features (array): List of key interaction features observed that support this level\n"
            "- summary (string): Brief summary of the interaction assessment\n"
            "Example:\n"
            "```json\n"
            '{"cefr_level": "B2", "confidence_score": 0.85, "reasoning": "The conversation demonstrates strong B2-level interaction skills, particularly in initiating discourse and managing turn-taking. While there are some sophisticated elements, the interaction lacks the natural flow and nuanced referencing typical of C1 level.", "key_features": ["Appropriate turn-taking", "Good topic management", "Clear conversation structure", "Effective comprehension checks"], "summary": "Strong B2-level interaction with clear structure and good turn management."}\n'
            "```"
        )
    }

    RANGE_EVALUATION = {
        'template': (
            "Evaluate the language range of the User and determine its CEFR level:\n"
            "Text to evaluate: {text}\n\n"
            "CEFR Range Criteria:\n{criteria}\n\n"
            "Please analyze the text and determine which CEFR level best describes the language range demonstrated.\n"
            "{formatter}"
        ),
        'criteria': (
            "C2 Level:\n"
            "- Has a good command of idiomatic expressions and colloquialisms\n"
            "- Shows awareness of connotative levels of meaning\n"
            "- Can vary formulation to avoid frequent repetition\n"
            "- Demonstrates sophisticated vocabulary range\n\n"
            "C1 Level:\n"
            "- Has a good range of vocabulary for matters relating to their field\n"
            "- Can vary formulation to avoid frequent repetition\n"
            "- Shows good command of idiomatic expressions and colloquialisms\n"
            "- Demonstrates broad vocabulary range\n\n"
            "B2 Level:\n"
            "- Has sufficient vocabulary to express him/herself with some circumlocutions\n"
            "- Shows good range of vocabulary for matters related to their field\n"
            "- Can use some idiomatic expressions\n"
            "- Demonstrates adequate vocabulary range\n\n"
            "B1 Level:\n"
            "- Has enough vocabulary to express him/herself with some circumlocutions\n"
            "- Shows good range of vocabulary for matters related to their field\n"
            "- Can use some idiomatic expressions\n"
            "- Demonstrates basic vocabulary range\n\n"
            "A2 Level:\n"
            "- Has sufficient vocabulary for routine tasks\n"
            "- Shows good range of vocabulary for matters related to their field\n"
            "- Can use some idiomatic expressions\n"
            "- Demonstrates limited vocabulary range\n\n"
            "A1 Level:\n"
            "- Has a basic range of simple phrases and sentences\n"
            "- Shows limited vocabulary range\n"
            "- Can use some basic idiomatic expressions\n"
            "- Demonstrates very limited vocabulary range"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- cefr_level (string): The assessed CEFR level (A1, A2, B1, B2, C1, C2)\n"
            "- reasoning (string): Detailed explanation of why this CEFR level was chosen\n"
            "- vocabulary_features (array): List of vocabulary features observed that support this level\n"
            "- summary (string): Brief summary of the language range assessment\n"
            "Example:\n"
            "```json\n"
            '{"cefr_level": "B2", "reasoning": "The text demonstrates B2-level language range with sufficient vocabulary to express ideas with some circumlocutions. While there is good use of idiomatic expressions, the vocabulary lacks the sophistication and nuance typical of C1 level.", "vocabulary_features": ["Good range of vocabulary", "Appropriate use of idiomatic expressions", "Some circumlocutions", "Field-specific terminology"], "summary": "Strong B2-level language range with good vocabulary variety."}\n'
            "```"
        )
    }

    FLUENCY_EVALUATION = {
        'template': (
            "Evaluate the fluency of the User and determine its CEFR level:\n"
            "Text to evaluate: {text}\n\n"
            "Additional metrics:\n"
            "- Pause frequency: {pause_frequency}\n"
            "- Average pause duration: {avg_pause_duration}\n"
            "- Speaking rate: {speaking_rate}\n\n"
            "CEFR Fluency Criteria:\n{criteria}\n\n"
            "Please analyze the text and determine which CEFR level best describes the fluency demonstrated.\n"
            "{formatter}"
        ),
        'criteria': (
            "C2 Level:\n"
            "- Can express him/herself spontaneously at length with a natural colloquial flow\n"
            "- Avoids or backtracks around any difficulty so smoothly that the interlocutor is hardly aware of it\n"
            "- Demonstrates effortless fluency with minimal pausing\n\n"
            "C1 Level:\n"
            "- Can express him/herself fluently and spontaneously, almost effortlessly\n"
            "- Only a conceptually difficult subject can hinder a natural, smooth flow of language\n"
            "- Shows high fluency with occasional strategic pausing\n\n"
            "B2 Level:\n"
            "- Can produce stretches of language with a fairly even tempo\n"
            "- May be hesitant as he/she searches for patterns and expressions\n"
            "- Shows few noticeably long pauses\n\n"
            "B1 Level:\n"
            "- Can keep going comprehensibly, even though pausing for grammatical and lexical planning and repair is very evident\n"
            "- Especially noticeable in longer stretches of free production\n"
            "- Shows moderate fluency with regular pausing\n\n"
            "A2 Level:\n"
            "- Can make him/herself understood in very short utterances\n"
            "- Pauses, false starts and reformulation are very evident\n"
            "- Shows limited fluency with frequent pausing\n\n"
            "A1 Level:\n"
            "- Can manage very short, isolated, mainly pre-packaged utterances\n"
            "- Much pausing to search for expressions, to articulate less familiar words, and to repair communication\n"
            "- Shows minimal fluency with extensive pausing"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- cefr_level (string): The assessed CEFR level (A1, A2, B1, B2, C1, C2)\n"
            "- reasoning (string): Detailed explanation of why this CEFR level was chosen\n"
            "- fluency_features (array): List of fluency features observed that support this level\n"
            "- summary (string): Brief summary of the fluency assessment\n"
            "Example:\n"
            "```json\n"
            '{"cefr_level": "B2", "reasoning": "The text demonstrates B2-level fluency with a fairly even tempo and few noticeably long pauses. While there is some hesitation when searching for expressions, the overall flow is maintained.", "fluency_features": ["Even tempo", "Few long pauses", "Occasional hesitation", "Comprehensible flow"], "summary": "Strong B2-level fluency with good flow and minimal disruption."}\n'
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
