import sys
sys.path.append("..")

import unittest
from unittest.mock import patch, MagicMock
from evaluator.evaluators import GrammarEvaluator

class TestGrammarEvaluator(unittest.TestCase):
    """Unit tests for the GrammarEvaluator class with CEFR-based approach"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = GrammarEvaluator()
        
        # Sample text for CEFR testing with grammar errors
        self.sample_text = """
        User: I've been working on implementing a sophisticated machine learning algorithm that demonstrates comprehensive understanding of neural networks. The analytical approach to data processing has yielded remarkable results while maintaining computational efficiency.
        
        NPC: That's fascinating! Could you tell me more about the specific techniques you're using?
        
        User: Certainly! The model's performance are consistently robust across various datasets, showcasing it's adaptability and reliability. I've developed an innovative methodology that has significantly advanced the field, while the theoretical framework provide a solid foundation for future developments.
        
        NPC: How do you handle different types of data inputs?
        
        User: We've implemented a flexible preprocessing pipeline that can handle diverse data formats. The system employ advanced feature extraction techniques and adaptive learning rates, which have proven particularly effective in real-world applications. The team are working on improving the algorithm further.
        """
    
    @patch('evaluator.evaluators.LLMClient')
    def test_evaluate_with_errors(self, mock_llm):
        """Test evaluation of text with some grammar errors"""
        # Mock LLM response with some errors
        mock_response = {
            "errors": [
                {
                    "category": "1. Subject-Verb Agreement",
                    "location": "The model's performance are consistently robust",
                    "correction": "The model's performance is consistently robust",
                    "explanation": "Singular subject requires singular verb"
                },
                {
                    "category": "5. Pronoun Reference",
                    "location": "showcasing it's adaptability",
                    "correction": "showcasing its adaptability",
                    "explanation": "Incorrect use of 'it's' (contraction) instead of 'its' (possessive)"
                },
                {
                    "category": "1. Subject-Verb Agreement",
                    "location": "the theoretical framework provide",
                    "correction": "the theoretical framework provides",
                    "explanation": "Singular subject requires singular verb"
                },
                {
                    "category": "1. Subject-Verb Agreement",
                    "location": "The system employ advanced",
                    "correction": "The system employs advanced",
                    "explanation": "Singular subject requires singular verb"
                },
                {
                    "category": "1. Subject-Verb Agreement",
                    "location": "The team are working",
                    "correction": "The team is working",
                    "explanation": "Collective noun requires singular verb"
                }
            ],
            "cefr_level": "B1",
            "reasoning": "Uses reasonably accurately a repertoire of frequently used 'routines' and patterns associated with more predictable situations."
        }
        
        # Configure mock
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = '{"errors": [{"category": "1. Subject-Verb Agreement", "location": "The model\'s performance are consistently robust", "correction": "The model\'s performance is consistently robust", "explanation": "Singular subject requires singular verb"}, {"category": "5. Pronoun Reference", "location": "showcasing it\'s adaptability", "correction": "showcasing its adaptability", "explanation": "Incorrect use of \'it\'s\' (contraction) instead of \'its\' (possessive)"}, {"category": "1. Subject-Verb Agreement", "location": "the theoretical framework provide", "correction": "the theoretical framework provides", "explanation": "Singular subject requires singular verb"}, {"category": "1. Subject-Verb Agreement", "location": "The system employ advanced", "correction": "The system employs advanced", "explanation": "Singular subject requires singular verb"}, {"category": "1. Subject-Verb Agreement", "location": "The team are working", "correction": "The team is working", "explanation": "Collective noun requires singular verb"}], "cefr_level": "B1", "reasoning": "Uses reasonably accurately a repertoire of frequently used \'routines\' and patterns associated with more predictable situations."}'
        mock_llm.return_value = mock_llm_instance
        
        # Run evaluation
        result = self.evaluator.evaluate(self.sample_text)
        
        # Assertions
        self.assertEqual(result["cefr_level"], "B1")
        self.assertEqual(result["num_errors"], 5)
        self.assertEqual(len(result["errors"]), 5)
        self.assertIn("reasoning", result)
        self.assertIn("Uses reasonably accurately", result["reasoning"])
    
    @patch('evaluator.evaluators.LLMClient')
    def test_evaluate_no_errors(self, mock_llm):
        """Test evaluation of text with no errors"""
        # Mock LLM response with no errors
        mock_response = {
            "errors": [],
            "cefr_level": "C2",
            "reasoning": "Maintains consistent grammatical control of complex language, even while attention is otherwise engaged (e.g. in forward planning, in monitoring others' reactions)."
        }
        
        # Configure mock
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = '{"errors": [], "cefr_level": "C2", "reasoning": "Maintains consistent grammatical control of complex language, even while attention is otherwise engaged (e.g. in forward planning, in monitoring others\' reactions)."}'
        mock_llm.return_value = mock_llm_instance
        
        # Run evaluation
        result = self.evaluator.evaluate(self.sample_text)
        
        # Assertions
        self.assertEqual(result["cefr_level"], "C2")
        self.assertEqual(result["num_errors"], 0)
        self.assertEqual(len(result["errors"]), 0)
        self.assertIn("reasoning", result)
        self.assertIn("Maintains consistent grammatical control", result["reasoning"])
    
    @patch('evaluator.evaluators.LLMClient')
    def test_error_handling(self, mock_llm):
        """Test error handling in the evaluator"""
        # Configure mock to raise an exception
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.side_effect = Exception("LLM error")
        mock_llm.return_value = mock_llm_instance
        
        # Run evaluation
        result = self.evaluator.evaluate(self.sample_text)
        
        # Assertions for error case
        self.assertEqual(result["cefr_level"], "A1")  # Default to lowest level on error
        self.assertEqual(result["num_errors"], -1)
        self.assertEqual(len(result["errors"]), 0)
        self.assertIn("reasoning", result)
        self.assertIn("Error processing response", result["reasoning"])

if __name__ == "__main__":
    unittest.main() 