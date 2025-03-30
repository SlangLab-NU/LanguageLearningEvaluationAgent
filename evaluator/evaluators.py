from __future__ import annotations  # for pervious python version e.g. 3.9

import asyncio
import json
from typing import List, Dict, Union, Any
from evaluator.base_evaluator import ConversationEvaluator
from evaluator.prompt_manager import EvaluationType, EvalPromptManager

from utils.llm import LLMClient

import os
import logging

logger = logging.getLogger(__name__)


class GrammarEvaluator(ConversationEvaluator):
    """
    Evaluates the grammatical correctness of generated text using predefined error categories.
    Based on comprehensive grammar evaluation criteria.
    """

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)

    def pre_process(
        self,
        script: str | List[str],
        **kwargs,
    ) -> str:
        return EvalPromptManager().build_prompt(
            script=script,
            eval_type=EvaluationType.GRAMMAR_EVALUATION,
            text=script,  # The text to evaluate is the script
        )

    def call_llm(self, processed_data: str) -> str:
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, Any]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = (
                llm_response.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(response_text)
            
            # Calculate score based on number of errors
            num_errors = len(result.get("errors", []))
            max_errors = 20  # Maximum number of possible error categories
            grammar_score = max(0, 1 - (num_errors / max_errors))
            
            scores = {
                "grammar_score": grammar_score,
                "num_errors": num_errors,
                "errors": result.get("errors", []),
                "summary": result.get("summary", ""),
                "raw_output": result
            }
            
            return scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing grammar evaluation response: {e}")
            return {
                "grammar_score": -1,
                "num_errors": -1,
                "errors": [],
                "summary": "Error processing response",
                "raw_output": response_text,
            }


class CoherenceEvaluator(ConversationEvaluator):
    """
    Evaluates the coherence of text based on completeness, relevance, and logical flow.
    Provides overall score and boolean scores for each parameter, accompanied by reasoning.
    """

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)

    def pre_process(
        self,
        script: str | List[str],
        **kwargs,
    ) -> str:
        return EvalPromptManager().build_prompt(
            script=script,
            eval_type=EvaluationType.COHERENCE_EVALUATION,
            text=script,  # The text to evaluate is the script
        )

    def call_llm(self, processed_data: str) -> str:
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, Any]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = (
                llm_response.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(response_text)
            
            scores = {
                "overall_score": result.get("overall_score", 0.0),
                "criterion_scores": result.get("criterion_scores", {}),
                "reasoning": result.get("reasoning", {}),
                "summary": result.get("summary", ""),
                "raw_output": result
            }
            
            return scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing coherence evaluation response: {e}")
            return {
                "overall_score": 0.0,
                "criterion_scores": {
                    "completeness": False,
                    "relevance": False,
                    "logical_flow": False
                },
                "reasoning": {
                    "completeness_reasoning": "Error processing response",
                    "relevance_reasoning": "Error processing response",
                    "logical_flow_reasoning": "Error processing response"
                },
                "summary": "Error processing response",
                "raw_output": response_text,
            }


class VocabularyEvaluator(ConversationEvaluator):
    """
    Evaluates the vocabulary usage in text based on word variety, level, choice,
    collocations, and academic/technical vocabulary usage.
    Provides detailed scores and analysis for each criterion.
    """

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)

    def pre_process(
        self,
        script: str | List[str],
        **kwargs,
    ) -> str:
        return EvalPromptManager().build_prompt(
            script=script,
            eval_type=EvaluationType.VOCABULARY_EVALUATION,
            text=script,  # The text to evaluate is the script
        )

    def call_llm(self, processed_data: str) -> str:
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, Any]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = (
                llm_response.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(response_text)
            
            scores = {
                "overall_score": result.get("overall_score", 0.0),
                "criterion_scores": result.get("criterion_scores", {}),
                "reasoning": result.get("reasoning", {}),
                "vocabulary_features": result.get("vocabulary_features", {}),
                "summary": result.get("summary", ""),
                "raw_output": result
            }
            
            return scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing vocabulary evaluation response: {e}")
            return {
                "overall_score": 0.0,
                "criterion_scores": {
                    "word_variety": 0.0,
                    "word_level": 0.0,
                    "word_choice": 0.0,
                    "collocations": 0.0,
                    "academic_vocab": 0.0
                },
                "reasoning": {
                    "word_variety_reasoning": "Error processing response",
                    "word_level_reasoning": "Error processing response",
                    "word_choice_reasoning": "Error processing response",
                    "collocations_reasoning": "Error processing response",
                    "academic_vocab_reasoning": "Error processing response"
                },
                "vocabulary_features": {
                    "unique_words": 0,
                    "total_words": 0,
                    "advanced_words": [],
                    "repeated_words": []
                },
                "summary": "Error processing response",
                "raw_output": response_text,
            }

