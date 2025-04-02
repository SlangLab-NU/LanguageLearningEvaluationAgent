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
    Evaluates the grammatical correctness of generated text using predefined error categories
    and assigns a CEFR level based on grammatical control and accuracy.
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
        """Parse JSON response into CEFR level assessment dictionary"""
        try:
            # Clean response and parse JSON
            response_text = (
                llm_response.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(response_text)
            
            # Get the errors for detailed analysis
            errors = result.get("errors", [])
            num_errors = len(errors)
            
            # Determine CEFR level based on number and types of errors
            # This is a simplified approach - in a real implementation, 
            # you might want to consider the types of errors as well
            cefr_level = self._determine_cefr_level(num_errors, errors)
            
            # Generate reasoning based on the errors and CEFR level
            reasoning = self._generate_reasoning(cefr_level, errors)
            
            scores = {
                "cefr_level": cefr_level,
                "num_errors": num_errors,
                "errors": errors,
                "reasoning": reasoning,
                "raw_output": result
            }
            
            return scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing grammar evaluation response: {e}")
            return {
                "cefr_level": "A1",
                "num_errors": -1,
                "errors": [],
                "reasoning": "Error processing response",
                "raw_output": response_text,
            }
    
    def _determine_cefr_level(self, num_errors: int, errors: List[Dict[str, Any]]) -> str:
        """
        Determine the CEFR level based on the number and types of errors.
        
        Args:
            num_errors: Number of grammatical errors found
            errors: List of error objects with details
            
        Returns:
            CEFR level (C2, C1, B2, B1, A2, A1)
        """
        # This is a simplified approach - in a real implementation,
        # you might want to consider the types of errors as well
        
        if num_errors == 0:
            return "C2"  # No errors - highest level
        elif num_errors <= 2:
            return "C1"  # Very few errors - high level
        elif num_errors <= 5:
            return "B2"  # Some errors but not causing misunderstanding
        elif num_errors <= 10:
            return "B1"  # More errors but still using common patterns correctly
        elif num_errors <= 15:
            return "A2"  # Many basic errors
        else:
            return "A1"  # Most basic level with many errors
    
    def _generate_reasoning(self, cefr_level: str, errors: List[Dict[str, Any]]) -> str:
        """
        Generate reasoning for the CEFR level assessment based on the errors found.
        
        Args:
            cefr_level: The determined CEFR level
            errors: List of error objects with details
            
        Returns:
            Detailed reasoning for the CEFR level assessment
        """
        # Base reasoning on CEFR level
        if cefr_level == "C2":
            return "Maintains consistent grammatical control of complex language, even while attention is otherwise engaged (e.g. in forward planning, in monitoring others' reactions)."
        elif cefr_level == "C1":
            return "Consistently maintains a high degree of grammatical accuracy; errors are rare, difficult to spot and generally corrected when they do occur."
        elif cefr_level == "B2":
            return "Shows a relatively high degree of grammatical control. Does not make errors which cause misunderstanding, and can correct most of his/her mistakes."
        elif cefr_level == "B1":
            return "Uses reasonably accurately a repertoire of frequently used 'routines' and patterns associated with more predictable situations."
        elif cefr_level == "A2":
            return "Uses some simple structures correctly, but still systematically makes basic mistakes."
        else:  # A1
            return "Shows only limited control of a few simple grammatical structures and sentence patterns in a memorised repertoire."


class CoherenceEvaluator(ConversationEvaluator):
    """
    Evaluates the coherence of text based on CEFR levels, focusing on the ability to create
    coherent and cohesive discourse using appropriate organisational patterns and connectors.
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
                "cefr_level": result.get("cefr_level", "A1"),
                "reasoning": result.get("reasoning", ""),
                "raw_output": result
            }
            
            return scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing coherence evaluation response: {e}")
            return {
                "cefr_level": "A1",
                "reasoning": "Error processing response",
                "raw_output": response_text,
            }


class RangeEvaluator(ConversationEvaluator):
    """
    Evaluates the language range based on CEFR levels, focusing on the breadth and flexibility
    of language use, including vocabulary range, sentence complexity, and ability to express
    ideas in different ways.
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
            eval_type=EvaluationType.RANGE_EVALUATION,
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
                "cefr_level": result.get("cefr_level", "A1"),
                "reasoning": result.get("reasoning", ""),
                "raw_output": result
            }
            
            return scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing range evaluation response: {e}")
            return {
                "cefr_level": "A1",
                "reasoning": "Error processing response",
                "raw_output": response_text,
            }


class InteractionEvaluator(ConversationEvaluator):
    """
    Evaluates interaction skills in conversations and determines the appropriate CEFR level.
    Provides a single CEFR level assessment with confidence score and supporting evidence.
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
            eval_type=EvaluationType.INTERACTION_EVALUATION,
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
                "cefr_level": result.get("cefr_level", "A1"),
                "confidence_score": result.get("confidence_score", 0.0),
                "reasoning": result.get("reasoning", ""),
                "key_features": result.get("key_features", []),
                "summary": result.get("summary", ""),
                "raw_output": result
            }
            
            return scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing interaction evaluation response: {e}")
            return {
                "cefr_level": "A1",
                "confidence_score": 0.0,
                "reasoning": "Error processing response",
                "key_features": [],
                "summary": "Error processing response",
                "raw_output": response_text,
            }

