from __future__ import annotations  # for pervious python version e.g. 3.9
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from datasets import Dataset

from utils.llm import LLMClient, OpenAIClientLLM
from .prompt_manager import EvaluationType
import asyncio


class ConversationEvaluator(ABC):
    """Base class for evaluating RAG outputs using LLM-as-a-judge pattern."""

    def __init__(
        self,
        llm_class: type[LLMClient] = None,
        **llm_kwargs
    ):
        self.llm = llm_class(**llm_kwargs) if llm_class else OpenAIClientLLM(**llm_kwargs)


    @abstractmethod
    def pre_process(self, script: str | List[str], **kwargs) -> Any:
        """
        Prepare and format the evaluation input.
        
        Args:
            script: User's script to evaluate
            kwargs: Additional template parameters
            
        Returns:
            Processed data ready for LLM evaluation
        """
        pass

    @abstractmethod
    def call_llm(self, processed_data: Any) -> str:
        """
        Execute the LLM call with the processed evaluation prompt.
        
        Args:
            processed_data: Formatted evaluation prompt from pre_process
            
        Returns:
            Raw LLM response string
        """
        pass

    @abstractmethod
    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        """
        Convert LLM response into evaluation scores.
        
        Args:
            llm_response: Raw response string from LLM
            
        Returns:
            Dictionary of evaluation metrics and scores
        """
        pass

    def evaluate(self, script: str | List[str] = None, **kwargs) -> Dict:
        """
        Main evaluation workflow.
        
        Args:
            script: User's script to evaluate
            kwargs: Additional template parameters
            
        Returns:
            Dictionary of evaluation metrics and scores
        """
        processed_data = self.pre_process(script, **kwargs)
        llm_response = self.call_llm(processed_data)
        return self.post_process(llm_response)