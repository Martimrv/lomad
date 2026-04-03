"""
AIS-LLM Explanation Module

Converts numerical AIS predictions into natural language explanations.
Consists of:
- Prompt Encoder: Formats AIS data as structured tokens
- Explanation Generator: Produces human-readable explanations
"""

from .prompt_encoder import PromptEncoder
from .generator import ExplanationGenerator, TemplateGenerator

__all__ = ["PromptEncoder", "ExplanationGenerator", "TemplateGenerator"]

