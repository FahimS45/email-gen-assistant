"""
state.py

Defines the state structure for the email generation pipeline.
"""

from typing import TypedDict, Optional


class EmailState(TypedDict):

    # Inputs 
    intent: str
    key_facts: list[str]
    tone: str
    model: str  # "openai" | "ollama"

    # Intermediate 
    system_prompt: str       # built by prompt_builder node
    user_prompt: str         # built by prompt_builder node
    cot_reasoning: str       # extracted chain-of-thought from the model's raw output

    # Output
    email_output: str        # the final clean email (CoT stripped)
    model_used: str

    # Error handling 
    error: Optional[str]
