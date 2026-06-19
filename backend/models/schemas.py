"""
schemas.py

Pydantic models for the email generation API.
"""

from pydantic import BaseModel, Field
from typing import Literal

from backend.config import TONE_OPTIONS


ToneLiteral = Literal[tuple(TONE_OPTIONS)]


class EmailRequest(BaseModel):
    intent: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="The core purpose of the email (e.g. 'Follow up after meeting').",
    )
    key_facts: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Bullet points of facts to include in the email.",
    )
    tone: ToneLiteral = Field(
        ...,
        description=f"Desired tone of the email. One of: {', '.join(TONE_OPTIONS)}.",
    )
    model: Literal["openai", "ollama"] = Field(
        default="openai",
        description="Which LLM backend to use.",
    )


class EmailResponse(BaseModel):
    email: str
    model_used: str
    prompt_technique: str = "Role-Playing + Few-Shot + Chain-of-Thought"


class StreamChunk(BaseModel):
    type: Literal["token", "done", "error"]
    content: str


class ToneOptionsResponse(BaseModel):
    tones: list[str] = TONE_OPTIONS