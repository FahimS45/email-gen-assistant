"""
llm_judge.py

Custom Metric 2: Tone Adherence Score
This module implements an LLM-as-judge approach to evaluate how well the generated email's tone matches the user's 
requested tone. It defines a clear rubric for scoring, constructs a system prompt to guide the judge model, 
and includes robust parsing and error handling to ensure reliable metric computation even if the judge's response is imperfect.
"""

import json
import re

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator


# Pydantic models

class JudgeRawResponse(BaseModel):
    """Shape of the JSON the LLM judge is asked to return."""
    score: int = Field(ge=1, le=5)
    justification: str = "No justification provided."

    @field_validator("score", mode="before")
    @classmethod
    def coerce_score(cls, v):
        """Coerce string digits and clamp to [1, 5]."""
        try:
            return max(1, min(5, int(v)))
        except (TypeError, ValueError):
            return 3  # neutral fallback


class ToneAdherenceResult(BaseModel):
    """Full result for Metric 2: Tone Adherence."""
    raw_score: int = Field(ge=1, le=5, description="LLM judge score on 1-5 Likert scale.")
    normalized_score: float = Field(ge=0.0, le=1.0, description="raw_score normalized to [0, 1].")
    justification: str = ""

    @model_validator(mode="before")
    @classmethod
    def compute_normalized(cls, values):
        """Auto-compute normalized_score from raw_score if not explicitly set."""
        if isinstance(values, dict):
            raw = values.get("raw_score", 3)
            try:
                raw_int = max(1, min(5, int(raw)))
            except (TypeError, ValueError):
                raw_int = 3
            values["raw_score"] = raw_int
            # Allow explicit override but default to the formula
            if "normalized_score" not in values or values["normalized_score"] is None:
                values["normalized_score"] = round((raw_int - 1) / 4, 3)
        return values


# Judge prompts

_JUDGE_SYSTEM_PROMPT = """\
You are an expert writing evaluator specializing in tone analysis. You will \
be given a target tone and a generated email. Rate how well the email's actual \
tone matches the target tone using this rubric:

5 = Tone is unmistakable and consistent throughout; word choice, sentence \
structure, and register are all clearly aligned with the target tone.
4 = Tone is mostly aligned with only minor lapses.
3 = Tone is mixed/inconsistent, or generic/neutral rather than clearly \
hitting the target.
2 = Tone is mostly wrong, with only incidental alignment.
1 = Tone is opposite or unrelated to what was requested.

Respond with ONLY a JSON object (no markdown fences, no preamble), shaped \
exactly like:
{"score": <integer 1-5>, "justification": "one to two short sentences"}
"""


def _build_judge_user_prompt(email_output: str, target_tone: str) -> str:
    return (
        f"TARGET TONE: {target_tone}\n\n"
        f"GENERATED EMAIL:\n\"\"\"\n{email_output}\n\"\"\"\n\n"
        f"Rate how well this email's tone matches \"{target_tone}\" using the rubric."
    )


# Scorer

async def score_tone_adherence(
    email_output: str,
    target_tone: str,
    judge_client: AsyncOpenAI,
    judge_model: str,
) -> ToneAdherenceResult:
    """
    Uses an LLM-as-judge to rate tone adherence on a 1-5 scale.

    Args:
        email_output: the clean generated email (CoT already stripped).
        target_tone: the tone the user requested (e.g. "formal").
        judge_client: an AsyncOpenAI-compatible client for the judge model.
        judge_model: model name string to use for judging.

    Returns:
        ToneAdherenceResult with raw 1-5 score, normalized 0-1 score,
        and a short justification string for transparency / debugging.
    """
    response = await judge_client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_judge_user_prompt(email_output, target_tone)},
        ],
        temperature=0.0,
        max_tokens=300,
    )

    raw = response.choices[0].message.content or "{}"
    raw_clean = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed_json = json.loads(raw_clean)
        result = ToneAdherenceResult(
            raw_score=parsed_json.get("score", 3),
            justification=parsed_json.get("justification", ""),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        result = ToneAdherenceResult(
            raw_score=3,
            justification="Judge response unparsable; defaulted to neutral score.",
        )

    return result