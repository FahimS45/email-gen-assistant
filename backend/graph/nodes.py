"""
nodes.py

Three LangGraph nodes that form the email generation pipeline.
"""

import re
from openai import AsyncOpenAI

from backend.graph.state import EmailState
from backend.graph.prompt_engineering import build_prompts, extract_cot_and_email
from backend.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_API_KEY,
    TONE_OPTIONS,
)


# NODE 1: Input Validator

async def validate_input(state: EmailState) -> dict:
    """
    Lightweight validation before spending tokens.
    Checks that intent is non-trivial and key_facts is non-empty.
    """
    intent = state.get("intent", "").strip()
    key_facts = state.get("key_facts", [])
    tone = state.get("tone", "").strip()

    if not intent or len(intent) < 5:
        return {"error": "Intent is too short or missing. Please describe the email's purpose."}

    cleaned_facts = [f.strip() for f in key_facts if f.strip()]
    if not cleaned_facts:
        return {"error": "At least one key fact is required."}

    valid_tones = set(TONE_OPTIONS)
    if tone.lower() not in valid_tones:
        return {"error": f"Invalid tone '{tone}'. Choose from: {', '.join(sorted(valid_tones))}"}

    return {
        "intent": intent,
        "key_facts": cleaned_facts,
        "tone": tone.lower(),
        "error": None,
    }

# NODE 2: Prompt Builder

async def build_prompt(state: EmailState) -> dict:
    """
    Constructs the system + user prompts using the three-technique strategy
    (Role-Playing + Few-Shot + Chain-of-Thought) defined in prompt_engineering.py.
    """
    if state.get("error"):
        return {}  # pass-through on error

    system_prompt, user_prompt = build_prompts(
        intent=state["intent"],
        key_facts=state["key_facts"],
        tone=state["tone"],
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


# NODE 3: Email Generator

def _get_client(model: str) -> tuple[AsyncOpenAI, str]:
    """Returns (AsyncOpenAI client, model_name_string) based on selection."""
    if model == "ollama":
        client = AsyncOpenAI(
            api_key=OLLAMA_API_KEY,
            base_url=OLLAMA_BASE_URL,
        )
        return client, OLLAMA_MODEL
    else:
        client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        return client, OPENAI_MODEL


async def generate_email(state: EmailState) -> dict:
    """
    Calls the chosen LLM with the built prompts.
    Extracts CoT reasoning and the clean email from tagged output.
    """
    if state.get("error"):
        return {}

    client, model_name = _get_client(state.get("model", "openai"))

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": state["system_prompt"]},
                {"role": "user",   "content": state["user_prompt"]},
            ],
            temperature=0.7,
            max_completion_tokens=1500,
        )

        raw_output = response.choices[0].message.content or ""
        cot_reasoning, email_output = extract_cot_and_email(raw_output)

        if not email_output:
            return {"error": "Model did not return a parseable email. Raw output stored.", "email_output": raw_output}

        return {
            "email_output": email_output,
            "cot_reasoning": cot_reasoning,
            "model_used": model_name,
        }

    except Exception as e:
        return {"error": f"LLM call failed: {str(e)}"}


# STREAMING VERSION of generate_email (for WebSocket use)

async def generate_email_stream(state: EmailState):
    """
    Generator that yields raw token strings as they arrive from the LLM.
    Caller is responsible for accumulating and then calling extract_cot_and_email.
    Used by the WebSocket endpoint in main.py.
    """
    client, model_name = _get_client(state.get("model", "openai"))

    stream = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": state["system_prompt"]},
            {"role": "user",   "content": state["user_prompt"]},
        ],
        temperature=0.7,
        max_completion_tokens=1500,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta, model_name