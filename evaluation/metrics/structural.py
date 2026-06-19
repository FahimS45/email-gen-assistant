"""
structural.py

Custom Metric 3: Structural Completeness Score
This module implements a structural completeness metric that evaluates whether the generated email follows standard email 
conventions, such as including a subject line, greeting, body, and sign-off. It uses regex pattern matching to identify these 
components and checks for common failure modes like missing sections or leaked CoT tags. The final score is a composite of these factors, 
providing a holistic assessment of the email's structural quality.
"""

import re
from typing import TypedDict


# TypedDict result type 

class StructuralResult(TypedDict):
    score: float
    has_subject_line: bool
    has_greeting: bool
    has_body: bool
    has_sign_off: bool
    no_leaked_artifacts: bool
    notes: list[str]


# Regex pattern banks

GREETING_PATTERNS = [
    r"^\s*dear\b",
    r"^\s*hi\b",
    r"^\s*hello\b",
    r"^\s*hey\b",
    r"^\s*good (morning|afternoon|evening)\b",
    r"^\s*to whom it may concern\b",
    r"^\s*team\s*,?\s*$",
    r"^\s*all\s*,?\s*$",
    r"^\s*everyone\s*,?\s*$",
]

SIGNOFF_PATTERNS = [
    r"\bsincerely\b",
    r"\bbest regards\b",
    r"\bbest\b\s*,?\s*$",
    r"\bregards\b",
    r"\bthank you\b\s*,?\s*$",
    r"\bthanks\b\s*,?\s*$",
    r"\bwarm(ly)?\b",
    r"\bwith apologies\b",
    r"\bcheers\b",
    r"\blooking forward\b",
    r"\byours (truly|sincerely|faithfully)\b",
]

LEAKED_ARTIFACT_PATTERNS = [
    r"<\s*thinking\s*>",
    r"<\s*/\s*thinking\s*>",
    r"<\s*reasoning\s*>",
    r"<\s*/\s*reasoning\s*>",
    r"<\s*email\s*>",
    r"<\s*/\s*email\s*>",
    r"\bas an ai (language model|assistant)\b",
    r"\bi (cannot|can't) generate\b",
    r"\{\{.*?\}\}",         # unfilled template braces like {{intent}}
    r"\[insert .*?\]",       # leftover placeholder instructions
]


# Component checkers

def _has_subject_line(lines: list[str]) -> bool:
    for line in lines[:3]:
        if re.match(r"^\s*subject\s*:", line, re.IGNORECASE):
            return True
    return False


def _has_greeting(lines: list[str]) -> bool:
    # Look in the first few non-empty lines (skip Subject: if present)
    candidates = [l for l in lines if l.strip()][:5]
    for line in candidates:
        for pattern in GREETING_PATTERNS:
            if re.search(pattern, line.strip(), re.IGNORECASE):
                return True
    return False


def _has_sign_off(lines: list[str]) -> bool:
    # Look in the last few non-empty lines
    candidates = [l for l in lines if l.strip()][-5:]
    tail_text = " ".join(candidates)
    for pattern in SIGNOFF_PATTERNS:
        if re.search(pattern, tail_text, re.IGNORECASE):
            return True
    return False


def _has_body(email: str) -> bool:
    # A real email body requires meaningful content beyond greeting + signoff.
    # 25 words is a deliberately low bar — we only want to catch near-empty outputs.
    word_count = len(email.split())
    return word_count >= 25


def _no_leaked_artifacts(email: str) -> bool:
    for pattern in LEAKED_ARTIFACT_PATTERNS:
        if re.search(pattern, email, re.IGNORECASE):
            return False
    return True


# Public scorer

def score_structural(email_output: str) -> StructuralResult:
    """
    Computes the Structural Completeness Score for a single generated email.

    Args:
        email_output: the clean email text (CoT already stripped upstream
                       by extract_cot_and_email in prompt_engineering.py).

    Returns:
        StructuralResult TypedDict with the overall score (0.0-1.0) and the
        breakdown of each binary component, plus human-readable notes
        explaining any deductions (useful for debugging low scores).
    """
    email_output = email_output or ""
    lines = email_output.splitlines()
    notes: list[str] = []

    has_subject  = _has_subject_line(lines)
    has_greeting = _has_greeting(lines)
    has_body     = _has_body(email_output)
    has_signoff  = _has_sign_off(lines)
    clean        = _no_leaked_artifacts(email_output)

    if not has_subject:
        notes.append("Missing a 'Subject:' line.")
    if not has_greeting:
        notes.append("Missing a recognizable greeting/salutation.")
    if not has_body:
        notes.append("Body text too short (< 25 words).")
    if not has_signoff:
        notes.append("Missing a recognizable sign-off/closing line.")
    if not clean:
        notes.append("Detected leaked CoT tags, AI self-reference, or unfilled placeholders.")

    components = [has_subject, has_greeting, has_body, has_signoff, clean]
    score = sum(1 for c in components if c) / len(components)

    return StructuralResult(
        score=round(score, 3),
        has_subject_line=has_subject,
        has_greeting=has_greeting,
        has_body=has_body,
        has_sign_off=has_signoff,
        no_leaked_artifacts=clean,
        notes=notes,
    )