"""
fact_recall.py

Custom Metric 1: Fact Recall Score
This module implements a fact-recall metric that evaluates how well the generated email includes the key facts specified by the user. 
It uses semantic embeddings to compute cosine similarity between the key facts and the generated email, allowing for paraphrasing and variations in phrasing. 
The module also provides a fallback keyword-overlap method for offline testing without API calls.
"""

import re
import math

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from backend.config import EMBEDDING_MODEL


# A fact is considered "present" if any email chunk has cosine similarity
# >= this threshold with the fact embedding. 0.65 is intentionally strict
# enough to require clear semantic overlap while allowing paraphrasing.
SIMILARITY_THRESHOLD = 0.65


# Pydantic models

class PerFactVerdict(BaseModel):
    """Result for a single key fact."""
    fact: str
    similarity_score: float = Field(ge=0.0, le=1.0, description="Max cosine similarity across email chunks.")
    found: bool = Field(description=f"True if similarity_score >= {SIMILARITY_THRESHOLD}.")
    justification: str = ""

    @field_validator("similarity_score", mode="before")
    @classmethod
    def round_similarity(cls, v):
        return round(float(v), 4)


class FactRecallResult(BaseModel):
    """Full result for Metric 1: Fact Recall."""
    score: float = Field(ge=0.0, le=1.0, description="Mean per-fact similarity score across all facts.")
    total_facts: int = Field(ge=0)
    facts_found: int = Field(ge=0, description=f"Facts with similarity >= {SIMILARITY_THRESHOLD}.")
    per_fact: list[PerFactVerdict]

    @field_validator("score", mode="before")
    @classmethod
    def round_score(cls, v):
        return round(float(v), 3)


# Embedding utilities

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _chunk_email(email: str, max_chars: int = 400) -> list[str]:
    """
    Splits an email into overlapping sentence-level chunks for embedding.
    We use sentence boundaries rather than fixed-length windows so that
    each chunk is a coherent unit of meaning.

    Chunking matters because a long email may mention different facts in
    different paragraphs — embedding the whole email as one vector averages
    everything and dilutes specific facts. Per-sentence max-pooling gives
    each fact the best possible chance to surface.
    """
    # Split on sentence-ending punctuation, keeping the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', email.strip())
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    # Always include the full email as one chunk too (catches cross-sentence facts)
    if len(email) <= 2000:
        chunks.append(email)
    return chunks or [email]


async def _get_embeddings(
    texts: list[str],
    client: AsyncOpenAI,
) -> list[list[float]]:
    """
    Fetches embeddings for a batch of texts in a single API call.
    text-embedding-3-small supports up to 2048 texts per request.
    """
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # Results are returned in the same order as input
    return [item.embedding for item in response.data]


# Primary method: Embedding cosine similarity

async def score_fact_recall_embedding(
    email_output: str,
    key_facts: list[str],
    embed_client: AsyncOpenAI,
    threshold: float = SIMILARITY_THRESHOLD,
) -> FactRecallResult:
    """
    Scores Fact Recall using semantic embedding similarity.

    Strategy:
      - Embed each key_fact as a single vector.
      - Embed each sentence-level chunk of the email.
      - For each fact, find the max cosine similarity across all chunks.
      - Score = mean of per-fact max-similarities (continuous, not binary).
      - facts_found = count of facts where max_similarity >= threshold.

    A single batched embedding call handles all texts at once, keeping
    API latency and cost minimal (text-embedding-3-small: ~$0.00002/1K tokens).

    Args:
        email_output: clean generated email (CoT already stripped).
        key_facts: list of facts the model was instructed to include.
        embed_client: AsyncOpenAI client (must be OpenAI — Ollama doesn't
                      serve an embeddings endpoint).
        threshold: cosine similarity above which a fact is considered present.

    Returns:
        FactRecallResult with continuous similarity scores and binary found flags.
    """
    if not key_facts:
        return FactRecallResult(score=1.0, total_facts=0, facts_found=0, per_fact=[])

    email_chunks = _chunk_email(email_output)

    # One batched API call for facts + chunks
    all_texts = key_facts + email_chunks
    all_embeddings = await _get_embeddings(all_texts, embed_client)

    fact_embeddings  = all_embeddings[:len(key_facts)]
    chunk_embeddings = all_embeddings[len(key_facts):]

    per_fact: list[PerFactVerdict] = []
    similarity_scores: list[float] = []

    for fact, fact_emb in zip(key_facts, fact_embeddings):
        # Max-pool over all chunks: a fact present anywhere in the email
        # should score high regardless of where it appears
        similarities = [_cosine_similarity(fact_emb, chunk_emb) for chunk_emb in chunk_embeddings]
        max_sim = max(similarities)
        found = max_sim >= threshold
        similarity_scores.append(max_sim)

        per_fact.append(PerFactVerdict(
            fact=fact,
            similarity_score=max_sim,
            found=found,
            justification=(
                f"Max cosine similarity: {max_sim:.4f} "
                f"({'≥' if found else '<'} threshold {threshold})."
            ),
        ))

    # Overall score = mean similarity (graded, not just binary recall fraction)
    # This produces meaningful variance even when all facts technically "pass"
    overall_score = sum(similarity_scores) / len(similarity_scores)
    facts_found = sum(1 for v in per_fact if v.found)

    return FactRecallResult(
        score=round(overall_score, 3),
        total_facts=len(key_facts),
        facts_found=facts_found,
        per_fact=per_fact,
    )


# Fallback method: Keyword overlap (free, no API calls)

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "this", "that", "at", "as",
    "it", "by", "from", "has", "have", "had", "will", "would", "should",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def score_fact_recall_keyword(
    email_output: str,
    key_facts: list[str],
    overlap_threshold: float = 0.5,
) -> FactRecallResult:
    """
    Free keyword-overlap fallback for offline smoke testing.
    Uses non-stopword token intersection as a proxy for semantic similarity.

    The overlap ratio is used as the "similarity_score" so the output shape
    is consistent with score_fact_recall_embedding.
    """
    if not key_facts:
        return FactRecallResult(score=1.0, total_facts=0, facts_found=0, per_fact=[])

    email_keywords = _keywords(email_output)
    per_fact: list[PerFactVerdict] = []
    similarity_scores: list[float] = []

    for fact in key_facts:
        fact_keywords = _keywords(fact)
        if not fact_keywords:
            overlap = 1.0
        else:
            overlap = len(fact_keywords & email_keywords) / len(fact_keywords)

        found = overlap >= overlap_threshold
        similarity_scores.append(overlap)
        per_fact.append(PerFactVerdict(
            fact=fact,
            similarity_score=overlap,
            found=found,
            justification=f"Keyword overlap ratio: {overlap:.3f} (threshold {overlap_threshold}). No API call.",
        ))

    overall_score = sum(similarity_scores) / len(similarity_scores)

    return FactRecallResult(
        score=round(overall_score, 3),
        total_facts=len(key_facts),
        facts_found=sum(1 for v in per_fact if v.found),
        per_fact=per_fact,
    )