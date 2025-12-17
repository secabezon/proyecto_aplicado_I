from __future__ import annotations

from typing import List

from LLM_prompt import llm_prompt
from project_config import cfg


def decompose_query(query: str) -> List[str]:
    """
    Decompose a user query into multiple sub-queries.

    Args:
        query: User query.

    Returns:
        A non-empty list of sub-queries.
    """
    prompt = getattr(cfg, "PROMPT_QUERY_DECOMPOSITION").format(query=query)
    messages = [{"role": "user", "content": prompt}]
    temperature = float(getattr(cfg, "LLM_TEMPERATURE", 0.0))
    out = llm_prompt(messages, temperature=temperature)

    raw_lines = [line.strip() for line in (out or "").splitlines() if line.strip()]

    max_subqueries = int(getattr(cfg, "MAX_SUBQUERIES", 3))
    min_subqueries = int(getattr(cfg, "MIN_SUBQUERIES", 1))

    cleaned: List[str] = []
    seen = set()
    for line in raw_lines:
        if len(line) > 200:
            continue
        if line in seen:
            continue
        seen.add(line)
        cleaned.append(line)

    if not cleaned:
        return [query]

    if len(cleaned) < min_subqueries:
        return [query]

    return cleaned[:max_subqueries]


def generate_hyde_document(query: str) -> str:
    """
    Generate a HyDE-style synthetic document for retrieval.

    Args:
        query: Sub-query.

    Returns:
        HyDE text.
    """
    prompt = getattr(cfg, "PROMPT_HYDE").format(query=query)
    messages = [{"role": "user", "content": prompt}]
    temperature = float(getattr(cfg, "LLM_TEMPERATURE", 0.0))
    return llm_prompt(messages, temperature=temperature).strip()
