from __future__ import annotations

from LLM_prompt import llm_prompt
from project_config import cfg


def repack(query: str, document_chunk: str) -> str:
    """
    Extract literal relevant spans from a chunk.

    Args:
        query: User query.
        document_chunk: Chunk content.

    Returns:
        Extracted spans or NO_RELEVANT_CONTENT.
    """
    prompt = getattr(cfg, "PROMPT_REPACK").format(query=query, doc=document_chunk)
    messages = [{"role": "user", "content": prompt}]
    temperature = float(getattr(cfg, "LLM_TEMPERATURE", 0.0))
    return llm_prompt(messages, temperature=temperature).strip()
