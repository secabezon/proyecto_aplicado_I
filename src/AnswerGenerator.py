from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from LLM_prompt import llm_prompt
from project_config import cfg


@dataclass(frozen=True)
class EvidenceItem:
    """
    A grounded evidence item used to generate the final answer.

    Attributes:
        doc_id: Document identifier (PDF stem).
        chunk_id: Chunk identifier.
        paper_date: Paper date or year.
        paper_title: Paper title.
        text: Literal extracted text used as evidence.
    """
    doc_id: Optional[str]
    chunk_id: Optional[str]
    paper_date: Optional[str]
    paper_title: Optional[str]
    text: str


def normalize_meta(value: Optional[str], default: str = "UNKNOWN") -> str:
    """
    Normalize optional metadata into a safe string.

    Args:
        value: Incoming value.
        default: Default when the value is empty.

    Returns:
        Normalized string.
    """
    if value is None:
        return default
    cleaned = str(value).strip()
    return cleaned if cleaned else default


def dedupe_evidence(items: List[EvidenceItem]) -> List[EvidenceItem]:
    """
    Deduplicate evidence items to reduce repeated context.

    Args:
        items: Raw evidence items.

    Returns:
        Deduplicated evidence items.
    """
    seen: Set[Tuple[str, str, str]] = set()
    out: List[EvidenceItem] = []

    for e in items:
        text = (e.text or "").strip()
        if not text:
            continue

        key = (
            normalize_meta(e.doc_id, default="N/A"),
            normalize_meta(e.chunk_id, default="N/A"),
            text,
        )
        if key in seen:
            continue

        seen.add(key)
        out.append(e)

    return out


def build_evidence_block(items: List[EvidenceItem], max_items: int) -> str:
    """
    Build an evidence block with tags [E1], [E2], ... for grounded generation.

    Args:
        items: Evidence items.
        max_items: Max number of items to include.

    Returns:
        Evidence block text.
    """
    blocks: List[str] = []
    for i, it in enumerate(items[:max_items], start=1):
        paper_date = normalize_meta(it.paper_date)
        paper_title = normalize_meta(it.paper_title)
        doc_id = normalize_meta(it.doc_id, default="N/A")
        chunk_id = normalize_meta(it.chunk_id, default="N/A")
        header = f"[E{i}] paper_date={paper_date} | paper_title={paper_title} | doc_id={doc_id} | chunk_id={chunk_id}"
        body = (it.text or "").strip()
        blocks.append(f"{header}\n{body}")

    return "\n\n".join(blocks).strip()


def generate_final_answer(query: str, evidence: List[EvidenceItem]) -> str:
    """
    Generate a grounded final answer using only the provided evidence.

    Args:
        query: User query.
        evidence: Evidence items.

    Returns:
        Final answer text.
    """
    deduped = dedupe_evidence(evidence)
    if not deduped:
        return "No hay información suficiente en la evidencia."

    max_items = int(getattr(cfg, "MAX_EVIDENCE_ITEMS", 10))
    evidence_block = build_evidence_block(deduped, max_items=max_items)

    prompt = getattr(cfg, "PROMPT_FINAL_ANSWER").format(query=query, evidence=evidence_block)
    messages = [{"role": "user", "content": prompt}]
    temperature = float(getattr(cfg, "LLM_TEMPERATURE", 0.0))
    return llm_prompt(messages, temperature=temperature).strip()
