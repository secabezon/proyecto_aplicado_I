from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from project_config import cfg


_CROSS_ENCODER: Optional[CrossEncoder] = None
_CROSS_ENCODER_NAME: Optional[str] = None


def _get_cross_encoder() -> CrossEncoder:
    """
    Get a cached cross-encoder instance.

    Returns:
        CrossEncoder model.
    """
    global _CROSS_ENCODER, _CROSS_ENCODER_NAME

    model_name = getattr(cfg, "CROSSENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    if _CROSS_ENCODER is not None and _CROSS_ENCODER_NAME == model_name:
        return _CROSS_ENCODER

    _CROSS_ENCODER = CrossEncoder(model_name)
    _CROSS_ENCODER_NAME = model_name
    return _CROSS_ENCODER


def rerank(query: str, chunks: List[Document]) -> List[Dict[str, object]]:
    """
    Rerank retrieved chunks using a cross-encoder.

    Args:
        query: User query.
        chunks: Retrieved documents.

    Returns:
        A list of dictionaries with content, score, and metadata.
    """
    top_k = int(getattr(cfg, "RERANK_TOP_K", 5))
    date_key = getattr(cfg, "PAPER_DATE_KEY", "paper_date")
    title_key = getattr(cfg, "PAPER_TITLE_KEY", "paper_title")

    cross_encoder = _get_cross_encoder()

    pairs = [[query, doc.page_content] for doc in chunks]
    scores = cross_encoder.predict(pairs)

    results: List[Dict[str, object]] = []
    for doc, score in zip(chunks, scores):
        results.append(
            {
                "doc_id": doc.metadata.get("doc_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "order": doc.metadata.get("order"),
                "content": doc.page_content,
                "score": float(score),
                "paper_date": doc.metadata.get(date_key, "UNKNOWN"),
                "paper_title": doc.metadata.get(title_key, "UNKNOWN"),
            }
        )
    results.sort(key=lambda x: float(x["score"]), reverse=True)
    return results[:top_k]
