from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from project_config import cfg


def rerank(query: str, chunks: List[Document]) -> List[Dict[str, object]]:
    """
    Rerank retrieved chunks using a cross-encoder.

    Args:
        query: User query.
        chunks: Retrieved documents.

    Returns:
        A list of dictionaries with content, score, and metadata.
    """
    model_name = getattr(cfg, "CROSSENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k = int(getattr(cfg, "RERANK_TOP_K", 5))

    date_key = getattr(cfg, "PAPER_DATE_KEY", "paper_date")
    title_key = getattr(cfg, "PAPER_TITLE_KEY", "paper_title")

    cross_encoder = CrossEncoder(model_name)
    pairs = [[query, doc.page_content] for doc in chunks]
    scores = cross_encoder.predict(pairs)

    results: List[Dict[str, object]] = []
    for doc, score in zip(chunks, scores):
        results.append(
            {
                "doc_id": doc.metadata.get("doc_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content": doc.page_content,
                "score": float(score),
                "paper_date": doc.metadata.get(date_key, "UNKNOWN"),
                "paper_title": doc.metadata.get(title_key, "UNKNOWN"),
            }
        )

    results.sort(key=lambda x: float(x["score"]), reverse=True)
    return results[:top_k]
