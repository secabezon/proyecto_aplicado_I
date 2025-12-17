from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from Retrieval import retrieve

EVAL_PATH = Path("data/processed/eval_set.json")

DEFAULT_EVAL_SET: List[Dict[str, Any]] = [
    {"query": "m/z 449.107 actividad antidiabética", "relevant_doc_ids": ["myricetin_paper_1"]}
]


def load_eval_set() -> List[Dict[str, Any]]:
    """
    Load an evaluation set from disk, or return a default set.

    Returns:
        A list of evaluation items with keys: query, relevant_doc_ids.
    """
    if EVAL_PATH.is_file():
        return json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    return DEFAULT_EVAL_SET


def eval_metrics_at_k(client: Any, k: int = 5) -> Dict[str, Any]:
    """
    Compute macro Precision@k and Recall@k over an evaluation set.

    Args:
        client: Qdrant client.
        k: Retrieval depth.

    Returns:
        Dictionary with precision_at_k, recall_at_k, and per_query breakdown.
    """
    eval_set = load_eval_set()
    per_query_results: List[Dict[str, Any]] = []

    for item in eval_set:
        query = item["query"]
        relevant_ids = set(item.get("relevant_doc_ids", []))

        docs = retrieve(client, query, k=k)
        retrieved_ids = [d.metadata.get("doc_id") for d in docs if d.metadata.get("doc_id")]

        hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
        precision = hits / max(len(retrieved_ids), 1)
        recall = hits / max(len(relevant_ids), 1)

        per_query_results.append(
            {
                "query": query,
                "retrieved_doc_ids": retrieved_ids,
                "relevant_doc_ids": list(relevant_ids),
                "num_relevant": len(relevant_ids),
                "num_retrieved": len(retrieved_ids),
                "num_hits": hits,
                "precision": precision,
                "recall": recall,
            }
        )

    macro_precision = mean(r["precision"] for r in per_query_results)
    macro_recall = mean(r["recall"] for r in per_query_results)

    return {
        "precision_at_k": macro_precision,
        "recall_at_k": macro_recall,
        "per_query": per_query_results,
    }
