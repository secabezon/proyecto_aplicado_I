from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple

from Retrieval import extract_query_filters, retrieve
from ReRank import rerank
from ReWrite import decompose_query, generate_hyde_document
from project_config import cfg


EVAL_PATH = Path("data/processed/set_eval.json")

DEFAULT_EVAL_SET = [
    {
        "query": "What are the main health concerns associated with synthetic food colorants like tartrazine?",
        "relevant_doc_ids": {"order": 39, "doc_id": "Natural_Food_Colorants_and_Preservatives"},
    }
]


def position_weighted_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Compute position-weighted precision: 1/rank of the first relevant hit, else 0.

    Args:
        retrieved_ids: Ranked retrieved ids (doc_id:order).
        relevant_ids: Set of relevant ids.

    Returns:
        Position-weighted precision score.
    """
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / float(i)
    return 0.0


def load_eval_set() -> List[Dict[str, Any]]:
    """
    Load an evaluation set from disk, or return a default set.

    Returns:
        List of evaluation items with keys: query, relevant_doc_ids.
    """
    if EVAL_PATH.is_file():
        return json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    return DEFAULT_EVAL_SET


def _normalize_relevant_ids(rel: Any) -> Set[str]:
    """
    Normalize relevant ids into a set of strings 'doc_id:order'.

    Supported formats:
      - {"doc_id": "...", "order": 39}
      - ["doc:39", "doc:40", ...]
      - "doc:39"

    Args:
        rel: Raw relevant ids structure.

    Returns:
        Set of normalized relevant ids.
    """
    if isinstance(rel, dict) and "doc_id" in rel and "order" in rel:
        return {f"{rel['doc_id']}:{rel['order']}"}

    if isinstance(rel, str):
        return {rel.strip()} if rel.strip() else set()

    if isinstance(rel, list):
        out: Set[str] = set()
        for x in rel:
            if isinstance(x, str) and x.strip():
                out.add(x.strip())
            elif isinstance(x, dict) and "doc_id" in x and "order" in x:
                out.add(f"{x['doc_id']}:{x['order']}")
        return out

    return set()


def _docs_to_ids(docs: List[Any]) -> List[str]:
    """
    Convert LangChain Documents to 'doc_id:order' ids.

    Args:
        docs: Retrieved documents (LangChain Document objects).

    Returns:
        List of ids in retrieval order.
    """
    ids: List[str] = []
    for d in docs:
        doc_id = d.metadata.get("doc_id")
        order = d.metadata.get("order")
        ids.append(f"{doc_id}:{order}")
    return ids


def _run_naive(client: Any, query: str, retrieve_k: int, k_eval: int) -> List[str]:
    """
    Baseline retrieval: raw query -> hybrid retrieve -> take top-k_eval.

    Args:
        client: Qdrant client.
        query: User query.
        retrieve_k: Candidates retrieved from vector store.
        k_eval: Depth at which we evaluate metrics.

    Returns:
        Retrieved ids at evaluation depth.
    """
    docs = retrieve(client, query, k=retrieve_k)
    ids = _docs_to_ids(docs)
    return ids[:k_eval]


def _run_processed(client: Any, query: str, retrieve_k: int, k_eval: int) -> List[str]:
    """
    Final pipeline retrieval approximation (matching Streamlit logic):
    1) Extract filters (doc_id, paper_date)
    2) Decompose query into subqueries
    3) For each subquery: generate HyDE text -> retrieve candidates -> rerank w/ original query
    4) Aggregate reranked results across subqueries (max score per chunk), then take top-k_eval

    Note: We intentionally do NOT include RePack here, because the benchmark is measuring retrieval quality.

    Args:
        client: Qdrant client.
        query: User query.
        retrieve_k: Candidates retrieved per subquery.
        k_eval: Depth at which we evaluate metrics.

    Returns:
        Retrieved ids at evaluation depth.
    """
    try:
        doc_id, paper_date = extract_query_filters(query)
    except Exception:
        doc_id, paper_date = None, None

    try:
        subqueries = decompose_query(query)
    except Exception:
        subqueries = [query]

    best_score_by_id: Dict[str, float] = {}

    for subq in subqueries:
        try:
            hyde_text = generate_hyde_document(subq)
        except Exception:
            hyde_text = subq

        docs = retrieve(
            client,
            hyde_text,
            k=retrieve_k,
            doc_id=doc_id,
            paper_date=paper_date,
        )

        reranked = rerank(query, docs)  # Uses cfg.RERANK_TOP_K internally

        for item in reranked:
            rid = f"{item.get('doc_id')}:{item.get('order')}"
            score = float(item.get("score", 0.0))
            prev = best_score_by_id.get(rid)
            if prev is None or score > prev:
                best_score_by_id[rid] = score

    ranked_ids = sorted(best_score_by_id.items(), key=lambda x: x[1], reverse=True)
    return [rid for rid, _ in ranked_ids[:k_eval]]


def eval_metrics_at_k(client: Any, k_eval: int = 5) -> Dict[str, Any]:
    """
    Compute macro Precision@k_eval and Recall@k_eval over an evaluation set,
    comparing baseline vs processed (final) retrieval.

    Args:
        client: Qdrant client.
        k_eval: Evaluation depth (e.g., 5 means Precision@5/Recall@5).

    Returns:
        Metrics dictionary including per-query details.
    """
    eval_set = load_eval_set()

    retrieve_k = int(getattr(cfg, "RETRIEVE_TOP_K", 50))
    # Note: rerank depth is controlled by cfg.RERANK_TOP_K in ReRank.rerank()

    per_query_naive: List[Dict[str, Any]] = []
    per_query_processed: List[Dict[str, Any]] = []

    for item in eval_set:
        query = item["query"]
        relevant_ids = _normalize_relevant_ids(item.get("relevant_doc_ids"))

        retrieved_naive = _run_naive(client, query, retrieve_k=retrieve_k, k_eval=k_eval)
        retrieved_processed = _run_processed(client, query, retrieve_k=retrieve_k, k_eval=k_eval)

        hits_naive = len(set(retrieved_naive) & relevant_ids)
        hits_processed = len(set(retrieved_processed) & relevant_ids)

        precision_naive = hits_naive / float(len(retrieved_naive) or 1)
        recall_naive = hits_naive / float(len(relevant_ids) or 1)
        pwp_naive = position_weighted_precision(retrieved_naive, relevant_ids)

        precision_processed = hits_processed / float(len(retrieved_processed) or 1)
        recall_processed = hits_processed / float(len(relevant_ids) or 1)
        pwp_processed = position_weighted_precision(retrieved_processed, relevant_ids)

        per_query_naive.append(
            {
                "query": query,
                "relevant_doc_ids": sorted(list(relevant_ids)),
                "retrieved_doc_ids": retrieved_naive,
                "k_eval": k_eval,
                "retrieve_k": retrieve_k,
                "num_relevant": len(relevant_ids),
                "num_retrieved": len(retrieved_naive),
                "num_hits": hits_naive,
                "precision": precision_naive,
                "recall": recall_naive,
                "position_weighted_precision_at_k": pwp_naive,
            }
        )

        per_query_processed.append(
            {
                "query": query,
                "relevant_doc_ids": sorted(list(relevant_ids)),
                "retrieved_doc_ids": retrieved_processed,
                "k_eval": k_eval,
                "retrieve_k": retrieve_k,
                "num_relevant": len(relevant_ids),
                "num_retrieved": len(retrieved_processed),
                "num_hits": hits_processed,
                "precision": precision_processed,
                "recall": recall_processed,
                "position_weighted_precision_at_k": pwp_processed,
            }
        )

    macro_precision_naive = mean(r["precision"] for r in per_query_naive) if per_query_naive else 0.0
    macro_recall_naive = mean(r["recall"] for r in per_query_naive) if per_query_naive else 0.0
    macro_pwp_naive = mean(r["position_weighted_precision_at_k"] for r in per_query_naive) if per_query_naive else 0.0

    macro_precision_processed = mean(r["precision"] for r in per_query_processed) if per_query_processed else 0.0
    macro_recall_processed = mean(r["recall"] for r in per_query_processed) if per_query_processed else 0.0
    macro_pwp_processed = mean(r["position_weighted_precision_at_k"] for r in per_query_processed) if per_query_processed else 0.0

    return {
        "k_eval": k_eval,
        "retrieve_k": retrieve_k,
        "precision_at_k_naive": macro_precision_naive,
        "pwp_naive": macro_pwp_naive,
        "recall_at_k_naive": macro_recall_naive,
        "per_query_naive": per_query_naive,
        "precision_at_k_processed": macro_precision_processed,
        "pwp_processed": macro_pwp_processed,
        "recall_at_k_processed": macro_recall_processed,
        "per_query_processed": per_query_processed,
    }
