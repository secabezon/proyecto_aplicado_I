# benchmark.py
from pathlib import Path
import json
from statistics import mean
from Retrieval import retrieve

EVAL_PATH = Path("data/processed/eval_set.json")

DEFAULT_EVAL_SET = [
    {
        "query": "m/z 449.107 actividad antidiabética",
        "relevant_doc_ids": ["myricetin_paper_1"]
    }
]


def load_eval_set(path: Path = EVAL_PATH):
    if not path.exists():
        print(f"[WARN] {path} no encontrado. Usando DEFAULT_EVAL_SET.")
        return DEFAULT_EVAL_SET

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def eval_metrics_at_k(client, k: int = 5, eval_set=None):
    if eval_set is None:
        eval_set = load_eval_set()

    per_query_results = []

    for item in eval_set:
        query = item["query"]
        relevant_ids = set(item["relevant_doc_ids"])

        docs = retrieve(client, query, k=k)
        retrieved_ids = [d.metadata["doc_id"] for d in docs]

        retrieved_set = set(retrieved_ids)
        hits = len(relevant_ids & retrieved_set)

        precision = hits / len(retrieved_ids) if retrieved_ids else 0.0
        recall = hits / len(relevant_ids) if relevant_ids else 0.0

        per_query_results.append({
            "query": query,
            "relevant_doc_ids": list(relevant_ids),
            "retrieved_doc_ids": retrieved_ids,
            "num_relevant": len(relevant_ids),
            "num_retrieved": len(retrieved_ids),
            "num_hits": hits,
            "precision": precision,
            "recall": recall,
        })

    macro_precision = mean(r["precision"] for r in per_query_results)
    macro_recall = mean(r["recall"] for r in per_query_results)

    return {
        "precision_at_k": macro_precision,
        "recall_at_k": macro_recall,
        "per_query": per_query_results,
    }
