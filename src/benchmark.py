# benchmark.py
from pathlib import Path
import json
from statistics import mean
from Retrieval import retrieve
import math
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

def position_weighted_precision(retrieved_ids, relevant_ids):
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1 / i
    return 0.0

EVAL_PATH = Path("data/processed/set_eval.json")

DEFAULT_EVAL_SET = [
  {
    "query": "What are the main health concerns associated with synthetic food colorants like tartrazine?",
    "relevant_doc_ids": {'order': 39, 'doc_id': 'Natural_Food_Colorants_and_Preservatives'},
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

        # Normaliza relevant ids
        rel = item["relevant_doc_ids"]
        if isinstance(rel, dict):  
            relevant_ids = { f"{rel['doc_id']}:{rel['order']}" }
        else:
            relevant_ids = set(rel)

        # Retrieval
        docs = retrieve(client, query, k=k)

        retrieved_ids = [
            f"{d.metadata['doc_id']}:{d.metadata.get('order')}"
            for d in docs
        ]

        retrieved_set = set(retrieved_ids)
        hits = len(relevant_ids & retrieved_set)

        precision = hits / len(retrieved_ids) if retrieved_ids else 0.0
        recall = hits / len(relevant_ids) if relevant_ids else 0.0

        pwp  = position_weighted_precision(retrieved_ids, relevant_ids)

        per_query_results.append({
            "query": query,
            "relevant_doc_ids": list(relevant_ids),
            "retrieved_doc_ids": retrieved_ids,
            "num_relevant": len(relevant_ids),
            "num_retrieved": len(retrieved_ids),
            "num_hits": hits,
            "precision": precision,
            "recall": recall,
            "position_weighted_precision_at_k": pwp,
        })

    macro_precision = mean(r["precision"] for r in per_query_results)
    macro_recall = mean(r["recall"] for r in per_query_results)
    macro_position_weighted_precision_at_k = mean(r["position_weighted_precision_at_k"] for r in per_query_results)

    return {
        "precision_at_k": macro_precision,
        "recall_at_k": macro_recall,
        "position_weighted_precision_at_k": macro_position_weighted_precision_at_k,
        "per_query": per_query_results,
    }
