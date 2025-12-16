# benchmark.py
from pathlib import Path
import json
from statistics import mean
from Retrieval import retrieve
from ReRank import reRank
from ReWrite import response_hyde, stepback_query, descomposition_query
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

    per_query_results_naive = []
    per_query_results_processed = []

    for item in eval_set:
        query = item["query"]

        # Normaliza relevant ids
        rel = item["relevant_doc_ids"]
        if isinstance(rel, dict):  
            relevant_ids = { f"{rel['doc_id']}:{rel['order']}" }
        else:
            relevant_ids = set(rel)



    ################### NAIVE ###################

        docs_naive = retrieve(client, query, k=k)
        retrieved_ids_naive = [
            f"{d.metadata['doc_id']}:{d.metadata.get('order')}"
            for d in docs_naive
        ]

        retrieved_set_naive = set(retrieved_ids_naive)

        hits_naive = len(relevant_ids & retrieved_set_naive)

        precision_naive = hits_naive / len(retrieved_ids_naive) if retrieved_ids_naive else 0.0
        recall_naive = hits_naive / len(relevant_ids) if relevant_ids else 0.0
        pwp_naive  = position_weighted_precision(retrieved_ids_naive, relevant_ids)
        per_query_results_naive.append({
            "query": query,
            "relevant_doc_ids": list(relevant_ids),
            "retrieved_doc_ids": retrieved_ids_naive,
            "num_relevant": len(relevant_ids),
            "num_retrieved": len(retrieved_ids_naive),
            "num_hits": hits_naive,
            "precision": precision_naive,
            "recall": recall_naive,
            "position_weighted_precision_at_k": pwp_naive,
        })

    ################### PROCESSED ###################

    

        response = retrieve(client, query,k=20)
        rerank=reRank(query,response)
        print(rerank)
        retrieved_ids_processed = [
                                    f"{d['doc_id']}:{d['order']}"
                                    for d in rerank
                                   ]

        retrieved_set_processed = set(retrieved_ids_processed)

        hits_processed = len(relevant_ids & retrieved_set_processed)

        precision_processed  = hits_processed / len(retrieved_ids_processed) if retrieved_ids_processed else 0.0
        recall_processed  = hits_processed / len(relevant_ids) if relevant_ids else 0.0
        pwp_processed  = position_weighted_precision(retrieved_ids_processed, relevant_ids)

        per_query_results_processed.append({
            "query": query,
            "relevant_doc_ids": list(relevant_ids),
            "retrieved_doc_ids": retrieved_ids_processed,
            "num_relevant": len(relevant_ids),
            "num_retrieved": len(retrieved_ids_processed),
            "num_hits": hits_processed,
            "precision": precision_processed,
            "recall": recall_processed,
            "position_weighted_precision_at_k": pwp_processed,
        })


    
    macro_precision_naive = mean(r["position_weighted_precision_at_k"] for r in per_query_results_naive)
    macro_recall_naive = mean(r["recall"] for r in per_query_results_naive)

    macro_precision_processed = mean(r["position_weighted_precision_at_k"] for r in per_query_results_processed)
    macro_recall_processed = mean(r["recall"] for r in per_query_results_processed)

    return {
        "precision_at_k_naive": macro_precision_naive,
        "recall_at_k_naive": macro_recall_naive,
        "per_query_naive": per_query_results_naive,
        "precision_at_k_processed": macro_precision_processed,
        "recall_at_k_processed": macro_recall_processed,
        "per_query_processed": per_query_results_processed,
    }
