
from pathlib import Path
from typing import List
from sentence_transformers import CrossEncoder
import sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from config.config import CROSSENCODER_MODEL



def reRank(query: str,chunks: List
):
    cross_encoder = CrossEncoder(CROSSENCODER_MODEL)
    pairs = [[query, doc] for doc in chunks]
    scores = cross_encoder.predict(pairs)

    results_with_scores = [
        (doc_id, score)
        for doc_id, score in zip(chunks, scores)
    ]

    best_result = sorted(results_with_scores, key=lambda x: x[2], reverse=True)[:1]

    return best_result
