
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
    pairs = [[query,  doc.page_content] for doc in chunks]
    scores = cross_encoder.predict(pairs)

    results_with_scores = [
        {
            "doc_id": doc.metadata["doc_id"],
            "chunk_id": doc.metadata["chunk_id"],
            "content": doc.page_content,
            "score": float(score)
        }
        for doc, score in zip(chunks, scores)
    ]

    best_result = sorted(results_with_scores, key=lambda x: x["score"],  reverse=True)[:3]

    return best_result
