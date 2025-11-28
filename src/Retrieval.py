from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models
from langchain_community.embeddings import HuggingFaceEmbeddings

from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from config.config import EMBEDDING_MODEL


def retrieve(
    client,
    query: str,
    k: int = 5,
    doc_id: str | int | None = None,
):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    qdrant = QdrantVectorStore(
        client=client,
        collection_name="bioactives_collection",
        embedding=embedding_model,
    )

    qdrant_filter = None

    if doc_id is not None:
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.doc_id",
                    match=models.MatchValue(
                        value=doc_id  # <-- ahora es str/int, NO dict
                    )
                )
            ]
        )

    results = qdrant.search(
        query=query,
        k=k,
        search_type="mmr",
        filter=qdrant_filter,
    )

    return results