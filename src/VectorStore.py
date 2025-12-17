from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from project_config import cfg, project_root


ProgressCallback = Callable[[int, int], None]


def read_chunk_records(chunks_file: Path) -> List[dict]:
    """
    Read chunk records from a JSON file containing a list of objects.

    Args:
        chunks_file: Path to the chunk JSON file.

    Returns:
        Chunk records.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the JSON does not contain a list.
    """
    if not chunks_file.is_file():
        raise FileNotFoundError(f"Missing chunks file: {chunks_file}")

    data = json.loads(chunks_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Chunk file must contain a list of records: {chunks_file}")

    return data


def vector_store_config(client: QdrantClient, progress_callback: Optional[ProgressCallback] = None) -> None:
    """
    Recreate the Qdrant collection and upsert documents for all chunk records.

    Args:
        client: Qdrant client.
        progress_callback: Optional callback invoked as (done, total).
    """
    root = project_root()
    out_dir = root / getattr(cfg, "OUTPUT_DIR", "outputs/ingest")
    base = getattr(cfg, "BASE_OUTPUT_NAME", "corpus_chunks")
    chunks_file = out_dir / f"{base}.json"

    records = read_chunk_records(chunks_file)

    embedding_model_name = getattr(cfg, "EMBEDDING_MODEL", "thenlper/gte-small")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    dim = len(embedding_model.embed_query("dimension check"))

    collection_name = getattr(cfg, "COLLECTION_NAME", "food_science_papers_v1")
    dense_name = getattr(cfg, "DENSE_VECTOR_NAME", "dense")
    sparse_name = getattr(cfg, "SPARSE_VECTOR_NAME", "sparse")
    sparse_model = getattr(cfg, "SPARSE_MODEL_NAME", "Qdrant/bm25")

    date_key = getattr(cfg, "PAPER_DATE_KEY", "paper_date")
    title_key = getattr(cfg, "PAPER_TITLE_KEY", "paper_title")

    sparse_embeddings = FastEmbedSparse(model_name=sparse_model)

    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
    except Exception:
        try:
            client.get_collection(collection_name)
            client.delete_collection(collection_name)
        except Exception:
            pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config={dense_name: VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={sparse_name: SparseVectorParams()},
    )

    docs: List[Document] = []
    for r in records:
        docs.append(
            Document(
                page_content=str(r.get("text", "")),
                metadata={
                    "doc_id": r.get("doc_id"),
                    "chunk_id": r.get("chunk_id"),
                    "order": r.get("order"),
                    date_key: r.get(date_key, "UNKNOWN"),
                    title_key: r.get(title_key, "UNKNOWN"),
                },
            )
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=dense_name,
        sparse_vector_name=sparse_name,
    )

    batch_size = int(getattr(cfg, "UPSERT_BATCH_SIZE", 64))
    total = len(docs)
    done = 0

    logger.info(
        "Upserting documents | collection=%s | total=%d | batch_size=%d",
        collection_name,
        total,
        batch_size,
    )

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        vector_store.add_documents(docs[start:end])
        done = end

        if progress_callback is not None:
            progress_callback(done, total)

        logger.info("Upsert progress | done=%d | total=%d", done, total)
