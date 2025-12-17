from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models

from LLM_prompt import llm_prompt, safe_json_loads
from project_config import cfg


def extract_query_filters(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract optional retrieval filters from the user query.

    Args:
        query: User query.

    Returns:
        (doc_id, paper_date) where each value can be None.
    """
    prompt = getattr(cfg, "PROMPT_QUERY_FILTERS").format(query=query)
    messages = [{"role": "user", "content": prompt}]

    try:
        out = llm_prompt(messages, temperature=float(getattr(cfg, "LLM_TEMPERATURE", 0.0))).strip()
        data = safe_json_loads(out)
    except Exception:
        return None, None

    doc_id = data.get("doc_id")
    paper_date = data.get("paper_date")

    doc_id_val = doc_id.strip() if isinstance(doc_id, str) and doc_id.strip() else None
    date_val = paper_date.strip() if isinstance(paper_date, str) and paper_date.strip() else None

    return doc_id_val, date_val


def retrieve(
    client: QdrantClient,
    query: str,
    k: Optional[int] = None,
    doc_id: Optional[str] = None,
    paper_date: Optional[str] = None,
) -> List:
    """
    Retrieve chunks from Qdrant using hybrid search with optional filters.

    Args:
        client: Qdrant client.
        query: Query string.
        k: Top-k.
        doc_id: Optional filter by doc_id.
        paper_date: Optional filter by paper_date.

    Returns:
        A list of LangChain Document objects (deduplicated by chunk_id).
    """
    collection_name = getattr(cfg, "COLLECTION_NAME", "food_science_papers_v1")
    dense_name = getattr(cfg, "DENSE_VECTOR_NAME", "dense")
    sparse_name = getattr(cfg, "SPARSE_VECTOR_NAME", "sparse")
    sparse_model = getattr(cfg, "SPARSE_MODEL_NAME", "Qdrant/bm25")

    embedding_model = HuggingFaceEmbeddings(
        model_name=getattr(cfg, "EMBEDDING_MODEL", "thenlper/gte-small")
    )
    sparse_embeddings = FastEmbedSparse(model_name=sparse_model)

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=dense_name,
        sparse_vector_name=sparse_name,
    )

    k_val = int(k or getattr(cfg, "RETRIEVE_TOP_K", 10))
    date_key = getattr(cfg, "PAPER_DATE_KEY", "paper_date")

    must: List[models.FieldCondition] = []
    if doc_id:
        must.append(
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        )
    if paper_date:
        must.append(
            models.FieldCondition(key=date_key, match=models.MatchValue(value=paper_date))
        )

    qdrant_filter = models.Filter(must=must) if must else None
    results = store.similarity_search(query=query, k=k_val, filter=qdrant_filter)

    unique = {}
    for d in results:
        chunk_id = d.metadata.get("chunk_id")
        if chunk_id and chunk_id not in unique:
            unique[chunk_id] = d

    return list(unique.values())
