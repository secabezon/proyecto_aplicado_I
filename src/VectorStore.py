import numpy as np
import json
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from config.config import EMBEDDING_MODEL

def vector_store_config(client):
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    chunks_path = Path("data/chunks.json")

    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    docs = []

    embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
    )
    for chunk in chunks:
        docs.append(
            Document(
            page_content=chunk["text"],
            metadata={
                "order": chunk["order"],
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],  # <-- acá agregas tu propio id
            }
        )
        )


    for col in client.get_collections().collections:
        client.delete_collection(collection_name=col.name)

    client.create_collection(
    collection_name="bioactives_collection",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="bioactives_collection",
        embedding=embedding_model,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID
    )
    vector_store.add_documents(docs)