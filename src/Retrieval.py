from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client.http import models
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from config.config import EMBEDDING_MODEL


def extract_self_query(query: str):
    prompt = f"""
    You are an information extraction system.

    Extract structured filters from the user query.
    Allowed fields:
    - doc_id (string)

    If no filter is present, return null.

    Return ONLY valid JSON with keys:
    - query_text
    - filters

    User query:
    {query}
    """
    client = InferenceClient(token=HF_TOKEN)
    response = client.text_generation(
        prompt,
        model="google/flan-t5-base",
        max_new_tokens=200,
        temperature=0.0
    )

    return response.generated_text

def retrieve(
    client,
    query: str,
    k: int = 5
):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    qdrant = QdrantVectorStore(
        client=client,
        collection_name="bioactives_collection",
        embedding=embedding_model,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID
    )

    doc_id=extract_self_query(query)['doc_id']
    qdrant_filter=None
    if doc_id is not None:
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.doc_id",
                    match=models.MatchValue(
                        value=doc_id  
                    )
                )
            ]
        )

    results = qdrant.search(
        query=query,
        k=k,
        filter=qdrant_filter
    )

    return results