from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client.http import models
from langchain_community.embeddings import HuggingFaceEmbeddings
from LLM_prompt import llm_prompt
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
    message=[
        {"role": "user", "content": prompt}
    ]
    response=llm_prompt(message,0)
    return response.choices[0].message.content

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