from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client.http import models
from langchain_community.embeddings import HuggingFaceEmbeddings
from LLM_prompt import llm_prompt
from pathlib import Path
import json
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

    If no filter is present, return None.

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
    print(response)
    print('-----type------',type(json.loads(response)))
    try:
        return json.loads(response)['filters']['doc_id']
    except:
        return None

def retrieve(
    client,
    query: str,
    k: int = 5,
    doc_id: str = None
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
    
    if doc_id in ("None","null"):
        doc_id=None
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
        search_type="similarity",
        filter=qdrant_filter
    )
    unique={}
    for r in results:
        cid = r.metadata["chunk_id"]
        if cid not in unique:
            unique[cid] = r

    results = list(unique.values())
    return results