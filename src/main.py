from __future__ import annotations

import shutil
from typing import List

import streamlit as st
from qdrant_client import QdrantClient

from AnswerGenerator import EvidenceItem, generate_final_answer
from DataIngest import main as ingest_main
from RePack import repack
from ReRank import rerank
from Retrieval import extract_query_filters, retrieve
from ReWrite import decompose_query, generate_hyde_document
from VectorStore import vector_store_config
from benchmark import eval_metrics_at_k
from project_config import cfg, project_root


st.title("RAG Food Science Papers")

root = project_root()
qdrant_path = root / getattr(cfg, "QDRANT_PATH", "data/tmp/langchain_qdrant")


def build_qdrant_client() -> QdrantClient:
    """
    Create a local Qdrant client using an absolute path.

    Returns:
        Qdrant client.
    """
    qdrant_path.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(qdrant_path))


if "pipeline_ready" not in st.session_state:
    st.write("Preparando datos (ingesta y carga en vector store)...")

    ingest_main()

    try:
        if qdrant_path.exists():
            shutil.rmtree(qdrant_path)
    except Exception:
        pass

    client_init = build_qdrant_client()
    vector_store_config(client_init)

    st.session_state.client = client_init
    st.session_state.pipeline_ready = True
    st.success("Pipeline listo.")

client: QdrantClient = st.session_state.client

query = st.text_input("Pregunta sobre los PDFs en data/raw (papers de alimentos)")

if st.button("Enviar"):
    st.subheader("Recuperación (debug)")
    naive_docs = retrieve(client, query, k=5)
    st.write(naive_docs)

    st.subheader("Pipeline aumentado")

    try:
        doc_id, paper_date = extract_query_filters(query)
    except Exception as exc:
        st.error(f"Fallo en extracción de filtros: {exc}")
        doc_id, paper_date = None, None

    try:
        subqueries = decompose_query(query)
    except Exception as exc:
        st.error(f"Fallo en descomposición de consulta: {exc}")
        subqueries = [query]

    evidence_items: List[EvidenceItem] = []

    for subquery in subqueries:
        try:
            hyde_text = generate_hyde_document(subquery)
        except Exception:
            hyde_text = subquery

        docs = retrieve(
            client,
            hyde_text,
            k=int(getattr(cfg, "RETRIEVE_TOP_K", 10)),
            doc_id=doc_id,
            paper_date=paper_date,
        )

        reranked = rerank(query, docs)

        for item in reranked:
            extracted = repack(query, str(item.get("content", "")))
            if extracted and extracted != "NO_RELEVANT_CONTENT":
                evidence_items.append(
                    EvidenceItem(
                        doc_id=item.get("doc_id"),
                        chunk_id=item.get("chunk_id"),
                        paper_date=str(item.get("paper_date", "UNKNOWN")),
                        paper_title=str(item.get("paper_title", "UNKNOWN")),
                        text=extracted,
                    )
                )

    final_answer = generate_final_answer(query, evidence_items)

    st.subheader("Respuesta")
    st.write(final_answer)

    st.subheader("Detalle de evidencia:")
    st.write([e.__dict__ for e in evidence_items])

st.subheader("Evaluar recuperador (Benchmark)")

if st.button("Ejecutar Benchmark"):
    metrics = eval_metrics_at_k(client, k=5)
    st.write(f"Precision@5: {metrics['precision_at_k']:.3f}")
    st.write(f"Recall@5: {metrics['recall_at_k']:.3f}")

    st.write("Detalle por consulta")
    for row in metrics["per_query"]:
        st.write(
            {
                "query": row["query"],
                "retrieved": row["retrieved_doc_ids"],
                "relevant": row["relevant_doc_ids"],
                "precision": row["precision"],
                "recall": row["recall"],
                "hits": row["num_hits"],
            }
        )
