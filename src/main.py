from __future__ import annotations

import logging
import shutil
import time
from typing import Any, Dict, List

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


logging.basicConfig(level=logging.INFO)

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


def _doc_to_debug_row(doc: Any, max_chars: int = 240) -> Dict[str, Any]:
    """
    Convert a LangChain Document into a compact debug-friendly dictionary.

    Args:
        doc: LangChain Document-like object.
        max_chars: Max number of characters for the preview.

    Returns:
        Dictionary with id and a short preview.
    """
    doc_id = doc.metadata.get("doc_id")
    order = doc.metadata.get("order")
    chunk_id = doc.metadata.get("chunk_id")
    preview = (doc.page_content or "").strip().replace("\n", " ")
    if len(preview) > max_chars:
        preview = preview[: max_chars - 3] + "..."
    return {
        "id": f"{doc_id}:{order}",
        "doc_id": doc_id,
        "order": order,
        "chunk_id": chunk_id,
        "preview": preview,
    }


def _render_per_query_details(title: str, rows: List[Dict[str, Any]], key_prefix: str) -> None:
    st.write(f"### {title}")
    if not rows:
        st.info("No hay consultas en el set de evaluación.")
        return

    selector_key = f"{key_prefix}_query_selector"

    # Inicializar SOLO si no existe
    if selector_key not in st.session_state:
        st.session_state[selector_key] = 0

    with st.expander(f"Ver {title}", expanded=False):
        labels = [f"{i+1}) {r['query']}" for i, r in enumerate(rows)]

        sel = st.selectbox(
            "Selecciona una consulta para inspeccionar",
            options=list(range(len(rows))),
            format_func=lambda i: labels[i],
            key=selector_key
        )

        row = rows[sel]

        st.write(
            f"**Hits:** {row['num_hits']} | "
            f"**Precision@{row['k_eval']}:** {row['precision']:.3f} | "
            f"**Recall@{row['k_eval']}:** {row['recall']:.3f} | "
            f"**PWP:** {row['position_weighted_precision_at_k']:.3f}"
        )

        box = st.container(height=320)
        with box:
            st.json({
                "query": row["query"],
                "retrieved": row["retrieved_doc_ids"],
                "relevant": row["relevant_doc_ids"],
                "metrics": {
                    "hits": row["num_hits"],
                    "precision": row["precision"],
                    "recall": row["recall"]
                }
            })



if "pipeline_ready" not in st.session_state:
    st.write("Preparando datos (ingesta y carga en vector store)...")

    with st.spinner("Ejecutando ingesta de PDFs..."):
        ingest_main()

    try:
        if qdrant_path.exists():
            shutil.rmtree(qdrant_path)
    except Exception:
        pass

    client_init = build_qdrant_client()

    st.write("Creando colección y subiendo chunks al vector store...")
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def _progress_cb(done: int, total: int) -> None:
        pct = int((done / max(total, 1)) * 100)
        progress_bar.progress(pct)
        progress_text.write(f"Indexando: {done}/{total} chunks ({pct}%)")

    with st.spinner("Indexando embeddings (esto puede tardar la primera vez)..."):
        vector_store_config(client_init, progress_callback=_progress_cb)

    progress_text.empty()
    progress_bar.empty()

    st.session_state.client = client_init
    st.session_state.pipeline_ready = True
    st.success("Pipeline listo.")

client: QdrantClient = st.session_state.client

query = st.text_input("Question (EN) about the PDFs in data/raw (food science papers)")

if st.button("Enviar"):
    st.subheader("Recuperación naive (debug)")
    t0 = time.perf_counter()
    naive_docs = retrieve(client, query, k=5)
    naive_debug = [_doc_to_debug_row(d) for d in naive_docs]
    st.caption(f"Tiempo recuperación naive: {time.perf_counter() - t0:.2f}s")

    with st.expander("Ver resultados naive (top-5)", expanded=False):
        st.json(naive_debug)

    st.subheader("Pipeline aumentado")

    stage = st.empty()
    progress = st.progress(0)

    stage.write("Extrayendo filtros de consulta (self-query)...")
    t1 = time.perf_counter()
    try:
        doc_id, paper_date = extract_query_filters(query)
    except Exception as exc:
        st.error(f"Fallo en extracción de filtros: {exc}")
        doc_id, paper_date = None, None
    st.caption(f"Tiempo self-query: {time.perf_counter() - t1:.2f}s")

    stage.write("Descomponiendo consulta...")
    t2 = time.perf_counter()
    try:
        subqueries = decompose_query(query)
    except Exception as exc:
        st.error(f"Fallo en descomposición de consulta: {exc}")
        subqueries = [query]
    st.caption(f"Tiempo descomposición: {time.perf_counter() - t2:.2f}s")

    rerank_top_k = int(getattr(cfg, "RERANK_TOP_K", 5))
    total_steps = 2 + (len(subqueries) * (3 + rerank_top_k)) + 1
    step = 0

    evidence_items: List[EvidenceItem] = []

    for i, subquery in enumerate(subqueries, start=1):
        stage.write(f"Generando HyDE para subconsulta {i}/{len(subqueries)}...")
        t_hyde = time.perf_counter()
        try:
            hyde_text = generate_hyde_document(subquery)
        except Exception:
            hyde_text = subquery
        step += 1
        progress.progress(int((step / total_steps) * 100))
        st.caption(f"Tiempo HyDE (subconsulta {i}): {time.perf_counter() - t_hyde:.2f}s")

        stage.write(f"Recuperando evidencia (hybrid search) para subconsulta {i}/{len(subqueries)}...")
        t_ret = time.perf_counter()
        docs = retrieve(
            client,
            hyde_text,
            k=int(getattr(cfg, "RETRIEVE_TOP_K", 10)),
            doc_id=doc_id,
            paper_date=paper_date,
        )
        step += 1
        progress.progress(int((step / total_steps) * 100))
        st.caption(f"Tiempo retrieval (subconsulta {i}): {time.perf_counter() - t_ret:.2f}s")

        stage.write(f"Rerankeando chunks para subconsulta {i}/{len(subqueries)}...")
        t_rr = time.perf_counter()
        reranked = rerank(query, docs)
        step += 1
        progress.progress(int((step / total_steps) * 100))
        st.caption(f"Tiempo rerank (subconsulta {i}): {time.perf_counter() - t_rr:.2f}s")

        for j, item in enumerate(reranked, start=1):
            stage.write(f"Repacking evidencia (subconsulta {i}/{len(subqueries)} | chunk {j}/{len(reranked)})...")
            t_rp = time.perf_counter()
            extracted = repack(query, str(item.get("content", "")))
            step += 1
            progress.progress(int((step / total_steps) * 100))
            st.caption(f"Tiempo repack (subconsulta {i}, chunk {j}): {time.perf_counter() - t_rp:.2f}s")

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

    stage.write("Generando respuesta final (grounded generation)...")
    t_final = time.perf_counter()
    final_answer = generate_final_answer(query, evidence_items)
    step += 1
    progress.progress(int((step / total_steps) * 100))
    st.caption(f"Tiempo generación final: {time.perf_counter() - t_final:.2f}s")

    stage.empty()
    progress.empty()

    st.subheader("Respuesta")
    st.write(final_answer)

    st.subheader("Evidencia usada")
    st.write(f"Total evidencia: {len(evidence_items)}")
    with st.expander("Ver evidencia (JSON)", expanded=False):
        box = st.container(height=320)
        with box:
            st.json([e.__dict__ for e in evidence_items])

st.subheader("Evaluar desempeño del recuperador (Benchmark)")

k_eval = st.slider("k para métricas (Precision@k / Recall@k)", min_value=1, max_value=20, value=5, step=1)

# 1. Inicializar el estado de sesión si no existe
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

# 2. El botón SOLO gatilla el cálculo y lo guarda en la sesión
if st.button("Ejecutar Benchmark"):
    with st.spinner("Ejecutando evaluación completa..."):
        st.session_state.benchmark_results = eval_metrics_at_k(client, k_eval=int(k_eval))
    st.success("¡Benchmark finalizado!")

# 3. La visualización ocurre SIEMPRE que haya datos en la sesión (fuera del bloque del botón)
if st.session_state.benchmark_results is not None:
    metrics = st.session_state.benchmark_results
    
    st.subheader("Evaluación con Baseline (hybrid retrieve, sin LLM steps)")
    st.write(f"**Precision@{metrics['k_eval']} Baseline:** {metrics['precision_at_k_naive']:.3f}")
    st.write(f"**Position weighted precision@{metrics['k_eval']} Baseline:** {metrics['pwp_naive']:.3f}")
    st.write(f"**Recall@{metrics['k_eval']} Baseline:** {metrics['recall_at_k_naive']:.3f}")

    # Ahora el selector funcionará sin reiniciar el cálculo
    _render_per_query_details("Detalle por consulta (Baseline)", metrics["per_query_naive"], key_prefix="baseline")

    st.divider()

    st.subheader("Evaluación con Pipeline Final (filters + decomposition + HyDE + rerank)")
    st.write(f"**Precision@{metrics['k_eval']} Final:** {metrics['precision_at_k_processed']:.3f}")
    st.write(f"**Position weighted precision@{metrics['k_eval']} Final:** {metrics['pwp_processed']:.3f}")
    st.write(f"**Recall@{metrics['k_eval']} Final:** {metrics['recall_at_k_processed']:.3f}")

    _render_per_query_details("Detalle por consulta (Final)", metrics["per_query_processed"], key_prefix="final")