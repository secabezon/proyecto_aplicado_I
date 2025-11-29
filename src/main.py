import streamlit as st
from Retrieval import retrieve
from VectorStore import vector_store_config
from qdrant_client import QdrantClient
from DataIngest import main
from benchmark import eval_metrics_at_k  # <-- IMPORTANTE


st.title("RAG Biotechnologies")

# --------------------------------------------------------------------
# EJECUTAR EL PIPELINE SOLO UNA VEZ
# --------------------------------------------------------------------
if "pipeline_ready" not in st.session_state:
    st.write("Preparando datos...")

    main()  # genera chunks
    client = QdrantClient(path="../data/tmp/langchain_qdrant")
    vector_store_config(client)  # crea colección y sube embeddings

    st.session_state.client = client
    st.session_state.pipeline_ready = True
    st.success("Pipeline listo.")

client = st.session_state.client

# --------------------------------------------------------------------
# CONSULTA
# --------------------------------------------------------------------
query = st.text_input("Pregunta sobre los archivos")

if st.button("Enviar"):
    response = retrieve(client, query)
    st.subheader("Respuesta Recuperada:")
    st.write(response)

# --------------------------------------------------------------------
# BENCHMARK
# --------------------------------------------------------------------
st.subheader("Evaluar desempeño del recuperador (Benchmark)")

if st.button("Ejecutar Benchmark"):
    metrics = eval_metrics_at_k(client, k=5)

    st.write(f"**Precision@5:** {metrics['precision_at_k']:.3f}")
    st.write(f"**Recall@5:** {metrics['recall_at_k']:.3f}")

    # Detalle por query
    st.write("### Detalle por consulta")
    for row in metrics["per_query"]:
        st.write({
            "query": row["query"],
            "retrieved": row["retrieved_doc_ids"],
            "relevant": row["relevant_doc_ids"],
            "precision": row["precision"],
            "recall": row["recall"],
            "hits": row["num_hits"],
        })
