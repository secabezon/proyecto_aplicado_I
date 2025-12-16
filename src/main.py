import streamlit as st
from Retrieval import retrieve, extract_self_query
from VectorStore import vector_store_config
from qdrant_client import QdrantClient
from DataIngest import main
from benchmark import eval_metrics_at_k
from RePack import rePack
from ReRank import reRank
from ReWrite import response_hyde, stepback_query, descomposition_query


st.title("RAG Biotechnologies")

# --------------------------------------------------------------------
# EJECUTAR EL PIPELINE SOLO UNA VEZ
# --------------------------------------------------------------------
if "pipeline_ready" not in st.session_state:
    st.write("Preparando datos...")

    main() 
    client = QdrantClient(path="../data/tmp/langchain_qdrant")

    vector_store_config(client)
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
    st.subheader("Respuesta Recuperada naive:")
    st.write(response)
    st.subheader("Respuesta Recuperada Mejorada:")
    doc_id=extract_self_query(query)
    descomposition_querys=descomposition_query(query)
    answers=[]
    for desc_query in descomposition_querys:
        hyde=response_hyde(desc_query)
        response = retrieve(client, hyde, doc_id=doc_id)
        rerank=reRank(query,response)
        for i in rerank:
            repack=rePack(query,i['content'])
            if repack != 'NO_RELEVANT_CONTENT':
                answers.append(repack)
    # st.write('------------hyde-----------')
    # st.write(hyde)
    # st.write('------------Normal-----------')
    # st.write(response)
    # st.write('------------Rerank-----------')
    # st.write(rerank)
    # st.write('------------Final-----------')
    st.write(answers)

# --------------------------------------------------------------------
# BENCHMARK
# --------------------------------------------------------------------
st.subheader("Evaluar desempeño del recuperador (Benchmark)")

if st.button("Ejecutar Benchmark"):
    metrics = eval_metrics_at_k(client, k=100)


    st.subheader("Evaluación con RAG Naive")
    st.write(f"**Precision@5 Naive:** {metrics['precision_at_k_naive']:.3f}")
    st.write(f"**Recall@5 Naive:** {metrics['recall_at_k_naive']:.3f}")

    # Detalle por query
    st.write("### Detalle por consulta")
    for row in metrics["per_query_naive"]:
        st.write({
            "query": row["query"],
            "retrieved": row["retrieved_doc_ids"],
            "relevant": row["relevant_doc_ids"],
            "precision": row["precision"],
            "recall": row["recall"],
            "hits": row["num_hits"],
        })

    
    st.subheader("Evaluación con RAG Processed")


    st.write(f"**Precision@5 Processed:** {metrics['precision_at_k_processed']:.3f}")
    st.write(f"**Recall@5 Processed:** {metrics['recall_at_k_processed']:.3f}")

    # Detalle por query
    st.write("### Detalle por consulta")
    for row in metrics["per_query_processed"]:
        st.write({
            "query": row["query"],
            "retrieved": row["retrieved_doc_ids"],
            "relevant": row["relevant_doc_ids"],
            "precision": row["position_weighted_precision_at_k"],
            "recall": row["recall"],
            "hits": row["num_hits"],
        })
