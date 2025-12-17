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
    desc_querys=[]
    sqs=[]
    hydes=[]
    responses=[]
    reranks=[]
    context=[]
    for desc_query in descomposition_querys:
        sq=stepback_query(desc_query)
        hyde=response_hyde(sq)
        response = retrieve(client, hyde, doc_id=doc_id)
        rerank=reRank(query,response)
        for i in rerank:
            repack=rePack(query,i['content'])
            if repack != 'NO_RELEVANT_CONTENT' or 'NO_RELEVANT_CONTENT' not in repack:
                context.append(repack)
        desc_querys.append(desc_query)
        hydes.append(hyde)
        responses.append(response)
        reranks.append(rerank)
    st.write('------------Query Descompuesta - descomposition_querys -----------')
    st.write(desc_querys)
    st.write('------------Pregunta más amplia - stepback query -----------')
    st.write(sqs)
    st.write('------------Respuesta LLM - HYDE -----------')
    st.write(hydes)
    st.write('------------Respuesta Proceso - Reitrieve -----------')
    st.write(responses)
    st.write('------------Orden Relevantes - Rerank-----------')
    st.write(reranks)
    st.write('------------Contexto - Repack-----------')
    st.write(context)

# --------------------------------------------------------------------
# BENCHMARK
# --------------------------------------------------------------------
st.subheader("Evaluar desempeño del recuperador (Benchmark)")

if st.button("Ejecutar Benchmark"):
    metrics = eval_metrics_at_k(client, k=100)


    st.subheader("Evaluación con RAG Naive")
    st.write(f"**Precision@5 Naive:** {metrics['precision_at_k_naive']:.3f}")
    st.write(f"**Position weighted precision @5 Naive:** {metrics['pwp_naive']:.3f}")
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
            "Position weighted precision": row["position_weighted_precision_at_k"],
        })

    
    st.subheader("Evaluación con RAG Processed")


    st.write(f"**Precision@5 Processed:** {metrics['precision_at_k_processed']:.3f}")
    st.write(f"**Position weighted precision @5 Processed:** {metrics['pwp_processed']:.3f}")
    st.write(f"**Recall@5 Processed:** {metrics['recall_at_k_processed']:.3f}")

    # Detalle por query
    st.write("### Detalle por consulta")
    for row in metrics["per_query_processed"]:
        st.write({
            "query": row["query"],
            "retrieved": row["retrieved_doc_ids"],
            "relevant": row["relevant_doc_ids"],
            "precision": row["precision"],
            "recall": row["recall"],
            "hits": row["num_hits"],
            "Position weighted precision": row["position_weighted_precision_at_k"],
        })
