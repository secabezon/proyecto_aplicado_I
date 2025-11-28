import streamlit as st
from Retrieval import retrieve
from VectorStore import vector_store_config
from qdrant_client import QdrantClient
client = QdrantClient(path="../data/tmp/langchain_qdrant")

# Designing the interface
st.title("RAG Biotechnologies")

vector_store_config(client)
#-------------------------------------------------------------------------------------------------
query = st.text_input("Pregunta sobre los archivos")
st.button("Enviar")
response=retrieve(client,query)
st.write("Respuesta:")
st.write(response)
#streamlit run main.py