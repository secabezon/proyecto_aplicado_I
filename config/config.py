# config/config.py
from __future__ import annotations


INPUT_DIR = "data/raw"
OUTPUT_DIR = "outputs/ingest"
BASE_OUTPUT_NAME = "corpus_chunks"
WORDS_PER_CHUNK = 120
OVERLAP = 40
LOG_LEVEL = "INFO"

COLLECTION_NAME = "food_science_papers_v1"
QDRANT_PATH = "data/tmp/langchain_qdrant"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL_NAME = "Qdrant/bm25"

EMBEDDING_MODEL = "thenlper/gte-small"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVE_TOP_K = 10
RERANK_TOP_K = 5
MAX_EVIDENCE_ITEMS = 10

PAPER_DATE_KEY = "paper_date"
PAPER_TITLE_KEY = "paper_title"
PAPER_META_PAGES = 2
PAPER_META_MAX_CHARS = 9000

HF_TOKEN = ""
HF_PROVIDER = "auto"
HF_CHAT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 900

PROMPT_QUERY_FILTERS = """\
Eres un sistema de extracción estricta.

A partir de la consulta del usuario, extrae filtros opcionales:
- doc_id: el identificador del documento si el usuario menciona un PDF específico (por ejemplo el nombre del archivo sin extensión).
- paper_date: un año o fecha si el usuario la menciona explícitamente.

Reglas:
- Usa null cuando no exista información explícita.
- Responde solo con JSON válido, sin texto adicional.
- Formato exacto:
{{"doc_id": <string|null>, "paper_date": <string|null>}}

Consulta:
{query}
"""

PROMPT_PAPER_METADATA = """\
Eres un sistema de extracción estricta de metadatos de papers académicos.

A partir del texto, extrae:
- paper_title: título principal del paper
- paper_date: año (preferido) o fecha completa, solo si aparece explícitamente

Reglas:
- No inventes.
- Si no está presente, usa null.
- Responde solo con JSON válido y sin texto adicional.
- Formato exacto:
{{"paper_title": <string|null>, "paper_date": <string|null>}}

Texto:
{text}
"""

PROMPT_QUERY_DECOMPOSITION = """\
Eres un reescritor experto de consultas para un sistema RAG sobre papers de ciencia y tecnología de alimentos.

Genera entre 3 y 6 sub-consultas cortas para recuperar evidencia.

Reglas:
- Máximo 20 palabras por línea.
- Prioriza términos técnicos de ciencia de alimentos.
- No agregues hechos externos.
- Devuelve una sub-consulta por línea, sin numeración y sin texto adicional.

Consulta:
{query}
"""

PROMPT_HYDE = """\
Redacta un párrafo breve (máximo 120 palabras) que sería una respuesta ideal a la consulta del usuario.
Este texto se usará solo para recuperación de documentos.

Reglas:
- Un solo párrafo.
- No menciones que es hipotético.
- Usa vocabulario técnico de ciencia de alimentos.
- No inventes datos concretos si no están en la consulta.

Consulta:
{query}
"""

PROMPT_REPACK = """\
Eres un asistente de extracción literal de evidencia desde papers.

Objetivo: extraer fragmentos textuales exactos del documento que sean directamente relevantes para la consulta.

Reglas:
- Copia literalmente, sin parafrasear.
- Incluye solo fragmentos útiles para responder.
- Si no hay nada relevante, responde exactamente: NO_RELEVANT_CONTENT
- No agregues texto fuera de los fragmentos extraídos.

Consulta:
{query}

Documento:
{doc}

Extracciones relevantes:
"""

PROMPT_FINAL_ANSWER = """\
Responde a la pregunta del usuario usando exclusivamente la evidencia provista.
No menciones modelos, proveedores, tokens ni detalles de implementación.

Reglas:
- No uses conocimiento externo.
- No inventes.
- Cuando recomiendes papers, incluye siempre paper_date y paper_title.
- Cita evidencia con [E1], [E2], etc.
- Escribe en español, técnico y claro.

Pregunta del usuario:
{query}

Evidencia:
{evidence}

Formato:
1) Papers recomendados (lista):
   - FECHA/AÑO — TÍTULO: razón breve con citas [E#]
2) Notas (opcional): limitaciones o ausencia de evidencia específica
3) Evidencia usada: lista de [E#]
"""
