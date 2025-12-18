# config/config.py

from __future__ import annotations

"""
Configuration for an English-corpus RAG system (food science papers):
- All retrieval-side processing (filters, query decomposition, HyDE, repack) runs in ENGLISH.
- Final answer is generated in SPANISH, grounded strictly on English evidence.
"""

INPUT_DIR = "data/raw"
OUTPUT_DIR = "outputs/ingest"
BASE_OUTPUT_NAME = "corpus_chunks"
WORDS_PER_CHUNK = 100
OVERLAP = 40
LOG_LEVEL = "INFO"

COLLECTION_NAME = "food_science_papers_v1"
QDRANT_PATH = "data/tmp/langchain_qdrant"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL_NAME = "Qdrant/bm25"

# English-first retrieval stack (query + corpus in English)
EMBEDDING_MODEL = "thenlper/gte-small"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


RETRIEVE_TOP_K = 50
RERANK_TOP_K = 10
MAX_EVIDENCE_ITEMS = 8

PAPER_DATE_KEY = "paper_date"
PAPER_TITLE_KEY = "paper_title"
PAPER_META_PAGES = 2
PAPER_META_MAX_CHARS = 9000


HF_TOKEN = ""
HF_PROVIDER = "auto"
HF_CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 900

PROMPT_QUERY_FILTERS = """\
You are a strict information extraction system.

From the user's query, extract optional filters:
- doc_id: the document identifier if the user mentions a specific PDF (e.g., the filename without extension).
- paper_date: a year or date if explicitly mentioned.

Rules:
- Use null when the information is not explicitly present.
- Output ONLY valid JSON, with no extra text.
- Exact format:
{{"doc_id": <string|null>, "paper_date": <string|null>}}

User query:
{query}
"""

PROMPT_PAPER_METADATA = """\
You are a strict metadata extraction system for academic papers.

From the text, extract:
- paper_title: the main title of the paper
- paper_date: year (preferred) or full date, only if explicitly present

Rules:
- Do NOT guess or hallucinate.
- If missing, use null.
- Output ONLY valid JSON, with no extra text.
- Exact format:
{{"paper_title": <string|null>, "paper_date": <string|null>}}

Text:
{text}
"""

PROMPT_QUERY_DECOMPOSITION = """\
You are an expert query rewriter for a RAG system over food science & technology papers.

Generate 3 to 6 SHORT English sub-queries to retrieve evidence.

Rules:
- Output in ENGLISH only.
- Max 20 words per line.
- Prefer food science technical terms.
- Do NOT add external facts.
- Return one sub-query per line, no numbering, no extra text.

User query:
{query}
"""

PROMPT_HYDE = """\
Write a brief paragraph (max 120 words) that would be an ideal answer to the user's query.
This text will be used ONLY for document retrieval (HyDE).

Rules:
- Output in ENGLISH only.
- One paragraph only.
- Do NOT mention it is hypothetical.
- Use food science technical vocabulary.
- Do NOT invent concrete data not present in the query.

User query:
{query}
"""

PROMPT_REPACK = """\
You are a literal evidence extractor for academic papers.

Goal: extract exact textual fragments from the document that are directly relevant to the query.

Rules:
- Copy text verbatim (no paraphrasing).
- Include only fragments that help answer the query.
- If nothing relevant is found, respond exactly: NO_RELEVANT_CONTENT
- Do NOT add any text outside the extracted fragments.

User query:
{query}

Document:
{doc}

Relevant extractions:
"""

PROMPT_FINAL_ANSWER = """\
Responde la pregunta del usuario usando exclusivamente la evidencia provista.
No menciones modelos, proveedores, tokens ni detalles de implementación.

Reglas:
- No uses conocimiento externo.
- No inventes.
- Cuando recomiendes papers, incluye siempre paper_date y paper_title.
- No traduzcas ni alteres paper_title; mantenlo exactamente como aparece en la evidencia.
- No traduzcas ni reescribas la evidencia; cítala solo mediante [E1], [E2], etc.
- Cita evidencia con [E1], [E2], etc.
- Escribe en español, técnico y claro.

Pregunta del usuario (EN):
{query}

Evidencia (EN):
{evidence}

Formato:
1) Papers recomendados (lista):
   - FECHA/AÑO — TÍTULO: razón breve con citas [E#]
2) Notas (opcional): limitaciones o ausencia de evidencia específica
3) Evidencia usada: lista de [E#]
"""
