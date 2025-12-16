# Embedding config
EMBEDDING_MODEL = "thenlper/gte-small"
# Data ingest config
INPUT_DIR = "data/raw"
OUTPUT_DIR = "outputs/ingest"
BASE_OUTPUT_NAME = "corpus_chunks"
WORDS_PER_CHUNK = 100
OVERLAP = 50
LOG_LEVEL = "INFO"
#LLM IMPROVE QUERY
# CHAT_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
CHAT_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
CROSSENCODER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"