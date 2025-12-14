
from huggingface_hub import InferenceClient
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
import sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from config.config import CHAT_MODEL



def rePank(query: str, doc: str
):
    
    client = InferenceClient(token=HF_TOKEN)
    prompt = f"""You are a precise text extraction assistant. Your task is to extract ONLY the sentences or phrases from the given document that are directly relevant to answering the user's query.

    Rules:
    1. Extract exact sentences/phrases from the document - do not paraphrase or summarize
    2. Only extract content that directly helps answer the query
    3. If multiple relevant sentences exist, include all of them
    4. If no content in the document is relevant to the query, respond with: "NO_RELEVANT_CONTENT"
    5. Preserve the original wording exactly as it appears in the document

    Query: {query}

    Document:
    {doc}

    Relevant extractions:"""


    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0 
    )

    return completion.choices[0].message.content.strip()
