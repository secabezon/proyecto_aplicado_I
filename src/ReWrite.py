
from huggingface_hub import InferenceClient
from pathlib import Path
import json

from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

import sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from config.config import CHAT_MODEL



def response_hyde(query: str
):

    client = InferenceClient(token=HF_TOKEN)

    # 2. Create a prompt to generate a hypothetical document
    hyde_prompt = f"""
    Please write a brief, one-paragraph hypothetical document that perfectly answers 
    the following question. This document will be used for a vector search.

    Question: {query}

    Hypothetical Document:
    """

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": hyde_prompt}],
        temperature=0.0
        )
    hypothetical_document = response.choices[0].message.content


    return hypothetical_document

def descomposition_query(query: str
):

    client = InferenceClient(token=HF_TOKEN)

    # 2. Create a prompt to generate a hypothetical document
    decomposition_prompt = f"""
    Decompose the following complex user question into a list of 
    simple, self-contained sub-queries. Return them as a JSON object with a single key "queries".

    Question: {query}
    """

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": decomposition_prompt}]
    )

    sub_queries = json.loads(response.choices[0].message.content)["queries"]    


    return sub_queries

def stepback_query(query: str
):

    client = InferenceClient(token=HF_TOKEN)

    # 2. Create a prompt to generate a hypothetical document
    step_back_prompt = f"""
    You will be given a specific question. Your task is to generate a more general,
    high-level "step-back question" that helps provide context to answer the original.

    Question: {query}

    Step-back question:
    """

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": step_back_prompt}],
        temperature=0.0
    )
    step_back_query = response.choices[0].message.content.strip()   


    return step_back_query