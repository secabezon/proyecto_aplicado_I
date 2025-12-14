
from huggingface_hub import InferenceClient
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
import sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from config.config import CHAT_MODEL



def llm_prompt(messages: List, temperature: float, response_format: Optional[Dict] = None
):
    
    client = InferenceClient(token=HF_TOKEN)


    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        response_format=response_format,
        temperature=temperature
    )

    return completion.choices[0].message.content.strip()
