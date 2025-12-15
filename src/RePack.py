from LLM_prompt import llm_prompt


def rePack(query: str, doc: str
):
    
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

    message=[
            {"role": "user", "content": prompt}
        ]
    response=llm_prompt(message,0)
    return response
