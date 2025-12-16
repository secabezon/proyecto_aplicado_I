import json
from LLM_prompt import llm_prompt


def response_hyde(query: str
):

    hyde_prompt = f"""
    Please write a brief, one-paragraph hypothetical document that perfectly answers 
    the following question. This document will be used for a vector search.

    Question: {query}

    Hypothetical Document:
    """
    message=[
            {"role": "user", "content": hyde_prompt}
        ]
    hypothetical_document=llm_prompt(message,0)


    return hypothetical_document

def descomposition_query(query: str
):

    decomposition_prompt = f"""
    Decompose the following complex user question into a list of 
    simple, self-contained sub-queries. Return them as a JSON object with a single key "queries".

    Question: {query}
    """
    
    message=[
            {"role": "user", "content": decomposition_prompt}
        ]
    response_format={"type": "json_object"}
    response=llm_prompt(message,0,response_format)
    sub_queries = json.loads(response)["queries"]    


    return sub_queries

def stepback_query(query: str
):


    step_back_prompt = f"""
    You will be given a specific question. Your task is to generate a more general,
    high-level "step-back question" that helps provide context to answer the original.

    Question: {query}

    Step-back question:
    """


    message=[
           {"role": "user", "content": step_back_prompt}
        ]
    response=llm_prompt(message,0)
    step_back_query = response.strip()   


    return step_back_query