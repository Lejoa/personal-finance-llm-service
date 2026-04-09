from pathlib import Path
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.services.llm_provider import get_llm_provider


def build_chat_chain():
    """
    Construye la cadena LangChain para chat conversacional.
    """
    prompt_path = Path("app/prompts/chat_assistant.txt")
    prompt_text = prompt_path.read_text(encoding="utf-8")

    prompt = PromptTemplate(
        input_variables=[
            "message",
            "financial_level",
            "total_income",
            "total_expenses",
            "savings_rate",
            "currency",
            "categories",
            "over_budget"
        ],
        template=prompt_text
    )

    llm = get_llm_provider()
    chain = prompt | llm

    return chain


def build_chat_chain_structured():
    """
    Construye la cadena LangChain para chat conversacional con detección de intención.
    Usa JsonOutputParser para parsear automáticamente el JSON del LLM.
    Retorna un dict con: intent, message, transaction_data (o None).
    """
    prompt_path = Path("app/prompts/chat_assistant_structured.txt")
    prompt_text = prompt_path.read_text(encoding="utf-8")

    prompt = PromptTemplate(
        input_variables=[
            "message",
            "financial_level",
            "total_income",
            "total_expenses",
            "savings_rate",
            "currency",
            "categories",
            "over_budget",
            "current_date",
        ],
        template=prompt_text
    )

    llm = get_llm_provider()
    chain = prompt | llm | JsonOutputParser()

    return chain


# Singleton — se construye una vez al arrancar el servicio
_chat_chain_structured: Any | None = None


def get_chat_chain_structured():
    global _chat_chain_structured
    if _chat_chain_structured is None:
        _chat_chain_structured = build_chat_chain_structured()
    return _chat_chain_structured