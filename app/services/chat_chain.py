from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser

from app.services.llm_provider import get_llm_provider


def build_chat_chain_structured():
    """
    Construye la cadena LangChain para chat conversacional con detección de intención.

    Usa ChatPromptTemplate con MessagesPlaceholder para inyectar el historial de
    conversación como mensajes nativos del modelo (HumanMessage / AIMessage), lo que
    permite al LLM mantener coherencia en conversaciones multi-turno sin requerir que
    el usuario repita el contexto de preguntas anteriores.

    Retorna un dict con: intent, message, transaction_data (o None).
    """
    prompt_path = Path("app/prompts/chat_assistant_structured.txt")
    system_text = prompt_path.read_text(encoding="utf-8")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{message}"),
    ])

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