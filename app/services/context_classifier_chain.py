from pathlib import Path
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.services.llm_provider import get_llm_provider

VALID_CONTEXT_TYPES = {
    "transaction", "question", "trends", "historical",
    "budget", "categories", "savings", "none",
}


def build_context_classifier_chain():
    """
    Construye la cadena LangChain para clasificar el tipo de contexto financiero
    que necesita un mensaje del usuario.

    Recibe {message} y {current_date} como variables. current_date permite al LLM
    resolver referencias temporales relativas ("hace 2 meses", "este año", "en abril")
    y convertirlas a valores YYYY-MM concretos en el period_hint.
    """
    prompt_path = Path("app/prompts/context_classifier.txt")
    prompt_text = prompt_path.read_text(encoding="utf-8")

    prompt = PromptTemplate(
        input_variables=["message", "current_date", "available_categories"],
        template=prompt_text,
    )

    llm = get_llm_provider()
    chain = prompt | llm | JsonOutputParser()

    return chain


# Singleton — se construye una vez al arrancar el servicio
_context_classifier_chain: Any | None = None


def get_context_classifier_chain():
    global _context_classifier_chain
    if _context_classifier_chain is None:
        _context_classifier_chain = build_context_classifier_chain()
    return _context_classifier_chain
