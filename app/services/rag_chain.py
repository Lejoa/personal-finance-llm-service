from pathlib import Path
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.services.llm_provider import get_llm_provider


def build_rag_education_chain():
    """
    Construye la cadena LangChain para respuestas educativas con RAG.

    A diferencia del chain de chat (que usa ChatPromptTemplate + historial),
    este chain es stateless: recibe {question} y {rag_context} y genera
    una respuesta en texto plano citando las fuentes recuperadas.
    """
    prompt_path = Path("app/prompts/rag_education.txt")
    prompt_text = prompt_path.read_text(encoding="utf-8")

    prompt = PromptTemplate(
        input_variables=["question", "rag_context"],
        template=prompt_text,
    )

    llm = get_llm_provider()
    chain = prompt | llm | StrOutputParser()

    return chain


_rag_education_chain: Any | None = None


def get_rag_education_chain():
    global _rag_education_chain
    if _rag_education_chain is None:
        _rag_education_chain = build_rag_education_chain()
    return _rag_education_chain
