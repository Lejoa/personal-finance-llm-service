from pathlib import Path

from langchain_core.prompts import PromptTemplate

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