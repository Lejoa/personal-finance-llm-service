from pathlib import Path

from langchain_core.prompts import PromptTemplate

from app.services.llm_provider import get_llm_provider


def build_financial_chain():
    """
    Construye la cadena LangChain para insights financieros pedagógicos estructurados.
    """
    prompt_path = Path("app/prompts/financial_education.txt")
    prompt_text = prompt_path.read_text(encoding="utf-8")

    prompt = PromptTemplate(
        input_variables=[
            "financial_level",
            "period",
            "total_income",
            "total_expenses",
            "savings_rate",
            "previous_savings_rate",
            "currency",
            "categories",
            "over_budget",
            "top_tip",
            "goal",
        ],
        template=prompt_text,
    )

    llm = get_llm_provider()
    chain = prompt | llm

    return chain