from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import PromptTemplate
from pathlib import Path


def build_financial_chain():

    prompt_path = Path("app/prompts/financial_education.txt")
    prompt_text = prompt_path.read_text(encoding="utf-8")

    prompt = PromptTemplate(
        input_variables=[
            "financial_level",
            "total_income",
            "total_expenses",
            "currency",
            "categories",
            "over_budget",
        ],
        template=prompt_text,
    )

    llm = FakeListLLM(
        responses=[
            "Un fondo de emergencia es un ahorro que te protege ante imprevistos. "
            "Idealmente cubre entre 3 y 6 meses de tus gastos básicos."
        ]
    )

    chain = prompt | llm

    return chain
