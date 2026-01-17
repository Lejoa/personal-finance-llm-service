import os
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 


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

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.3,
    )

    chain = prompt | llm

    return chain
