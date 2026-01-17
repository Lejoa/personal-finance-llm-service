from fastapi import APIRouter
from app.models.schemas import (
    FinancialInsightsRequest,
    FinancialInsightsResponse,
    Insight
)
from app.services.financial_chain import build_financial_chain

router = APIRouter(prefix="/llm", tags=["LLM"])

@router.post(
    "/financial-insights",
    response_model=FinancialInsightsResponse
)
def get_financial_insights(payload: FinancialInsightsRequest):

    chain = build_financial_chain()

    categories = ", ".join([c.name for c in payload.categories])

    over_budget = (
        payload.budgets[0].name
        if payload.budgets and payload.budgets[0].spent > payload.budgets[0].limit
        else "Ninguno"
    )

    response_text = chain.invoke({
        "financial_level": payload.user_context.financial_level,
        "total_income": payload.summary.total_income,
        "total_expenses": payload.summary.total_expenses,
        "currency": payload.user_context.currency,
        "categories": categories,
        "over_budget": over_budget,
    })

    insights = [
        Insight(type="education", message=response_text)
    ]

    return FinancialInsightsResponse(
        insights=insights,
        confidence=0.85
    )
