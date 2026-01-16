from fastapi import APIRouter
from app.models.schemas import (
    FinancialInsightsRequest,
    FinancialInsightsResponse,
    Insight
)

router = APIRouter(prefix="/llm", tags=["LLM"])

@router.post(
    "/financial-insights",
    response_model=FinancialInsightsResponse
)
def get_financial_insights(payload: FinancialInsightsRequest):

    insights = [
        Insight(
            type="education",
            message="Un fondo de emergencia suele cubrir entre 3 y 6 meses de gastos."
        ),
        Insight(
            type="suggestion",
            message="Reducir gastos en entretenimiento puede mejorar tu ahorro mensual."
        )
    ]

    return FinancialInsightsResponse(
        insights=insights,
        confidence=0.85
    )
