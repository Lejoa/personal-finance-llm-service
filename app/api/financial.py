from fastapi import APIRouter
from app.models.schemas import (
    FinancialInsightsRequest,
    FinancialInsightsResponse,
    Insight,
    ChatRequest,
    ChatResponse,
)
from app.services.financial_chain import build_financial_chain
from app.services.chat_chain import build_chat_chain

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

    llm_response = chain.invoke({
        "financial_level": payload.user_context.financial_level,
        "total_income": payload.summary.total_income,
        "total_expenses": payload.summary.total_expenses,
        "currency": payload.user_context.currency,
        "categories": categories,
        "over_budget": over_budget,
    })

    insight_message = (
        llm_response.content
        if hasattr(llm_response, "content")
        else str(llm_response)
    )

    insights = [
        Insight(
            type="education",
            message=insight_message
        )
    ]

    return FinancialInsightsResponse(
        insights=insights,
        confidence=0.85
    )


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    """
    Endpoint para chat conversacional con contexto financiero.
    Recibe el mensaje del usuario junto con su contexto financiero
    y retorna una respuesta del asistente.
    """
    chain = build_chat_chain()

    # Preparar categorías como string
    categories_str = ", ".join([
        f"{c.name}: {c.amount} {payload.user_context.currency}"
        for c in payload.categories
    ])

    # Identificar presupuestos excedidos
    over_budget_list = [
        b.name for b in payload.budgets
        if b.spent > b.limit
    ]
    over_budget = ", ".join(over_budget_list) if over_budget_list else "Ninguno"

    # Invocar el chain con el contexto completo
    llm_response = chain.invoke({
        "message": payload.message,
        "financial_level": payload.user_context.financial_level,
        "total_income": payload.financial_summary.total_income,
        "total_expenses": payload.financial_summary.total_expenses,
        "savings_rate": payload.financial_summary.savings_rate,
        "currency": payload.user_context.currency,
        "categories": categories_str,
        "over_budget": over_budget,
    })

    # Extraer contenido de la respuesta
    message_content = (
        llm_response.content
        if hasattr(llm_response, "content")
        else str(llm_response)
    )

    return ChatResponse(
        message=message_content,
        metadata={
            "confidence": 0.85,
            "type": "conversational"
        }
    )
