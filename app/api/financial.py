from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    FinancialInsightsRequest,
    FinancialInsightsResponse,
    Insight,
    ChatRequest,
    ChatResponse,
)
from app.services.financial_chain import build_financial_chain
from app.services.chat_chain import build_chat_chain
from app.services.guardrails_service import get_guardrails_service, GuardrailsValidationError

router = APIRouter(prefix="/llm", tags=["LLM"])


@router.post(
    "/financial-insights",
    response_model=FinancialInsightsResponse
)
def get_financial_insights(payload: FinancialInsightsRequest):
    guardrails = get_guardrails_service()

    # 1. Validar INPUT
    try:
        guardrails.validate_input(payload.goal)
    except GuardrailsValidationError as e:
        raise HTTPException(status_code=422, detail={
            "error": e.error_type,
            "message": e.message,
        })

    # 2. Invocar el LLM
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

    # 3. Validar OUTPUT
    try:
        insight_message = guardrails.validate_output(insight_message)
    except GuardrailsValidationError as e:
        raise HTTPException(status_code=500, detail={
            "error": e.error_type,
            "message": e.message,
        })

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
    guardrails = get_guardrails_service()

    # 1. Validar INPUT
    try:
        guardrails.validate_input(payload.message)
    except GuardrailsValidationError as e:
        raise HTTPException(status_code=422, detail={
            "error": e.error_type,
            "message": e.message,
        })

    # 2. Invocar el chain con el contexto completo
    chain = build_chat_chain()

    categories_str = ", ".join([
        f"{c.name}: {c.amount} {payload.user_context.currency}"
        for c in payload.categories
    ])

    over_budget_list = [
        b.name for b in payload.budgets
        if b.spent > b.limit
    ]
    over_budget = ", ".join(over_budget_list) if over_budget_list else "Ninguno"

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

    message_content = (
        llm_response.content
        if hasattr(llm_response, "content")
        else str(llm_response)
    )

    # 3. Validar OUTPUT
    try:
        message_content = guardrails.validate_output(message_content)
    except GuardrailsValidationError as e:
        raise HTTPException(status_code=500, detail={
            "error": e.error_type,
            "message": e.message,
        })

    return ChatResponse(
        message=message_content,
        metadata={
            "confidence": 0.85,
            "type": "conversational"
        }
    )
