import logging
from datetime import date as date_type

from fastapi import APIRouter, HTTPException
from langchain_core.exceptions import OutputParserException
from app.models.schemas import (
    FinancialInsightsRequest,
    FinancialInsightsResponse,
    Insight,
    ChatRequest,
    ChatResponse,
    TransactionAction,
    GuardrailsErrorResponse,
    ClassifyContextRequest,
    ClassifyContextResponse,
)
from app.services.financial_chain import get_financial_chain
from app.services.chat_chain import get_chat_chain_structured
from app.services.context_classifier_chain import get_context_classifier_chain, VALID_CONTEXT_TYPES
from app.services.guardrails_service import get_guardrails_service, GuardrailsValidationError
from app.services.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])


@router.get(
    "/smoke-test",
    summary="Verifica que el LLM responde correctamente",
    description=(
        "Envía un prompt mínimo al LLM configurado y retorna la respuesta cruda. "
        "Útil para verificar conectividad y disponibilidad del modelo antes de "
        "ejecutar suites de evaluación."
    ),
)
def smoke_test():
    try:
        llm = get_llm_provider()
        response = llm.invoke("Responde solo con la palabra 'ok'.")
        content = response.content if hasattr(response, "content") else str(response)
        return {
            "status": "ok",
            "model_response": content.strip(),
        }
    except Exception as exc:
        logger.exception("smoke-test LLM error")
        raise HTTPException(status_code=500, detail={
            "error": type(exc).__name__,
            "message": str(exc),
        })


@router.post(
    "/financial-insights",
    response_model=FinancialInsightsResponse,
    responses={
        422: {
            "model": GuardrailsErrorResponse,
            "description": (
                "El campo `goal` fue rechazado por Guardrails: "
                "off-topic, contenido tóxico o información personal (PII)."
            ),
        },
        500: {
            "model": GuardrailsErrorResponse,
            "description": "La respuesta del LLM no superó la validación de output.",
        },
    },
    summary="Genera insights educativos financieros",
    description=(
        "Recibe el resumen financiero del usuario y su objetivo (`goal`), "
        "y retorna un insight educativo generado por el LLM.\n\n"
        "El campo `goal` pasa por validación **Guardrails INPUT** "
        "(RestrictToTopic + ToxicLanguage + DetectPII) antes de llegar al modelo. "
        "La respuesta pasa por validación **Guardrails OUTPUT** (ToxicLanguage) "
        "antes de retornarse.\n\n"
        "> **Nota de latencia:** La generación puede tomar entre 5 y 120 segundos "
        "según el modelo y hardware. Configura el cliente con `timeout ≥ 120s`."
    ),
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
    chain = get_financial_chain()

    categories = ", ".join([
        f"{c.name}: {c.amount} {payload.user_context.currency}"
        for c in payload.categories
    ])

    over_budget_list = [
        f"{b.name} (límite {b.limit}, gastado {b.spent} {payload.user_context.currency})"
        for b in payload.budgets
        if b.spent > b.limit
    ]
    over_budget = ", ".join(over_budget_list) if over_budget_list else "Ninguno"

    previous_savings_rate = payload.summary.previous_savings_rate
    savings_rate_comparison = (
        f"{previous_savings_rate:.1f}%"
        if previous_savings_rate is not None
        else "sin datos del mes anterior"
    )

    llm_response = chain.invoke({
        "financial_level": payload.user_context.financial_level,
        "period": payload.summary.period,
        "total_income": payload.summary.total_income,
        "total_expenses": payload.summary.total_expenses,
        "savings_rate": payload.summary.savings_rate,
        "previous_savings_rate": savings_rate_comparison,
        "currency": payload.user_context.currency,
        "categories": categories,
        "over_budget": over_budget,
        "top_tip": payload.top_tip or "No disponible",
        "goal": payload.goal,
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


@router.post(
    "/classify-context",
    response_model=ClassifyContextResponse,
    responses={
        422: {
            "model": GuardrailsErrorResponse,
            "description": (
                "El campo `message` fue rechazado por Guardrails: "
                "contenido tóxico o información personal (PII)."
            ),
        },
    },
    summary="Clasifica el tipo de contexto financiero que necesita un mensaje",
    description=(
        "Primer paso del pipeline de chat: valida el mensaje con Guardrails Safety "
        "(ToxicLanguage + DetectPII) y clasifica el tipo de contexto financiero adicional "
        "necesario (transaction, question, trends, budget, categories, savings, none). "
        "Al validar aquí se evita el llamado al LLM de chat si el mensaje es inválido. "
        "En caso de error de clasificación retorna 'none' para degradar con seguridad."
    ),
)
async def classify_context(payload: ClassifyContextRequest):
    guardrails = get_guardrails_service()

    # 1. Safety guard — primer filtro, evita llamadas innecesarias al LLM de chat
    try:
        await guardrails.async_validate_safety(payload.message)
    except GuardrailsValidationError as e:
        raise HTTPException(status_code=422, detail={
            "error": e.error_type,
            "message": e.message,
        })

    # 2. Clasificar contexto
    chain = get_context_classifier_chain()
    try:
        parsed = chain.invoke({"message": payload.message})
        context_type = parsed.get("context_type", "none")
        if context_type not in VALID_CONTEXT_TYPES:
            context_type = "none"
    except Exception:
        context_type = "none"

    return ClassifyContextResponse(context_type=context_type)


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        422: {
            "model": GuardrailsErrorResponse,
            "description": (
                "El campo `message` fue rechazado por Guardrails: "
                "off-topic, contenido tóxico o información personal (PII)."
            ),
        },
        500: {
            "model": GuardrailsErrorResponse,
            "description": "La respuesta del LLM no superó la validación de output.",
        },
    },
    summary="Chat conversacional con asistente financiero",
    description=(
        "Segundo paso del pipeline de chat: recibe el mensaje ya validado por "
        "`/llm/classify-context` junto con el contexto financiero del usuario y retorna "
        "una respuesta conversacional generada por el LLM.\n\n"
        "La validación Safety (ToxicLanguage + DetectPII) ocurre en `/llm/classify-context` "
        "antes de llegar aquí, evitando llamadas innecesarias al LLM si el mensaje es inválido. "
        "La respuesta pasa por validación **Guardrails OUTPUT** (ToxicLanguage) "
        "antes de retornarse.\n\n"
        "> **Nota de latencia:** La generación puede tomar entre 5 y 120 segundos "
        "según el modelo y hardware. Configura el cliente con `timeout ≥ 120s`."
    ),
)
async def chat(payload: ChatRequest):
    guardrails = get_guardrails_service()

    # Invocar el chain — JsonOutputParser retorna dict directamente
    chain = get_chat_chain_structured()

    categories_str = ", ".join([
        f"{c.name}: {c.amount} {payload.user_context.currency}"
        for c in payload.categories
    ])

    over_budget_list = [
        b.name for b in payload.budgets
        if b.spent > b.limit
    ]
    over_budget = ", ".join(over_budget_list) if over_budget_list else "Ninguno"

    try:
        parsed = chain.invoke({
            "message": payload.message,
            "financial_level": payload.user_context.financial_level,
            "total_income": payload.financial_summary.total_income,
            "total_expenses": payload.financial_summary.total_expenses,
            "savings_rate": payload.financial_summary.savings_rate,
            "currency": payload.user_context.currency,
            "categories": categories_str,
            "over_budget": over_budget,
            "current_date": date_type.today().isoformat(),
            "additional_context": payload.additional_context or "",
            "context_type": payload.context_type or "none",
        })
    except OutputParserException:
        parsed = {
            "intent": "question",
            "message": payload.message,
            "transaction_data": None,
        }
    except Exception as exc:
        logger.exception("LLM chain.invoke error in /chat")
        raise HTTPException(status_code=500, detail={
            "error": type(exc).__name__,
            "message": str(exc),
        })

    intent = parsed.get("intent", "question")
    message_content = parsed.get("message", "")

    # 3. Routing por intención
    if intent == "off_topic":
        return ChatResponse(
            message=message_content or (
                "Solo puedo ayudarte con temas de finanzas personales como "
                "presupuesto, ahorro, gastos e ingresos. ¿Tienes alguna consulta financiera?"
            ),
            metadata={"confidence": 1.0, "type": "off_topic"},
            transaction_action=None,
        )

    transaction_action = None

    if intent == "transaction" and parsed.get("transaction_data"):
        td = parsed["transaction_data"]
        try:
            transaction_action = TransactionAction(
                name=str(td["name"]),
                type=str(td["type"]),
                amount=float(td["amount"]),
                date=str(td["date"]),
                category_name=str(td.get("category_name") or "Otros"),
            )
        except (KeyError, ValueError, TypeError):
            transaction_action = None

    # 4. Validar OUTPUT
    try:
        message_content = await guardrails.async_validate_output(message_content)
    except GuardrailsValidationError as e:
        raise HTTPException(status_code=500, detail={
            "error": e.error_type,
            "message": e.message,
        })

    return ChatResponse(
        message=message_content,
        metadata={
            "confidence": 0.85,
            "type": "transaction" if transaction_action else "conversational",
        },
        transaction_action=transaction_action,
    )
