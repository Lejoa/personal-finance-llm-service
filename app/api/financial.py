import logging
from datetime import date as date_type

from fastapi import APIRouter, HTTPException
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, AIMessage
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
    PeriodHint,
)
from app.services.financial_chain import get_financial_chain
from app.services.chat_chain import get_chat_chain_structured
from app.services.rag_chain import get_rag_education_chain
from app.services.context_classifier_chain import get_context_classifier_chain, VALID_CONTEXT_TYPES
from app.services.guardrails_service import get_guardrails_service, GuardrailsValidationError
from app.services.llm_provider import get_llm_provider
from app.api.rag import rag_search
from app.models.schemas import RagSearchRequest

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
    summary="Genera insights educativos financieros",
    description=(
        "Recibe el resumen financiero del usuario y su objetivo (`goal`), "
        "y retorna un insight educativo generado por el LLM.\n\n"
        "El campo `goal` proviene de un formulario financiero controlado en la app; "
        "no requiere validación de Guardrails.\n\n"
        "> **Nota de latencia:** La generación puede tomar entre 5 y 120 segundos "
        "según el modelo y hardware. Configura el cliente con `timeout ≥ 120s`."
    ),
)
def get_financial_insights(payload: FinancialInsightsRequest):
    # 1. Invocar el LLM
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
        parsed = chain.invoke({
            "message": payload.message,
            "current_date": date_type.today().isoformat(),
            "available_categories": ", ".join(payload.available_categories or []),
        })
        context_type = parsed.get("context_type", "none")
        if context_type not in VALID_CONTEXT_TYPES:
            context_type = "none"

        # Construir period_hint solo si el LLM lo incluyó con datos mínimos
        period_hint = None
        raw_hint = parsed.get("period_hint")
        if isinstance(raw_hint, dict) and raw_hint.get("from_month") and raw_hint.get("to_month"):
            period_hint = PeriodHint(
                from_month=str(raw_hint["from_month"]),
                to_month=str(raw_hint["to_month"]),
                category=raw_hint.get("category"),
            )

    except Exception:
        context_type = "none"
        period_hint = None

    return ClassifyContextResponse(context_type=context_type, period_hint=period_hint)


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

    # Rama RAG: cuando context_type es "education" se ejecutan los pasos A→B→C
    # en lugar del chain de chat financiero estándar.
    if payload.context_type == "education":
        return await _handle_education(payload, guardrails)

    # Invocar el chain — JsonOutputParser retorna dict directamente
    chain = get_chat_chain_structured()

    categories_str = ", ".join([
        f"{c.name}: {c.amount} {payload.user_context.currency}"
        for c in payload.categories
    ])

    cur = payload.user_context.currency
    over_budget_list = [
        (
            f"{b.name} "
            f"(límite {b.limit} {cur}, "
            f"gastado {b.spent} {cur}, "
            f"exceso {b.spent - b.limit} {cur})"
        )
        for b in payload.budgets
        if b.spent > b.limit
    ]
    over_budget = "; ".join(over_budget_list) if over_budget_list else "Ninguno"

    budgets_detail_list = []
    for b in payload.budgets:
        excess = b.spent - b.limit
        if excess > 0:
            budgets_detail_list.append(
                f"{b.name}: límite {b.limit} {cur}, "
                f"gastado {b.spent} {cur}, "
                f"excedido en {excess} {cur}"
            )
        else:
            budgets_detail_list.append(
                f"{b.name}: límite {b.limit} {cur}, "
                f"gastado {b.spent} {cur}, "
                f"disponible {b.limit - b.spent} {cur}"
            )
    budgets_detail = "; ".join(budgets_detail_list) if budgets_detail_list else "Sin presupuestos configurados"

    history_messages = []
    for turn in (payload.conversation_history or []):
        if turn.role == "user":
            history_messages.append(HumanMessage(content=turn.content))
        elif turn.role == "assistant":
            history_messages.append(AIMessage(content=turn.content))

    previous_income = payload.financial_summary.previous_income
    previous_expenses = payload.financial_summary.previous_expenses
    previous_savings_rate = payload.financial_summary.previous_savings_rate

    available_categories_str = ", ".join(payload.available_categories or [])

    try:
        parsed = chain.invoke({
            "message": payload.message,
            "financial_level": payload.user_context.financial_level,
            "total_income": payload.financial_summary.total_income,
            "total_expenses": payload.financial_summary.total_expenses,
            "savings_rate": payload.financial_summary.savings_rate,
            "previous_income": previous_income if previous_income is not None else "sin datos",
            "previous_expenses": previous_expenses if previous_expenses is not None else "sin datos",
            "previous_savings_rate": previous_savings_rate if previous_savings_rate is not None else "sin datos",
            "currency": payload.user_context.currency,
            "categories": categories_str,
            "over_budget": over_budget,
            "budgets_detail": budgets_detail,
            "current_date": date_type.today().isoformat(),
            "additional_context": payload.additional_context or "",
            "context_type": payload.context_type or "none",
            "history": history_messages,
            "available_categories": available_categories_str,
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


async def _handle_education(payload: ChatRequest, guardrails) -> ChatResponse:
    """
    Pipeline RAG para mensajes de tipo 'education'.

    A — Genera el embedding de la pregunta del usuario via /llm/rag/search,
        que internamente llama al backend PHP para ejecutar la búsqueda pgvector.
    B — Construye el rag_context con los chunks recuperados y sus fuentes.
    C — Invoca el chain RAG con rag_education.txt para generar la respuesta.

    Si no hay chunks indexados (corpus vacío o error de red), cae al chain de
    chat estándar con context_type "education" para que el LLM responda con
    conocimiento general (comportamiento de Fase 1).
    """
    rag_context = ""
    rag_sources: list[str] = []

    try:
        search_result = rag_search(RagSearchRequest(query=payload.message, limit=3))
        if search_result.results:
            parts = []
            seen: set[str] = set()
            for chunk in search_result.results:
                parts.append(chunk.content)
                label = chunk.source_title or chunk.tip_title
                if label and label not in seen:
                    seen.add(label)
                    rag_sources.append(label)
            rag_context = "\n\n".join(parts)
    except Exception:
        logger.warning("RAG search failed for education query — falling back to general knowledge")

    if rag_context:
        # C — responder con el material recuperado
        try:
            rag_chain = get_rag_education_chain()
            message_content = rag_chain.invoke({
                "question": payload.message,
                "rag_context": rag_context,
            })
            if hasattr(message_content, "content"):
                message_content = message_content.content

            message_content = str(message_content)
            if rag_sources:
                message_content += "\n\n**Fuentes:** " + " · ".join(rag_sources)

            try:
                message_content = await guardrails.async_validate_output(message_content)
            except GuardrailsValidationError as e:
                raise HTTPException(status_code=500, detail={
                    "error": e.error_type,
                    "message": e.message,
                })

            return ChatResponse(
                message=message_content,
                metadata={"confidence": 0.90, "type": "education_rag"},
                transaction_action=None,
            )
        except HTTPException:
            raise
        except Exception:
            logger.exception("RAG chain failed — falling back to general knowledge")

    # Fallback: sin chunks disponibles, usar el chain de chat con conocimiento general
    chain = get_chat_chain_structured()
    try:
        parsed = chain.invoke({
            "message": payload.message,
            "financial_level": payload.user_context.financial_level,
            "total_income": payload.financial_summary.total_income,
            "total_expenses": payload.financial_summary.total_expenses,
            "savings_rate": payload.financial_summary.savings_rate,
            "previous_income": payload.financial_summary.previous_income or "sin datos",
            "previous_expenses": payload.financial_summary.previous_expenses or "sin datos",
            "previous_savings_rate": payload.financial_summary.previous_savings_rate or "sin datos",
            "currency": payload.user_context.currency,
            "categories": "",
            "over_budget": "Ninguno",
            "budgets_detail": "Sin presupuestos configurados",
            "current_date": date_type.today().isoformat(),
            "additional_context": "",
            "context_type": "education",
            "history": [],
            "available_categories": "",
        })
    except Exception as exc:
        logger.exception("Education fallback chain failed")
        raise HTTPException(status_code=500, detail={
            "error": type(exc).__name__,
            "message": str(exc),
        })

    message_content = parsed.get("message", "")
    try:
        message_content = await guardrails.async_validate_output(message_content)
    except GuardrailsValidationError as e:
        raise HTTPException(status_code=500, detail={
            "error": e.error_type,
            "message": e.message,
        })

    return ChatResponse(
        message=message_content,
        metadata={"confidence": 0.75, "type": "education_general"},
        transaction_action=None,
    )
