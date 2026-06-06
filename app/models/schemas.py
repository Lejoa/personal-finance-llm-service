from pydantic import BaseModel, Field
from typing import List, Optional


class UserContext(BaseModel):
    """Perfil y configuración regional del usuario."""

    currency: str = Field(
        description="Código de moneda ISO 4217.",
        examples=["COP"],
    )
    locale: str = Field(
        description="Locale del usuario en formato BCP 47.",
        examples=["es-CO"],
    )
    financial_level: str = Field(
        description="Nivel de conocimiento financiero del usuario.",
        examples=["beginner"],
    )


class Summary(BaseModel):
    """Resumen financiero del período."""

    period: str = Field(
        description="Período en formato YYYY-MM.",
        examples=["2026-02"],
    )
    total_income: float = Field(
        description="Ingresos totales del período en la moneda del usuario.",
        examples=[5000000.0],
    )
    total_expenses: float = Field(
        description="Gastos totales del período en la moneda del usuario.",
        examples=[3200000.0],
    )
    savings_rate: float = Field(
        description="Tasa de ahorro expresada como porcentaje (0–100).",
        examples=[36.0],
    )
    previous_savings_rate: Optional[float] = Field(
        default=None,
        description="Tasa de ahorro del mes anterior como porcentaje (0–100). Null si no hay datos.",
        examples=[28.0],
    )
    previous_income: Optional[float] = Field(
        default=None,
        description="Ingresos totales del mes anterior en la moneda del usuario. Null si no hay datos.",
        examples=[4800000.0],
    )
    previous_expenses: Optional[float] = Field(
        default=None,
        description="Gastos totales del mes anterior en la moneda del usuario. Null si no hay datos.",
        examples=[3100000.0],
    )


class Category(BaseModel):
    """Categoría de gasto con su monto total en el período."""

    name: str = Field(description="Nombre de la categoría.", examples=["Comida"])
    amount: float = Field(
        description="Monto total gastado en esta categoría.",
        examples=[1200000.0],
    )


class Budget(BaseModel):
    """Presupuesto asignado a una categoría y su estado de ejecución."""

    name: str = Field(description="Nombre del presupuesto.", examples=["Comida"])
    limit: float = Field(
        description="Límite presupuestado para la categoría.",
        examples=[1000000.0],
    )
    spent: float = Field(
        description="Monto gastado hasta el momento. Si spent > limit el presupuesto está excedido.",
        examples=[1200000.0],
    )


class FinancialInsightsRequest(BaseModel):
    """Datos necesarios para generar insights educativos financieros."""

    user_context: UserContext
    summary: Summary
    categories: List[Category] = Field(
        description="Lista de categorías de gasto del período."
    )
    budgets: Optional[List[Budget]] = Field(
        default=[],
        description="Lista de presupuestos por categoría. Opcional.",
    )
    top_tip: Optional[str] = Field(
        default=None,
        description="Consejo más relevante para el perfil del usuario. Formato: 'Título: descripción corta'.",
        examples=["Fondo de emergencia: Guarda 3 meses de gastos para imprevistos"],
    )
    goal: str = Field(
        description=(
            "Objetivo o consulta financiera del usuario. "
            "Debe ser de temática financiera (validado por Guardrails)."
        ),
        examples=["Quiero entender cómo mejorar mi tasa de ahorro"],
    )


class Insight(BaseModel):
    """Insight educativo generado por el LLM."""

    type: str = Field(
        description="Tipo de insight.",
        examples=["education"],
    )
    message: str = Field(
        description="Texto del insight generado por el LLM.",
        examples=["Con un ahorro del 36% estás por encima del promedio recomendado..."],
    )


class FinancialInsightsResponse(BaseModel):
    """Respuesta del endpoint de insights financieros."""

    insights: List[Insight] = Field(description="Lista de insights generados.")
    confidence: float = Field(
        description="Nivel de confianza de la respuesta (0.0–1.0).",
        examples=[0.85],
    )


class ClassifyContextRequest(BaseModel):
    """Datos para clasificar el tipo de contexto financiero que necesita un mensaje."""

    message: str = Field(
        description="Mensaje del usuario a clasificar.",
        examples=["¿Cómo va mi presupuesto este mes?"],
    )
    available_categories: Optional[List[str]] = Field(
        default=[],
        description=(
            "Lista de nombres de categorías disponibles en la app. "
            "Permite al clasificador normalizar el nombre de categoría en period_hint.category "
            "usando el nombre exacto de la BD en lugar del sinónimo del usuario."
        ),
        examples=[["Comida", "Transporte", "Entretenimiento"]],
    )


class PeriodHint(BaseModel):
    """Período temporal extraído del mensaje del usuario por el clasificador."""

    from_month: Optional[str] = Field(
        default=None,
        description="Primer mes del rango en formato YYYY-MM.",
        examples=["2026-04"],
    )
    to_month: Optional[str] = Field(
        default=None,
        description="Último mes del rango en formato YYYY-MM.",
        examples=["2026-05"],
    )
    category: Optional[str] = Field(
        default=None,
        description="Nombre de la categoría mencionada por el usuario, si aplica.",
        examples=["Comida", "Transporte"],
    )


class ClassifyContextResponse(BaseModel):
    """Resultado de la clasificación de contexto."""

    context_type: str = Field(
        description="Tipo de contexto: transaction|question|trends|historical|budget|categories|savings|none.",
        examples=["historical"],
    )
    period_hint: Optional[PeriodHint] = Field(
        default=None,
        description=(
            "Período temporal extraído del mensaje. "
            "Presente solo cuando context_type es 'historical' o 'trends'. "
            "Null para todos los demás tipos."
        ),
    )


class ConversationTurn(BaseModel):
    """Un turno de la conversación (mensaje de usuario o respuesta del asistente)."""

    role: str = Field(
        description="Rol del emisor: 'user' o 'assistant'.",
        examples=["user", "assistant"],
    )
    content: str = Field(
        description="Contenido del mensaje.",
        examples=["¿Cuánto gasté en Comida?"],
    )


class ChatRequest(BaseModel):
    """Datos necesarios para el chat conversacional con contexto financiero."""

    message: str = Field(
        description=(
            "Mensaje del usuario. Debe ser de temática financiera "
            "(validado por Guardrails)."
        ),
        examples=["¿Cómo puedo reducir mis gastos en Comida?"],
    )
    user_context: UserContext
    financial_summary: Summary
    categories: List[Category] = Field(
        description="Lista de categorías de gasto con sus montos."
    )
    budgets: Optional[List[Budget]] = Field(
        default=[],
        description="Lista de presupuestos por categoría. Opcional.",
    )
    additional_context: Optional[str] = Field(
        default="",
        description=(
            "Contexto financiero adicional inyectado condicionalmente según el tipo de pregunta. "
            "Vacío para transacciones y preguntas generales."
        ),
        examples=["Detalle de presupuestos: Comida: 80% usado, quedan 12 días"],
    )
    context_type: Optional[str] = Field(
        default="none",
        description=(
            "Tipo de consulta pre-clasificado: trends, budget, categories, savings, none. "
            "Orientación para el LLM sobre la intención del mensaje."
        ),
        examples=["budget"],
    )
    conversation_history: Optional[List[ConversationTurn]] = Field(
        default=[],
        description=(
            "Últimos N turnos de la conversación (mensajes anteriores). "
            "Permite al LLM mantener coherencia en conversaciones multi-turno. "
            "Cada elemento contiene 'role' (user|assistant) y 'content'."
        ),
        examples=[[
            {"role": "user", "content": "¿Cuánto gasté en Comida este mes?"},
            {"role": "assistant", "content": "Llevas 450.000 COP en Comida este mes."},
        ]],
    )
    available_categories: Optional[List[str]] = Field(
        default=[],
        description=(
            "Lista completa de nombres de categorías disponibles en la app. "
            "Permite al LLM mapear sinónimos del usuario (ej: 'alimentación' → 'Comida') "
            "a nombres reales de categoría al registrar transacciones o consultar historial."
        ),
        examples=[["Comida", "Transporte", "Entretenimiento", "Salud", "Vivienda"]],
    )


class TransactionAction(BaseModel):
    """Datos de una transacción detectada en el mensaje del usuario"""

    name: str = Field(
        description="Descripción breve del concepto de la transacción.",
        examples=["Hamburguesa, Salida a Cine"],
    )
    type: str = Field(
        description="Tipo de transacción: gasto o ingreso.",
        examples=["gasto", "ingreso"]
    )
    amount: float = Field(
        description="Monto de la transacción como número positivo.",
        examples=[50000.0],
    )
    date: str = Field(
        description="Fecha de la transacción en formato YYYY-MM-DD.",
        examples=["2026-03-19"],
    )
    category_name: Optional[str] = Field(
        default=None,
        description="Nombre de la categoría del presupuesto que mejor aplica. 'Otros' si ninguna aplica.",
        examples=["Comida", "Entretenimiento", "Otros"], 
    )

class ChatResponse(BaseModel):
    """Respuesta del asistente financiero conversacional."""

    message: str = Field(
        description="Respuesta generada por el LLM.",
        examples=["Veo que llevas 1,200,000 COP en Comida este mes..."],
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Metadatos adicionales de la respuesta.",
        examples=[{"confidence": 0.85, "type": "conversational"}],
    )
    transaction_action: Optional[TransactionAction] = Field( 
        default=None,
        description="Datos de transacción detectada en el mensaje del usuario. Null si es una pregunta.",
    )


class RagSearchRequest(BaseModel):
    """Parámetros para la búsqueda semántica RAG."""

    query: str = Field(
        description="Texto de la pregunta del usuario a buscar semánticamente.",
        examples=["Quiero aprender más sobre ahorro e inversión con propósito"],
    )
    limit: Optional[int] = Field(
        default=3,
        description="Número máximo de chunks a recuperar (1-10).",
        examples=[3],
    )


class RagChunk(BaseModel):
    """Un fragmento de texto recuperado en la búsqueda semántica."""

    content: str = Field(description="Texto del fragmento recuperado.")
    tip_title: Optional[str] = Field(default=None, description="Título del tip al que pertenece el chunk.")
    source_title: Optional[str] = Field(default=None, description="Título de la fuente bibliográfica. None si el chunk viene del tip directamente.")
    source_author: Optional[str] = Field(default=None, description="Autor de la fuente bibliográfica.")
    similarity: float = Field(description="Similitud coseno con la consulta (0-1).")


class RagSearchResponse(BaseModel):
    """Resultado de la búsqueda semántica RAG."""

    results: List[RagChunk] = Field(description="Fragmentos más similares a la consulta.")


class EmbedRequest(BaseModel):
    """Texto a convertir en vector de embedding."""

    text: str = Field(
        description="Texto a embeber.",
        examples=["Ahorra e invierte con propósito"],
    )


class EmbedResponse(BaseModel):
    """Vector de embedding generado por OpenAI text-embedding-3-small."""

    embedding: List[float] = Field(
        description="Vector de 1536 dimensiones.",
    )
    model: str = Field(
        description="Modelo utilizado para generar el embedding.",
        examples=["text-embedding-3-small"],
    )


class GuardrailsErrorResponse(BaseModel):
    """Respuesta de error cuando Guardrails rechaza el input o el output."""

    error: str = Field(
        description="Tipo de error de validación.",
        examples=["off_topic"],
    )
    error_type: str = Field(
        description="Categoría del error: off_topic, toxic, pii, unknown, output_invalid.",
        examples=["off_topic"],
    )
    message: str = Field(
        description="Mensaje legible para el usuario explicando el rechazo.",
        examples=["Lo siento, solo puedo ayudarte con preguntas sobre finanzas personales."],
    )