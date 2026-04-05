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