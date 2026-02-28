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