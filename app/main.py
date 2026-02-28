from fastapi import FastAPI

from app.api.financial import router as financial_router
from app.api.health import router as health_router

app = FastAPI(
    title="LLM Financial Service",
    version="0.1.0",
    description=(
        "Servicio LLM para asistencia financiera personalizada. "
        "Expone endpoints para generar insights educativos y chat conversacional "
        "usando modelos de lenguaje (Ollama / OpenAI) con LangChain.\n\n"
        "Todos los endpoints del grupo **LLM** pasan por una capa de validación "
        "**Guardrails-AI** que verifica el input antes de enviarlo al modelo "
        "y el output antes de retornarlo al cliente."
    ),
    contact={
        "name": "Personal Finance App",
    },
    license_info={
        "name": "MIT",
    },
)

app.include_router(health_router)
app.include_router(financial_router)