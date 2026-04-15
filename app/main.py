from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.financial import router as financial_router
from app.api.health import router as health_router
from app.services.chat_chain import get_chat_chain_structured
from app.services.context_classifier_chain import get_context_classifier_chain
from app.services.financial_chain import get_financial_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_chat_chain_structured()
    get_context_classifier_chain()
    get_financial_chain()
    yield


app = FastAPI(
    title="LLM Financial Service",
    lifespan=lifespan,
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