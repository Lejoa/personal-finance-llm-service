import os
import logging

from fastapi import APIRouter, HTTPException
from openai import OpenAI

from app.models.schemas import EmbedRequest, EmbedResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["Embedding"])

EMBEDDING_MODEL = "text-embedding-3-small"

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embedding generation")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Genera un embedding de texto usando OpenAI text-embedding-3-small",
    description=(
        "Convierte un texto en un vector de 1536 dimensiones usando el modelo "
        "text-embedding-3-small de OpenAI. Usado internamente por el backend PHP "
        "para indexar tips y referencias bibliográficas, y por el pipeline RAG "
        "para convertir la pregunta del usuario antes de la búsqueda de similitud."
    ),
)
def embed(payload: EmbedRequest) -> EmbedResponse:
    try:
        client = get_openai_client()
        result = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=payload.text,
        )
        return EmbedResponse(
            embedding=result.data[0].embedding,
            model=EMBEDDING_MODEL,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Error generating embedding")
        raise HTTPException(status_code=500, detail={
            "error": type(exc).__name__,
            "message": str(exc),
        })
