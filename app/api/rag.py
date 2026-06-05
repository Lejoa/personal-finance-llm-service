import os
import logging

import httpx
from fastapi import APIRouter, HTTPException

from app.api.embedding import get_openai_client, EMBEDDING_MODEL
from app.models.schemas import RagSearchRequest, RagSearchResponse, RagChunk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["RAG"])


def _get_backend_url() -> str:
    return os.getenv("BACKEND_BASE_URL", "http://php:80")


def _get_internal_token() -> str:
    return os.getenv("INTERNAL_API_TOKEN", "")


@router.post(
    "/rag/search",
    response_model=RagSearchResponse,
    summary="Búsqueda semántica RAG sobre tip_embeddings",
    description=(
        "Convierte la consulta en un vector de embedding (OpenAI text-embedding-3-small), "
        "llama al endpoint interno del backend PHP para ejecutar la búsqueda pgvector, "
        "y retorna los chunks más similares semánticamente con su fuente y similitud. "
        "Usado internamente por el pipeline /llm/chat cuando context_type es 'education'."
    ),
)
def rag_search(payload: RagSearchRequest) -> RagSearchResponse:
    # A — generar embedding del query
    try:
        client = get_openai_client()
        result = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=payload.query,
        )
        query_vector = result.data[0].embedding
    except Exception as exc:
        logger.exception("Error generating query embedding for RAG search")
        raise HTTPException(status_code=500, detail={
            "error": "EmbeddingError",
            "message": str(exc),
        })

    # B — llamar al backend PHP para ejecutar la búsqueda pgvector
    limit = max(1, min(10, payload.limit or 3))
    try:
        response = httpx.post(
            f"{_get_backend_url()}/internal/rag/search",
            json={"embedding": query_vector, "limit": limit},
            headers={"X-Internal-Token": _get_internal_token()},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("Backend RAG search returned %s", exc.response.status_code)
        raise HTTPException(status_code=502, detail={
            "error": "BackendError",
            "message": f"Backend returned {exc.response.status_code}",
        })
    except Exception as exc:
        logger.exception("Error calling backend RAG search")
        raise HTTPException(status_code=502, detail={
            "error": "BackendError",
            "message": str(exc),
        })

    chunks = [
        RagChunk(
            content=item["content"],
            source_title=item.get("source_title"),
            source_author=item.get("source_author"),
            similarity=float(item.get("similarity", 0.0)),
        )
        for item in data.get("results", [])
    ]

    return RagSearchResponse(results=chunks)
