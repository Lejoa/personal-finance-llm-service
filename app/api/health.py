import os

import httpx
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Estado del servicio",
    description="Verifica que el servicio FastAPI está activo y respondiendo.",
)
def health_check():
    return {"status": "ok", "message": "LLM service is running"}


@router.get(
    "/health/llm",
    summary="Estado del proveedor LLM",
    description=(
        "Verifica la conectividad con el proveedor LLM configurado "
        "en la variable de entorno `LLM_PROVIDER`.\n\n"
        "- **ollama / ollama-cloud**: hace una petición a `/api/tags` y lista los modelos disponibles.\n"
        "- **openai**: verifica que `OPENAI_API_KEY` esté configurada.\n\n"
        "Siempre retorna HTTP 200. El campo `status` indica el resultado real: "
        "`ok`, `warning` o `error`."
    ),
)
async def llm_health_check():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider in ("ollama", "ollama-cloud"):
        return await _check_ollama(provider)

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        return {
            "status": "ok" if api_key else "warning",
            "provider": "openai",
            "api_key_configured": bool(api_key),
        }

    return {"status": "unknown", "provider": provider}


async def _check_ollama(provider: str) -> dict:
    """Verifica conexión con Ollama local o Cloud."""
    if provider == "ollama-cloud":
        url = "https://ollama.com"
        api_key = os.getenv("OLLAMA_API_KEY", "")
        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = 10.0
    else:
        url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        headers = {}
        timeout = 5.0

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{url}/api/tags", headers=headers, timeout=timeout
            )
            models = response.json().get("models", [])
            return {
                "status": "ok",
                "provider": provider,
                "url": url,
                "models_available": len(models),
                "model_names": [m["name"] for m in models],
            }
    except Exception as e:
        return {
            "status": "error",
            "provider": provider,
            "error": str(e),
        }