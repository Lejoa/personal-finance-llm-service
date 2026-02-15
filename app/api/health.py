import os

import httpx
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    return {"status": "ok", "message": "LLM service is running"}


@router.get("/health/llm")
async def llm_health_check():
    """
    Health check que verifica la conexión con el proveedor LLM.
    """
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