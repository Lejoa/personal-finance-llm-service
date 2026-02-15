import os

import httpx
from fastapi import FastAPI

from app.api.financial import router as financial_router

app = FastAPI(
    title="LLM Financial Service",
    version="0.1.0"
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "LLM service is running"
    }


@app.get("/health/llm")
async def llm_health_check():
    """
    Health check que verifica la conexión con el proveedor LLM.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "ollama":
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{ollama_url}/api/tags", timeout=5.0)
                models = response.json().get("models", [])
                return {
                    "status": "ok",
                    "provider": "ollama",
                    "url": ollama_url,
                    "models_available": len(models),
                    "model_names": [m["name"] for m in models]
                }
        except Exception as e:
            return {
                "status": "error",
                "provider": "ollama",
                "error": str(e)
            }

    elif provider == "ollama-cloud":
        api_key = os.getenv("OLLAMA_API_KEY", "")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://ollama.com/api/tags",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10.0
                )
                models = response.json().get("models", [])
                return {
                    "status": "ok",
                    "provider": "ollama-cloud",
                    "url": "https://ollama.com",
                    "api_key_configured": bool(api_key),
                    "models_available": len(models),
                    "model_names": [m["name"] for m in models]
                }
        except Exception as e:
            return {
                "status": "error",
                "provider": "ollama-cloud",
                "error": str(e)
            }

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        return {
            "status": "ok" if api_key else "warning",
            "provider": "openai",
            "api_key_configured": bool(api_key)
        }

    return {"status": "unknown", "provider": provider}


@app.get("/debug/routes")
def debug_routes():
    return [route.path for route in app.routes]


app.include_router(financial_router)
