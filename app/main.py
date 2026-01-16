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

@app.get("/debug/routes")
def debug_routes():
    return [route.path for route in app.routes]


app.include_router(financial_router)
