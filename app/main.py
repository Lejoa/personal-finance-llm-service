from fastapi import FastAPI

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

@app.get("/hello")
def hello():
    return {
        "message": "Hola mundo desde el servicio LLM 🚀"
    }
