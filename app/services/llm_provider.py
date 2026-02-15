import os

from langchain_core.language_models.chat_models import BaseChatModel


def get_llm_provider() -> BaseChatModel:
    """
    Factory function que retorna el proveedor LLM configurado.
    Soporta OpenAI, Ollama local y Ollama Cloud.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("LLM_MODEL", "gemma2:2b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        )

    elif provider == "ollama-cloud":
        from langchain_ollama import ChatOllama

        api_key = os.getenv("OLLAMA_API_KEY", "")
        if not api_key:
            raise ValueError("OLLAMA_API_KEY is required for ollama-cloud provider")

        return ChatOllama(
            model=os.getenv("LLM_MODEL", "gemma2:2b"),
            base_url="https://ollama.com",
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
            client_kwargs={
                "headers": {"Authorization": f"Bearer {api_key}"}
            },
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        )

    else:
        raise ValueError(
            f"LLM_PROVIDER '{provider}' no soportado. "
            "Usa 'openai', 'ollama' o 'ollama-cloud'."
        )