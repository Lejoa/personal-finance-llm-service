import os

from langchain_core.language_models.chat_models import BaseChatModel


def get_llm_provider() -> BaseChatModel:
    """
    Factory that returns the configured LLM provider.

    Supported providers (set via LLM_PROVIDER env var):
    - "ollama-cloud" (default): Ollama Cloud via OpenAI-compatible API.
      Requires OLLAMA_API_KEY and LLM_MODEL (e.g. gpt-oss:120b).
    - "ollama": local Ollama instance running in Docker.
      Requires OLLAMA_BASE_URL (default: http://ollama:11434) and LLM_MODEL.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama-cloud").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("LLM_MODEL", "gemma2:2b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        )

    elif provider == "ollama-cloud":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OLLAMA_API_KEY", "")
        if not api_key:
            raise ValueError("OLLAMA_API_KEY is required for ollama-cloud provider")

        # langchain_openai.ChatOpenAI is reused here because Ollama Cloud exposes
        # an OpenAI-compatible API endpoint — no OpenAI account or key is involved.
        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-oss:120b"),
            base_url="https://ollama.com/v1",
            api_key=api_key,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            # Cap output tokens to avoid unnecessary latency: 3 sentences + JSON
            # fit well within 300 tokens. Without this limit the model can generate
            # 200+ extra tokens at ~21 tok/s, adding up to 10s of avoidable wait.
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "300")),
        )

    else:
        raise ValueError(
            f"LLM_PROVIDER '{provider}' is not supported. "
            "Use 'ollama-cloud' or 'ollama'."
        )