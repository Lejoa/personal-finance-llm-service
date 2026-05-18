# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
# git is required to install guardrails-ai directly from GitHub (PyPI quarantined 2026-05-11)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Validators are installed at build time using a BuildKit secret.
# The token is available only during this RUN step — it is never written
# to any image layer. ~/.guardrailsrc is deleted at the end of the same RUN
# to ensure guardrails configure does not leave the token on disk.
# At runtime, app/main.py reads GUARDRAILS_API_KEY from the environment
# and runs "guardrails configure" again to authenticate the session.
RUN --mount=type=secret,id=guardrails_token \
    GRTOKEN=$(cat /run/secrets/guardrails_token) && \
    guardrails configure \
      --token "$GRTOKEN" \
      --disable-metrics \
      --disable-remote-inferencing && \
    guardrails hub install hub://tryolabs/restricttotopic --quiet && \
    guardrails hub install hub://guardrails/toxic_language --quiet && \
    guardrails hub install hub://guardrails/detect_pii --quiet && \
    rm -f ~/.guardrailsrc

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
