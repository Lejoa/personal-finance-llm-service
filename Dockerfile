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

# GUARDRAILS HUB VALIDATORS — TEMPORARILY DISABLED (2026-05-18)
# The Guardrails Hub registry removed these packages from PyPI on 2026-05-11
# (same supply chain incident that quarantined guardrails-ai itself).
# hub install hub://tryolabs/restricttotopic, hub://guardrails/toxic_language,
# and hub://guardrails/detect_pii all return 404/no matching distribution.
# Re-enable this block once the Hub registry is restored:
#
# RUN --mount=type=secret,id=guardrails_token \
#     GRTOKEN=$(cat /run/secrets/guardrails_token) && \
#     guardrails configure \
#       --token "$GRTOKEN" \
#       --disable-metrics \
#       --disable-remote-inferencing && \
#     guardrails hub install hub://tryolabs/restricttotopic --quiet && \
#     guardrails hub install hub://guardrails/toxic_language --quiet && \
#     guardrails hub install hub://guardrails/detect_pii --quiet && \
#     rm -f ~/.guardrailsrc

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
