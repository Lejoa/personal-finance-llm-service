# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
# git is required to install guardrails-ai directly from GitHub (PyPI quarantined 2026-05-11)
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

RUN --mount=type=secret,id=guardrails_token \
    GRTOKEN=$(cat /run/secrets/guardrails_token) && \
    guardrails configure \
      --token "$GRTOKEN" \
      --disable-metrics \
      --disable-remote-inferencing && \
    guardrails hub install hub://tryolabs/restricttotopic --quiet && \
    guardrails hub install hub://guardrails/toxic_language --quiet && \
    guardrails hub install hub://guardrails/detect_pii --quiet && \
    printf 'enable_metrics=false\nuse_remote_inferencing=false\n' > ~/.guardrailsrc

# Pre-download the detoxify multilingual model (~500 MB) so it is baked into
# the image and the service does not need to fetch it on every cold start.
RUN python -c "import detoxify, torch; detoxify.Detoxify('multilingual', device=torch.device('cpu'))"

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
