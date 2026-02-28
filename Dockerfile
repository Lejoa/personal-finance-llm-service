FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG GUARDRAILS_API_KEY
RUN guardrails configure --token ${GUARDRAILS_API_KEY} --disable-metrics --disable-remote-inferencing
RUN guardrails hub install hub://tryolabs/restricttotopic --quiet
RUN guardrails hub install hub://guardrails/toxic_language --quiet
RUN guardrails hub install hub://guardrails/detect_pii --quiet

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
