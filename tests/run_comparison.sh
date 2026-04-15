#!/usr/bin/env bash
# run_comparison.sh
# Ejecuta los dos suites de test (guardrails+classifier, quality) para cada
# modelo financiero de Ollama Cloud y genera un reporte comparativo.
#
# Archivos de casos de prueba leídos:
#   tests/test_cases.json
#     └─ Fuente de TODOS los casos Guardrail (20 casos, categoría "guardrail").
#        Leído por eval_models.py (load_test_cases).
#        Estructura: { "test_cases": [ { "id": "G1", "category": "guardrail",
#                      "subcategory": "off_topic|on_topic",
#                      "guardrail_type": "topic|toxic|pii", ... } ] }
#
#   tests/classifier_test_cases.json
#     └─ Fuente de TODOS los casos del clasificador de contexto (18 casos, CL1–CL18).
#        Leído por eval_models.py (load_classifier_test_cases).
#        Estructura: { "test_cases": [ { "id": "CL1", "expected_context": "trends", ... } ] }
#
#   tests/quality_test_cases.json
#     └─ Fuente de TODOS los casos de calidad base (8 casos, Q1–Q8), sin additional_context.
#        Leído por test_llm_quality.py (load_test_cases).
#        Estructura: { "financial_context": {...}, "test_cases": [ { "id": "Q1", ... } ] }
#
#   tests/context_enriched_test_cases.json
#     └─ Fuente de TODOS los casos de calidad enriquecidos (5 casos, CE1–CE5),
#        con additional_context para simular contexto financiero real.
#        Leído por test_llm_quality.py (load_enriched_test_cases).
#        Estructura: { "financial_context": {...}, "test_cases": [ { "id": "CE1", ... } ] }
#
# Uso (desde la raíz del proyecto llm-service):
#   bash tests/run_comparison.sh
#
# Requisitos:
#   - Docker Compose instalado y daemon corriendo
#   - Imagen llm-service construida: docker compose build
#   - OLLAMA_API_KEY exportada o definida en .env
#   - JUDGE_MODEL exportada o usa el default gpt-oss:120b
#
# Los resultados individuales se guardan en tests/results/
# El reporte comparativo final en tests/results/comparison_report.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yaml"
COMPOSE_TEST_FILE="$PROJECT_DIR/docker-compose.test.yaml"

MODELS=(
  "gemini-3-flash-preview"
  "gpt-oss:120b"
  "qwen3.5:397b"
)

# El juez permanece fijo para comparativa justa
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss:120b}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Obtiene la IP del contenedor llm-service en la red Docker
get_llm_ip() {
  docker inspect llm-service \
    --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 2>/dev/null \
    | head -1
}

wait_for_service() {
  local retries=30
  local delay=5
  log "Esperando que llm-service responda..."
  for i in $(seq 1 $retries); do
    local ip
    ip=$(get_llm_ip)
    if [ -n "$ip" ] && curl -sf "http://$ip:8000/health" > /dev/null 2>&1; then
      log "Servicio listo en http://$ip:8000"
      return 0
    fi
    log "  Intento $i/$retries — reintentando en ${delay}s (ip=$ip)"
    sleep "$delay"
  done
  log "ERROR: El servicio no respondió después de $((retries * delay))s"
  return 1
}

cd "$PROJECT_DIR"

mkdir -p tests/results

for MODEL in "${MODELS[@]}"; do
  log "========================================"
  log "Evaluando modelo: $MODEL"
  log "========================================"

  export LLM_MODEL="$MODEL"

  # Reiniciar el servicio con el nuevo modelo
  log "Reiniciando llm-service con LLM_MODEL=$MODEL ..."
  docker compose -f "$COMPOSE_FILE" stop llm-service
  docker compose -f "$COMPOSE_FILE" up -d llm-service
  wait_for_service

  # Obtener IP actual del contenedor para smoke test desde el host
  LLM_IP=$(get_llm_ip)

  # 0. Smoke test — abortar si el modelo no responde
  log "--- [0/2] Smoke test (http://$LLM_IP:8000) ---"
  SMOKE_STATUS=$(curl -s -o /tmp/smoke_response.json -w "%{http_code}" \
    --max-time 120 \
    "http://$LLM_IP:8000/llm/smoke-test")
  if [ "$SMOKE_STATUS" != "200" ]; then
    log "SMOKE TEST FALLÓ para $MODEL (HTTP $SMOKE_STATUS)"
    cat /tmp/smoke_response.json || true
    log "Saltando modelo $MODEL — no está disponible."
    continue
  fi
  log "Smoke test OK → $(python3 -c "import sys,json; print(json.load(open('/tmp/smoke_response.json')).get('model_response',''))")"

  # 1. Guardrail + Classifier evaluation
  log "--- [1/2] Guardrail + Classifier evaluation ---"
  docker compose -f "$COMPOSE_TEST_FILE" run --rm \
    -e LLM_MODEL="$MODEL" \
    llm-tests \
    python tests/eval_models.py \
      --base-url http://llm-service:8000 \
      --model "$MODEL" || log "WARN: guardrail/classifier eval finalizó con errores para $MODEL"

  # 2. Quality evaluation
  log "--- [2/2] Quality evaluation ---"
  docker compose -f "$COMPOSE_TEST_FILE" --profile test-quality run --rm \
    -e LLM_MODEL="$MODEL" \
    -e JUDGE_MODEL="$JUDGE_MODEL" \
    llm-quality-tests || log "WARN: quality eval finalizó con errores para $MODEL"

  log "Modelo $MODEL completado."
  echo ""
done

log "========================================"
log "Todos los modelos evaluados."
log "Generando reporte comparativo..."
log "========================================"

python3 "$SCRIPT_DIR/compare_results.py"

log "Reporte guardado en tests/results/comparison_report.json"
