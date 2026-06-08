#!/usr/bin/env bash
# run_comparison.sh
# Ejecuta los dos suites de test (guardrails+classifier, quality) para cada
# modelo financiero de Ollama Cloud y genera un reporte comparativo.
#
# Archivos de casos de prueba leídos:
#   tests/test_cases/guard_rails_cases.json
#     └─ Fuente de TODOS los casos Guardrail (10 casos: 5 ToxicLanguage + 5 DetectPII).
#        Leído por eval_topics.py (load_test_cases).
#        Estructura: { "test_cases": [ { "id": "G1", "category": "guardrail",
#                      "subcategory": "off_topic|on_topic",
#                      "guardrail_type": "topic|toxic|pii", ... } ] }
#
#   tests/test_cases/classifier_test_cases.json
#     └─ Fuente de TODOS los casos del clasificador de contexto (18 casos, CL1–CL18).
#        Leído por eval_topics.py (load_classifier_test_cases).
#        Estructura: { "test_cases": [ { "id": "CL1", "expected_context": "trends", ... } ] }
#
#   tests/test_cases/quality_test_cases.json
#     └─ Fuente de TODOS los casos de calidad base (10 casos, sin additional_context).
#        context_types: question, savings, budget, categories, transaction.
#        Leído por test_llm_quality.py (load_test_cases).
#        Estructura: { "financial_context": {...}, "test_cases": [ { "id": "Q1", ... } ] }
#
#   tests/test_cases/context_enriched_test_cases.json
#     └─ Fuente de TODOS los casos de calidad enriquecidos (7 casos, con additional_context).
#        context_types: trends, budget, categories, savings, historical.
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
# Los resultados individuales se guardan en tests/results/full_tests/
# El reporte comparativo final en tests/results/full_tests/comparison_report.json

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

mkdir -p tests/results/full_tests

for MODEL in "${MODELS[@]}"; do
  log "========================================"
  log "Evaluando modelo: $MODEL"
  log "========================================"

  export LLM_MODEL="$MODEL"

  # Reiniciar el servicio con el nuevo modelo
  log "Reiniciando llm-service con LLM_MODEL=$MODEL ..."
  docker ps -q --filter "publish=8000" | xargs -r docker rm -f
  docker rm -f llm-service 2>/dev/null || true
  LLM_MODEL="$MODEL" docker compose -f "$COMPOSE_FILE" up -d llm-service
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
    python tests/eval_topics.py \
      --base-url http://llm-service:8000 \
      --model "$MODEL" \
      --results-dir /app/tests/results/full_tests \
    || log "WARN: guardrail/classifier eval finalizó con errores para $MODEL"

  # 2. Quality evaluation
  log "--- [2/2] Quality evaluation ---"
  docker compose -f "$COMPOSE_TEST_FILE" --profile test-quality run --rm \
    -e LLM_MODEL="$MODEL" \
    -e JUDGE_MODEL="$JUDGE_MODEL" \
    llm-quality-tests \
    python tests/test_llm_quality.py \
      --base-url http://llm-service:8000 \
      --results-dir /app/tests/results/full_tests \
    || log "WARN: quality eval finalizó con errores para $MODEL"

  log "Modelo $MODEL completado."
  echo ""
done

log "========================================"
log "Todos los modelos evaluados."
log "Generando reporte comparativo..."
log "========================================"

python3 "$SCRIPT_DIR/compare_results.py" \
  --results-dir "$PROJECT_DIR/tests/results/full_tests" \
  --output "$PROJECT_DIR/tests/results/full_tests/comparison_report.json"

log "Reporte guardado en tests/results/full_tests/comparison_report.json"
