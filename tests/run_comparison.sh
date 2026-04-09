#!/usr/bin/env bash
# run_comparison.sh
# Ejecuta los tres suites de test (guardrails, quality, multi-turn) para cada
# modelo financiero de Ollama Cloud y genera un reporte comparativo.
#
# Uso:
#   bash tests/run_comparison.sh
#
# Requisitos:
#   - Docker Compose instalado y daemon corriendo
#   - Imagen llm-service construida: docker compose build
#   - OLLAMA_API_KEY exportada o definida en .env
#   - JUDGE_MODEL exportada o usa el default gpt-oss:120b-cloud
#
# Los resultados individuales se guardan en tests/results/
# El reporte comparativo final en tests/results/comparison_report.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODELS=(
  "gemini-3-flash-preview"
  "gpt-oss:120b"
  "qwen3.5:397b"
)

# El juez permanece fijo para comparativa justa
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss:120b}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_service() {
  local url="$1"
  local retries=20
  local delay=5
  log "Esperando servicio en $url ..."
  for i in $(seq 1 $retries); do
    if curl -sf "$url/health" > /dev/null 2>&1; then
      log "Servicio listo."
      return 0
    fi
    log "  Intento $i/$retries — reintentando en ${delay}s"
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
  docker compose stop llm-service
  docker compose up -d llm-service
  wait_for_service "http://localhost:8000"

  # 0. Smoke test — abortar si el modelo no responde
  log "--- [0/3] Smoke test ---"
  SMOKE_STATUS=$(curl -s -o /tmp/smoke_response.json -w "%{http_code}" \
    "http://localhost:8000/llm/smoke-test")
  if [ "$SMOKE_STATUS" != "200" ]; then
    log "SMOKE TEST FALLÓ para $MODEL (HTTP $SMOKE_STATUS)"
    cat /tmp/smoke_response.json || true
    log "Saltando modelo $MODEL — no está disponible."
    continue
  fi
  log "Smoke test OK → $(cat /tmp/smoke_response.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('model_response',''))")"

  # 1. Guardrail evaluation
  log "--- [1/3] Guardrail evaluation ---"
  docker compose -f docker-compose.test.yaml run --rm \
    -e LLM_MODEL="$MODEL" \
    llm-tests \
    python tests/eval_models.py \
      --base-url http://llm-service:8000 \
      --model "$MODEL" || log "WARN: guardrail eval finalizó con errores para $MODEL"

  # 2. Quality evaluation
  log "--- [2/3] Quality evaluation ---"
  docker compose -f docker-compose.test.yaml --profile test-quality run --rm \
    -e LLM_MODEL="$MODEL" \
    -e JUDGE_MODEL="$JUDGE_MODEL" \
    llm-quality-tests || log "WARN: quality eval finalizó con errores para $MODEL"

  # 3. Multi-turn evaluation (deshabilitado temporalmente)
  # log "--- [3/3] Multi-turn evaluation ---"
  # docker compose -f docker-compose.test.yaml --profile test-multi-turn run --rm \
  #   -e LLM_MODEL="$MODEL" \
  #   -e JUDGE_MODEL="$JUDGE_MODEL" \
  #   llm-multi-turn-tests || log "WARN: multi-turn eval finalizó con errores para $MODEL"

  log "Modelo $MODEL completado."
  echo ""
done

log "========================================"
log "Todos los modelos evaluados."
log "Generando reporte comparativo..."
log "========================================"

python3 "$SCRIPT_DIR/compare_results.py"

log "Reporte guardado en tests/results/comparison_report.json"