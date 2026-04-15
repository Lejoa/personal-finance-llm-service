#!/usr/bin/env bash
# run_smoke_comparison.sh
# Suite reducida (≈25% de los casos del run_comparison.sh completo) para
# obtener datos de referencia rápidos sin arriesgar que el test caiga por
# tiempo o recursos antes de ver resultados.
#
# Casos seleccionados (1 representativo por categoría/tipo):
#   Guardrails  (5/20):  G1(topic-off), G2(topic-off), G3(on_topic), T1(toxic), P1(pii)
#   Clasificador(7/18):  CL1(trends), CL4(budget), CL7(categories), CL10(savings),
#                        CL13(transaction), CL15(question), CL17(none)
#   Quality base(2/8):   Q5(question), Q3(savings)
#   Quality enriq(2/5):  CE1(trends), CE3(budget)
#
# Archivos de casos de prueba leídos:
#   tests/test_cases.json
#     └─ Fuente de casos Guardrail (categoría "guardrail").
#        Leído por eval_models_smoke.py → eval_models.py (load_test_cases).
#        Estructura: { "test_cases": [ { "id": "G1", "category": "guardrail",
#                      "subcategory": "off_topic|on_topic",
#                      "guardrail_type": "topic|toxic|pii", ... } ] }
#
#   tests/classifier_test_cases.json
#     └─ Fuente de casos del clasificador de contexto (CL1–CL18).
#        Leído por eval_models_smoke.py → eval_models.py (load_classifier_test_cases).
#        Estructura: { "test_cases": [ { "id": "CL1", "expected_context": "trends", ... } ] }
#
#   tests/quality_test_cases.json
#     └─ Fuente de casos de calidad base (Q1–Q8), sin additional_context.
#        Leído por test_llm_quality_smoke.py → test_llm_quality.py (load_test_cases).
#        Estructura: { "financial_context": {...}, "test_cases": [ { "id": "Q1", ... } ] }
#
#   tests/context_enriched_test_cases.json
#     └─ Fuente de casos de calidad enriquecidos (CE1–CE5), con additional_context.
#        Leído por test_llm_quality_smoke.py → test_llm_quality.py (load_enriched_test_cases).
#        Estructura: { "financial_context": {...}, "test_cases": [ { "id": "CE1", ... } ] }
#
# Uso (desde la raíz del proyecto llm-service):
#   bash tests/run_smoke_comparison.sh                      # todos los modelos
#   bash tests/run_smoke_comparison.sh --model gpt-oss:120b # un solo modelo
#   bash tests/run_smoke_comparison.sh --skip-restart       # no reiniciar servicio
#
# Requisitos:
#   - llm-service corriendo: docker compose up -d
#   - Puerto 8000 mapeado al host (configurado en docker-compose.yaml)
#   - OLLAMA_API_KEY en .env para quality tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yaml"
COMPOSE_TEST_FILE="$PROJECT_DIR/docker-compose.test.yaml"

# IDs del subconjunto (~25%)
GUARDRAIL_IDS="G1,G2,G3,T1,P1"
CLASSIFIER_IDS="CL1,CL4,CL7,CL10,CL13,CL15,CL17"
QUALITY_BASE_IDS="Q5,Q3"
QUALITY_ENRICHED_IDS="CE1,CE3"

# Modelos a evaluar (por defecto todos)
# NOTA: Solo modelos con tag :cloud funcionan con LLM_PROVIDER=ollama-cloud.
# Lista completa en: https://ollama.com/search?c=cloud
DEFAULT_MODELS=(
  "gemini-3-flash-preview:cloud"
  "deepseek-v3.2:cloud"
  "gemma4:31b-cloud"
)

# URL del servicio desde el HOST (puerto mapeado, no IP interna Docker)
HOST_SERVICE_URL="http://localhost:8000"

# Parseo de argumentos
SINGLE_MODEL=""
SKIP_RESTART=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)        SINGLE_MODEL="$2"; shift 2 ;;
    --skip-restart) SKIP_RESTART=true; shift ;;
    --help|-h)
      echo "Uso: $0 [--model NOMBRE] [--skip-restart]"
      echo "  --model NOMBRE     Evalúa solo ese modelo (ej: gpt-oss:120b)"
      echo "  --skip-restart     No reinicia llm-service entre modelos"
      exit 0 ;;
    *) echo "Opción desconocida: $1. Usa --help para ver opciones."; exit 1 ;;
  esac
done

if [[ -n "$SINGLE_MODEL" ]]; then
  MODELS=("$SINGLE_MODEL")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

export JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss:120b}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Espera a que llm-service responda en localhost:8000 (puerto mapeado al host)
wait_for_service() {
  local retries=30
  local delay=5
  log "Esperando que llm-service responda en $HOST_SERVICE_URL ..."
  for i in $(seq 1 $retries); do
    if curl -sf --max-time 3 "$HOST_SERVICE_URL/health" > /dev/null 2>&1; then
      log "Servicio listo."
      return 0
    fi
    log "  Intento $i/$retries — reintentando en ${delay}s"
    sleep "$delay"
  done
  log "ERROR: El servicio no respondió en $HOST_SERVICE_URL después de $((retries * delay))s"
  log "  Verifica que llm-service esté corriendo: docker compose up -d"
  return 1
}

# Verifica que el servicio esté levantado antes de empezar
if ! curl -sf --max-time 5 "$HOST_SERVICE_URL/health" > /dev/null 2>&1; then
  log "ERROR: llm-service no responde en $HOST_SERVICE_URL"
  log "  Levanta el servicio primero: cd $PROJECT_DIR && docker compose up -d"
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p tests/results

TOTAL_CASES=$(( $(echo "$GUARDRAIL_IDS" | tr ',' '\n' | wc -l) + \
                $(echo "$CLASSIFIER_IDS" | tr ',' '\n' | wc -l) + \
                $(echo "$QUALITY_BASE_IDS" | tr ',' '\n' | wc -l) + \
                $(echo "$QUALITY_ENRICHED_IDS" | tr ',' '\n' | wc -l) ))

log "========================================================"
log "  SMOKE COMPARISON — $TOTAL_CASES tests por modelo (≈25% del total)"
log "  Guardrail:   $GUARDRAIL_IDS"
log "  Classifier:  $CLASSIFIER_IDS"
log "  Quality:     base=$QUALITY_BASE_IDS  enriched=$QUALITY_ENRICHED_IDS"
log "  Modelos:     ${MODELS[*]}"
log "  Juez:        $JUDGE_MODEL"
log "  Servicio:    $HOST_SERVICE_URL"
log "========================================================"
echo ""

for MODEL in "${MODELS[@]}"; do
  log "========================================"
  log "Evaluando modelo: $MODEL"
  log "========================================"

  export LLM_MODEL="$MODEL"

  # Reiniciar servicio con el nuevo modelo (a menos que se pase --skip-restart)
  if [[ "$SKIP_RESTART" == false ]]; then
    log "Reiniciando llm-service con LLM_MODEL=$MODEL ..."
    docker compose -f "$COMPOSE_FILE" stop llm-service
    LLM_MODEL="$MODEL" docker compose -f "$COMPOSE_FILE" up -d llm-service
    wait_for_service || { log "Saltando modelo $MODEL."; continue; }
  else
    log "(--skip-restart) Usando servicio ya levantado."
  fi

  # 0. Smoke test LLM desde el host (localhost:8000)
  log "--- [0/2] Smoke test ---"
  SMOKE_OUT=$(curl -sf --max-time 120 "$HOST_SERVICE_URL/llm/smoke-test" 2>/tmp/smoke_err.txt || true)
  if [[ -z "$SMOKE_OUT" ]]; then
    log "SMOKE TEST FALLÓ para $MODEL — $(cat /tmp/smoke_err.txt 2>/dev/null)"
    log "Saltando modelo $MODEL — LLM no disponible."
    continue
  fi
  SMOKE_RESP=$(echo "$SMOKE_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model_response','').strip())" 2>/dev/null || echo "?")
  log "Smoke test OK → $SMOKE_RESP"

  # 1. Guardrail + Classifier (subconjunto 25%)
  # Los contenedores de test se comunican con llm-service por nombre DNS interno
  log "--- [1/2] Guardrail + Classifier (smoke subset) ---"
  docker compose -f "$COMPOSE_TEST_FILE" run --rm \
    -e LLM_MODEL="$MODEL" \
    llm-tests \
    python tests/eval_models_smoke.py \
      --base-url http://llm-service:8000 \
      --model "$MODEL" \
      --guardrail-ids "$GUARDRAIL_IDS" \
      --classifier-ids "$CLASSIFIER_IDS" \
    || log "WARN: guardrail/classifier smoke eval finalizó con errores para $MODEL"

  # 2. Quality (subconjunto 25%)
  log "--- [2/2] Quality (smoke subset) ---"
  docker compose -f "$COMPOSE_TEST_FILE" --profile test-quality run --rm \
    -e LLM_MODEL="$MODEL" \
    -e JUDGE_MODEL="$JUDGE_MODEL" \
    llm-quality-tests \
    python tests/test_llm_quality_smoke.py \
      --base-url http://llm-service:8000 \
      --base-ids "$QUALITY_BASE_IDS" \
      --enriched-ids "$QUALITY_ENRICHED_IDS" \
    || log "WARN: quality smoke eval finalizó con errores para $MODEL"

  log "Modelo $MODEL completado."
  echo ""
done

log "========================================"
log "Smoke comparison completado."
log "Resultados en tests/results/ (prefijo smoke_)"
log "========================================"
