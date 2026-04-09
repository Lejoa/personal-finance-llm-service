"""
Evaluación de calidad de respuestas del LLM con deepEval.
Usa Ollama Cloud (gpt-oss:120b-cloud) como modelo juez vía DeepEvalBaseLLM.

Métricas evaluadas:
  - AnswerRelevancy:      ¿la respuesta es relevante al input?
  - Faithfulness:         ¿la respuesta se basa en el contexto sin inventar datos?
  - ContextualRelevancy:  ¿el contexto proporcionado es relevante para el input?
  - ContextualRecall:     ¿la respuesta cubre los puntos clave del contexto?
  - ContextualPrecision:  ¿los fragmentos del contexto usados son los correctos?

Prerrequisitos:
  - OLLAMA_API_KEY, LLM_MODEL (modelo bajo prueba) y JUDGE_MODEL (juez DeepEval, default gpt-oss:120b-cloud) definidas en .env
  - Servicio llm-service corriendo: docker compose up -d

Uso con Docker (recomendado):
  1. docker compose up -d
  2. docker compose -f docker-compose.test.yaml --profile test-quality up \\
       --abort-on-container-exit llm-quality-tests

Los resultados se guardan en tests/results/quality_<modelo>_<timestamp>.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_openai import ChatOpenAI

SCRIPT_DIR = Path(__file__).parent
QUALITY_TEST_CASES_PATH = SCRIPT_DIR / "quality_test_cases.json"
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 300.0

METRIC_THRESHOLD = 0.7

METRIC_LABELS = {
    "answer_relevancy": "AnswerRelevancy",
    "faithfulness": "Faithfulness",
    "contextual_relevancy": "ContextualRelevancy",
    "contextual_recall": "ContextualRecall",
    "contextual_precision": "ContextualPrecision",
}


# ---------------------------------------------------------------------------
# Juez: Ollama Cloud
# ---------------------------------------------------------------------------

class OllamaCloudJudge(DeepEvalBaseLLM):
    """Juez deepEval usando Ollama Cloud como backend."""

    def __init__(self):
        self._model = ChatOpenAI(
            model=os.getenv("JUDGE_MODEL", "gpt-oss:120b"),
            api_key=os.getenv("OLLAMA_API_KEY"),
            base_url="https://ollama.com/v1",
        )

    def load_model(self):
        return self._model

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.load_model().ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:
        return f"ollama-cloud/{os.getenv('JUDGE_MODEL', 'gpt-oss:120b')}"


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_test_cases() -> dict:
    with open(QUALITY_TEST_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Llamada al servicio
# ---------------------------------------------------------------------------

def get_llm_response(
    client: httpx.Client,
    base_url: str,
    financial_context: dict,
    user_message: str,
) -> tuple[str | None, float]:
    """Llama a /llm/chat y retorna (respuesta, tiempo_ms). Retorna (None, ms) si falla."""
    payload = {"message": user_message, **financial_context}
    start = time.time()
    try:
        response = client.post(
            f"{base_url}/llm/chat", json=payload, timeout=REQUEST_TIMEOUT
        )
        elapsed_ms = (time.time() - start) * 1000
        response.raise_for_status()
        return response.json()["message"], round(elapsed_ms, 2)
    except httpx.TimeoutException:
        elapsed_ms = (time.time() - start) * 1000
        return None, round(elapsed_ms, 2)
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return None, round(elapsed_ms, 2)


# ---------------------------------------------------------------------------
# Construcción de métricas
# ---------------------------------------------------------------------------

def build_metrics(metric_keys: list[str], judge: OllamaCloudJudge) -> list:
    """Instancia solo las métricas declaradas en el test case."""
    builders = {
        "answer_relevancy": lambda: AnswerRelevancyMetric(
            threshold=METRIC_THRESHOLD, model=judge
        ),
        "faithfulness": lambda: FaithfulnessMetric(
            threshold=METRIC_THRESHOLD, model=judge
        ),
        "contextual_relevancy": lambda: ContextualRelevancyMetric(
            threshold=METRIC_THRESHOLD, model=judge
        ),
        "contextual_recall": lambda: ContextualRecallMetric(
            threshold=METRIC_THRESHOLD, model=judge
        ),
        "contextual_precision": lambda: ContextualPrecisionMetric(
            threshold=METRIC_THRESHOLD, model=judge
        ),
    }
    return [builders[k]() for k in metric_keys if k in builders]


# ---------------------------------------------------------------------------
# Ejecución de un test case
# ---------------------------------------------------------------------------

def run_single_test(
    client: httpx.Client,
    base_url: str,
    financial_context: dict,
    test: dict,
    judge: OllamaCloudJudge,
) -> dict:
    preview = test["input"][:60]
    print(f"  [{test['id']}] {preview}...", end=" ", flush=True)

    actual_output, elapsed_ms = get_llm_response(
        client, base_url, financial_context, test["input"]
    )

    if actual_output is None:
        print(f"ERROR ({elapsed_ms:.0f}ms)")
        return {
            "id": test["id"],
            "input": test["input"],
            "description": test.get("description", ""),
            "metrics_evaluated": test["metrics"],
            "response_time_ms": elapsed_ms,
            "error": "No se obtuvo respuesta del servicio",
            "metric_results": {},
            "passed": False,
        }

    test_case = LLMTestCase(
        input=test["input"],
        actual_output=actual_output,
        expected_output=test.get("expected_output"),
        retrieval_context=test.get("retrieval_context", []),
    )

    metrics = build_metrics(test["metrics"], judge)
    metric_results = {}
    all_passed = True

    async def measure_metric(metric):
        t0 = time.time()
        try:
            await metric.a_measure(test_case)
            judge_ms = round((time.time() - t0) * 1000, 2)
            passed = metric.score >= METRIC_THRESHOLD
            return metric.__class__.__name__, {
                "score": round(metric.score, 4),
                "threshold": METRIC_THRESHOLD,
                "passed": passed,
                "reason": metric.reason,
                "judge_time_ms": judge_ms,
            }
        except Exception as e:
            judge_ms = round((time.time() - t0) * 1000, 2)
            return metric.__class__.__name__, {
                "score": None,
                "passed": False,
                "error": str(e),
                "judge_time_ms": judge_ms,
            }

    async def run_all_metrics():
        return await asyncio.gather(*[measure_metric(m) for m in metrics])

    judge_start = time.time()
    results_list = asyncio.run(run_all_metrics())
    total_judge_ms = round((time.time() - judge_start) * 1000, 2)

    for name, result in results_list:
        metric_results[name] = result
        if not result.get("passed", False):
            all_passed = False

    status = "PASS" if all_passed else "FAIL"
    print(f"{status} (llm={elapsed_ms:.0f}ms judge={total_judge_ms:.0f}ms)")

    return {
        "id": test["id"],
        "input": test["input"],
        "actual_output": actual_output,
        "description": test.get("description", ""),
        "metrics_evaluated": test["metrics"],
        "llm_time_ms": elapsed_ms,
        "judge_time_ms": total_judge_ms,
        "total_time_ms": round(elapsed_ms + total_judge_ms, 2),
        "passed": all_passed,
        "metric_results": metric_results,
    }


# ---------------------------------------------------------------------------
# Cálculo de scores agregados
# ---------------------------------------------------------------------------

def calculate_scores(results: list, wall_time_ms: float = 0.0) -> dict:
    # Todos los casos, incluyendo errores, cuentan para total y passed
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    errors = sum(1 for r in results if "error" in r)

    llm_times = [r.get("llm_time_ms", r.get("response_time_ms", 0)) for r in results]
    judge_times = [r.get("judge_time_ms", 0) for r in results]
    total_times = [r.get("total_time_ms", llm_times[i]) for i, r in enumerate(results)]
    times_sorted = sorted(llm_times)

    timing = {}
    if llm_times:
        p95_idx = min(int(len(times_sorted) * 0.95), len(times_sorted) - 1)
        timing = {
            "llm": {
                "avg_ms": round(sum(llm_times) / len(llm_times), 2),
                "min_ms": round(min(llm_times), 2),
                "max_ms": round(max(llm_times), 2),
                "p95_ms": round(times_sorted[p95_idx], 2),
            },
            "judge": {
                "avg_ms": round(sum(judge_times) / len(judge_times), 2),
                "min_ms": round(min(judge_times), 2),
                "max_ms": round(max(judge_times), 2),
                "total_ms": round(sum(judge_times), 2),
            },
            "total": {
                "avg_ms": round(sum(total_times) / len(total_times), 2),
                "wall_time_ms": round(wall_time_ms, 2),
            },
            # Retrocompatibilidad con compare_results.py
            "avg_ms": round(sum(llm_times) / len(llm_times), 2),
            "min_ms": round(min(llm_times), 2),
            "max_ms": round(max(llm_times), 2),
            "p95_ms": round(times_sorted[p95_idx], 2),
        }

    # Para métricas individuales solo se consideran los casos que sí obtuvieron respuesta
    valid = [r for r in results if "error" not in r]

    # Score por cada métrica individual
    per_metric: dict[str, dict] = {}
    for key, label in METRIC_LABELS.items():
        class_name = label + "Metric"
        scores_for_metric = []
        passes_for_metric = 0
        count_for_metric = 0
        for r in valid:
            mr = r.get("metric_results", {})
            if class_name in mr and mr[class_name].get("score") is not None:
                count_for_metric += 1
                scores_for_metric.append(mr[class_name]["score"])
                if mr[class_name].get("passed", False):
                    passes_for_metric += 1
        if count_for_metric:
            per_metric[key] = {
                "label": label,
                "evaluated": count_for_metric,
                "passed": passes_for_metric,
                "avg_score": round(sum(scores_for_metric) / count_for_metric, 4),
                "pass_rate_pct": round(passes_for_metric / count_for_metric * 100, 1),
            }

    return {
        "timing": timing,
        "overall": {
            "total": total,
            "passed": passed,
            "errors": errors,
            "score_pct": round(passed / total * 100, 1) if total else 0,
        },
        "per_metric": per_metric,
    }


# ---------------------------------------------------------------------------
# Impresión del resumen
# ---------------------------------------------------------------------------

def print_summary(judge_name: str, scores: dict):
    print("\n" + "=" * 65)
    print(f"  RESUMEN DE CALIDAD — Juez: {judge_name}")
    print("=" * 65)

    timing = scores.get("timing", {})
    llm_t = timing.get("llm", timing)
    judge_t = timing.get("judge", {})
    total_t = timing.get("total", {})
    print("\n  Tiempos de respuesta — LLM bajo prueba:")
    print(f"    Promedio: {llm_t.get('avg_ms', 0) / 1000:.1f}s")
    print(f"    Mínimo:   {llm_t.get('min_ms', 0) / 1000:.1f}s")
    print(f"    Máximo:   {llm_t.get('max_ms', 0) / 1000:.1f}s")
    print(f"    P95:      {llm_t.get('p95_ms', 0) / 1000:.1f}s")
    if judge_t:
        print(f"\n  Tiempos del juez (DeepEval, paralelo por test):")
        print(f"    Promedio por test: {judge_t.get('avg_ms', 0) / 1000:.1f}s")
        print(f"    Máximo por test:   {judge_t.get('max_ms', 0) / 1000:.1f}s")
        print(f"    Total acumulado:   {judge_t.get('total_ms', 0) / 1000:.1f}s")
    if total_t:
        print(f"\n  Tiempo total de la suite (wall clock): {total_t.get('wall_time_ms', 0) / 1000:.1f}s")

    ov = scores["overall"]
    print(f"\n  Resultado general: {ov['passed']}/{ov['total']} casos pasaron — {ov['score_pct']}%")
    if ov.get("errors", 0):
        print(f"  Sin respuesta del servicio: {ov['errors']} caso(s) contabilizados como FAIL")

    METRIC_DESCRIPTIONS = {
        "AnswerRelevancy": (
            "Evalúa qué tan relevante es la respuesta del LLM (actual_output) "
            "respecto al input del usuario. No requiere respuesta de referencia: "
            "el juez determina por sí solo si la respuesta aborda lo que se preguntó."
        ),
        "Faithfulness": (
            "Evalúa si la respuesta del LLM (actual_output) está factualmente alineada "
            "con el contexto recuperado (retrieval_context), sin inventar datos. "
            "Detecta alucinaciones o afirmaciones que no tienen soporte en el contexto."
        ),
        "ContextualPrecision": (
            "Evalúa si los fragmentos del retrieval_context que son relevantes para el input "
            "están posicionados por encima de los irrelevantes. Mide la calidad del ordenamiento "
            "del retriever: los nodos más útiles deben aparecer primero."
        ),
        "ContextualRecall": (
            "Evalúa en qué medida el retrieval_context cubre los puntos clave del expected_output. "
            "Un score bajo indica que el retriever no recuperó suficiente información para "
            "respaldar la respuesta esperada."
        ),
        "ContextualRelevancy": (
            "Evalúa la relevancia general de la información en el retrieval_context para el input. "
            "Mide si el retriever trajo fragmentos útiles o incluyó información innecesaria "
            "que no aporta a responder la pregunta."
        ),
    }

    print("\n  Desglose por métrica:")
    for _key, m in scores.get("per_metric", {}).items():
        desc = METRIC_DESCRIPTIONS.get(m["label"], "")
        print(
            f"\n    {m['label']:<25} "
            f"{m['passed']}/{m['evaluated']} pasaron "
            f"(avg score: {m['avg_score']:.2f}, pass rate: {m['pass_rate_pct']}%)"
        )
        if desc:
            # Imprimir descripción en líneas de máximo 60 chars con indentación
            words = desc.split()
            line, indent = "", "      "
            for word in words:
                if len(line) + len(word) + 1 > 60:
                    print(f"{indent}{line.strip()}")
                    line = word + " "
                else:
                    line += word + " "
            if line.strip():
                print(f"{indent}{line.strip()}")
    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# Guardado de resultados
# ---------------------------------------------------------------------------

def save_results(judge_name: str, results: list, scores: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = judge_name.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quality_{safe_name}_{timestamp}.json"
    filepath = RESULTS_DIR / filename

    output = {
        "judge": judge_name,
        "model": os.getenv("LLM_MODEL", "unknown"),
        "threshold": METRIC_THRESHOLD,
        "timestamp": datetime.now().isoformat(),
        "scores": scores,
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Resultados guardados en: {filepath}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluación de calidad de respuestas LLM con deepEval"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"URL base del servicio (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()

    data = load_test_cases()
    financial_context = data["financial_context"]
    test_cases = data["test_cases"]

    print(f"\n  Verificando servicio en {args.base_url}...")
    try:
        with httpx.Client() as client:
            health = client.get(f"{args.base_url}/health", timeout=5.0)
            health.raise_for_status()
            print(f"  Servicio activo: {health.json()}")
    except Exception as e:
        print(f"  ERROR: No se puede conectar al servicio: {e}")
        sys.exit(1)

    judge = OllamaCloudJudge()
    judge_name = judge.get_model_name()

    print(f"\n  Juez: {judge_name}")
    print(f"  Threshold: {METRIC_THRESHOLD}")
    print(f"  Test cases: {len(test_cases)}")
    print()

    results = []
    suite_start = time.time()
    with httpx.Client() as client:
        print(f"--- CALIDAD ({len(test_cases)} tests) ---")
        for test in test_cases:
            result = run_single_test(client, args.base_url, financial_context, test, judge)
            results.append(result)
        print()
    wall_time_ms = (time.time() - suite_start) * 1000

    scores = calculate_scores(results, wall_time_ms)
    print_summary(judge_name, scores)
    save_results(judge_name, results, scores)


if __name__ == "__main__":
    main()
