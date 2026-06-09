"""
Evaluación de calidad de respuestas del LLM con DeepEval.
Usa Ollama Cloud como modelo juez vía DeepEvalBaseLLM (configurado por $JUDGE_MODEL).

Métricas evaluadas (umbral: 0.7):
  - AnswerRelevancy:      ¿la respuesta es relevante al input?
  - Faithfulness:         ¿la respuesta se basa en el contexto sin inventar datos?
  - ContextualRelevancy:  ¿el contexto proporcionado es relevante para el input?
  - ContextualRecall:     ¿la respuesta cubre los puntos clave del contexto?
  - ContextualPrecision:  ¿los fragmentos del contexto usados son los correctos?

Variables de entorno requeridas:
  OLLAMA_API_KEY   Clave de acceso a Ollama Cloud
  LLM_MODEL        Modelo bajo prueba (ej: gemini-3-flash-preview:cloud)
  JUDGE_MODEL      Modelo juez DeepEval (default: gpt-oss:120b)

Argumentos CLI:
  --base-url      URL del servicio LLM (default: $BASE_URL o http://localhost:8000)
  --suite         Suite a ejecutar: 'base' | 'enriched' | 'all' (default: all)
  --base-ids      IDs de la suite base separados por coma (default: todos)
  --enriched-ids  IDs de la suite enriquecida separados por coma (default: todos)
  --results-dir   Directorio de salida (default: tests/results/)
  --results-tag   'full' (default) o 'smoke' — controla el prefijo del archivo de resultados

Archivos de salida:
  full:  {results-dir}/quality_{judge}_base_{ts}.json
         {results-dir}/quality_{judge}_enriched_{ts}.json
  smoke: {results-dir}/smoke_quality_base_{judge}_{ts}.json
         {results-dir}/smoke_quality_enriched_{judge}_{ts}.json

Uso directo (suite completa):
  docker compose -f docker-compose.test.yaml --profile test-quality run --rm llm-quality-tests \\
    python tests/test_llm_quality.py \\
      --base-url http://llm-service:8000 \\
      --results-dir /app/tests/results/full_tests

Uso con subconjunto smoke:
  docker compose -f docker-compose.test.yaml --profile test-quality run --rm llm-quality-tests \\
    python tests/test_llm_quality.py \\
      --base-url http://llm-service:8000 \\
      --base-ids Q5,Q3,Q21 --enriched-ids CE1,CE3,CE6 \\
      --results-dir /app/tests/results/smoke_tests \\
      --results-tag smoke

Orquestado por:
  tests/run_comparison.sh       → suite completa, guarda en tests/results/full_tests/
  tests/run_smoke_comparison.sh → subconjunto ~25%, guarda en tests/results/smoke_tests/
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
QUALITY_TEST_CASES_PATH = SCRIPT_DIR / "test_cases" / "quality_test_cases.json"
ENRICHED_TEST_CASES_PATH = SCRIPT_DIR / "test_cases" / "context_enriched_test_cases.json"
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


def load_enriched_test_cases() -> dict:
    with open(ENRICHED_TEST_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Formateador de additional_context
# Replica exactamente la lógica de FinancialContextService.php del backend
# ---------------------------------------------------------------------------

def _fmt(amount: float) -> str:
    """Formatea un número con puntos como separador de miles (formato COP)."""
    return f"{int(amount):,}".replace(",", ".")


def format_additional_context(context_type: str, data: dict) -> str:
    """
    Construye el string additional_context a partir de datos estructurados,
    replicando los métodos del backend PHP: formatTrendsContext,
    formatBudgetDetailContext, formatCategoriesRankingContext, formatSavingsContext
    (FinancialContextService) y buildContext (HistoricalFinancialQueryService).
    """
    if context_type == "trends":
        prev_income = data.get("previous_income", 0)
        prev_expenses = data.get("previous_expenses", 0)
        prev_savings_rate = data.get("previous_savings_rate", 0)
        trends = data.get("category_trends", [])
        budget_health = data.get("budget_health", [])

        prev_savings_str = f"{prev_savings_rate:.1f}".replace(".", ",")
        lines = [
            "Resumen del mes pasado:",
            f"- Ingresos: ${_fmt(prev_income)} COP",
            f"- Gastos: ${_fmt(prev_expenses)} COP",
            f"- Tasa de ahorro: {prev_savings_str}%",
        ]

        if trends:
            lines.append("")
            lines.append("Comparación por categoría (promedio 3 meses vs. este mes):")
            for t in trends:
                delta = f"+{t['delta_pct']}%" if t["delta_pct"] >= 0 else f"{t['delta_pct']}%"
                lines.append(
                    f"- {t['name']}: promedio ${_fmt(t['avg_3_months'])} COP,"
                    f" este mes ${_fmt(t['current_month'])} COP ({delta})"
                )

        if budget_health:
            lines.append("")
            lines.append("Estado actual de presupuestos (mes en curso):")
            for b in budget_health:
                spent = b.get("spent", 0)
                limit = b.get("limit", 0)
                lines.append(
                    f"- {b['name']}: {b['pct_used']}% usado"
                    f" (${_fmt(spent)} de ${_fmt(limit)} COP),"
                    f" quedan {b['days_remaining']} días"
                )

        return "\n".join(lines)

    if context_type == "budget":
        health = data.get("budget_health", [])
        if not health:
            return ""
        parts = [
            f"{b['name']}: {b['pct_used']}% usado, quedan {b['days_remaining']} días"
            for b in health
        ]
        return "Detalle de presupuestos: " + "; ".join(parts)

    if context_type == "categories":
        trends = data.get("category_trends", [])
        if not trends:
            return ""
        lines = ["Ranking de gastos por categoría este mes:"]
        for i, t in enumerate(trends, start=1):
            lines.append(
                f"- {i}. {t['name']}: ${_fmt(t['current_month'])} COP"
                f" (promedio 3 meses: ${_fmt(t['avg_3_months'])} COP)"
            )
        return "\n".join(lines)

    if context_type == "savings":
        velocity  = data.get("spending_velocity", 0)
        projected = data.get("projected_expenses", 0)
        prev_rate = data.get("previous_savings_rate", 0)
        return (
            f"Proyección de ahorro: Velocidad de gasto ${_fmt(velocity)} COP/día. "
            f"Proyección fin de mes: ${_fmt(projected)} COP en gastos. "
            f"Tasa de ahorro mes anterior: {prev_rate}%."
        )

    if context_type == "historical":
        period = data.get("period", "")
        hist_type = data.get("type", "totals")

        if hist_type == "category":
            category = data.get("category_name", "")
            rows = data.get("monthly_data", [])
            if not rows:
                return ""
            total = sum(r["amount"] for r in rows)
            lines = [f"Datos históricos — {category} ({period}):"]
            for row in rows:
                lines.append(f"- {row['month']}: ${_fmt(row['amount'])} COP")
            lines.append(f"Total en el período: ${_fmt(total)} COP")
            return "\n".join(lines)

        # totals (default) — replica buildTotalsContext
        rows = data.get("monthly_totals", [])
        if not rows:
            return ""
        lines = [f"Datos históricos ({period}):"]
        total_income = 0.0
        total_expenses = 0.0
        for row in rows:
            income = row["income"]
            expenses = row["expenses"]
            savings_rate = row["savings_rate"]
            total_income += income
            total_expenses += expenses
            savings_str = f"{savings_rate:.1f}".replace(".", ",")
            lines.append(
                f"- {row['month']}: ingresos ${_fmt(income)} COP"
                f" | gastos ${_fmt(expenses)} COP"
                f" | ahorro {savings_str}%"
            )

        total_savings = (
            round(((total_income - total_expenses) / total_income) * 100, 1)
            if total_income > 0 else 0.0
        )
        total_savings_str = f"{total_savings:.1f}".replace(".", ",")
        lines.append(
            f"Total período: ingresos ${_fmt(total_income)} COP"
            f" | gastos ${_fmt(total_expenses)} COP"
            f" | ahorro {total_savings_str}%"
        )

        categories = data.get("categories", [])
        if categories:
            lines.append("")
            lines.append(f"Desglose de gastos por categoría ({period}):")
            for cat in categories:
                lines.append(f"- {cat['name']}: ${_fmt(cat['amount'])} COP")

        return "\n".join(lines)

    return ""


# ---------------------------------------------------------------------------
# Llamada al servicio
# ---------------------------------------------------------------------------

def get_llm_response(
    client: httpx.Client,
    base_url: str,
    financial_context: dict,
    user_message: str,
    additional_context: str = "",
    context_type: str = "",
) -> tuple[str | None, float]:
    """
    Replica el flujo de producción: classify-context → chat.

    Si context_type ya está provisto (suite enriquecida), lo usa directamente
    y omite el classify-context para ahorrar tiempo.
    Si no, llama primero a /llm/classify-context para obtenerlo.
    Retorna (respuesta, tiempo_ms_total). Retorna (None, ms) si falla.
    """
    start = time.time()

    # Paso 1 — classify-context (solo si context_type no viene del test case)
    resolved_context_type = context_type
    if not resolved_context_type:
        try:
            classify_resp = client.post(
                f"{base_url}/llm/classify-context",
                json={"message": user_message},
                timeout=30.0,
            )
            if classify_resp.status_code == 422:
                # Safety guard bloqueó — no hay respuesta de chat esperada en quality tests
                elapsed_ms = (time.time() - start) * 1000
                return None, round(elapsed_ms, 2)
            resolved_context_type = classify_resp.json().get("context_type", "none")
        except Exception:
            resolved_context_type = "none"

    # Paso 2 — chat
    payload = {"message": user_message, **financial_context}
    if additional_context:
        payload["additional_context"] = additional_context
    if resolved_context_type:
        payload["context_type"] = resolved_context_type

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
    except Exception:
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

    # Construir additional_context si el test lo declara (suite enriquecida)
    context_type = test.get("context_type", "")
    additional_context = ""
    if "additional_context_data" in test:
        additional_context = format_additional_context(
            context_type,
            test["additional_context_data"],
        )

    actual_output, elapsed_ms = get_llm_response(
        client, base_url, financial_context, test["input"], additional_context, context_type
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
        "expected_output": test.get("expected_output", ""),
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

def save_results(judge_name: str, results: list, scores: dict, suite_label: str = "", results_dir: Path = None, suite_tag: str = "full"):
    out_dir = results_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = judge_name.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suite_tag == "smoke":
        filename = f"smoke_quality_{suite_label}_{safe_name}_{timestamp}.json"
        suite_field = f"smoke_quality_{suite_label}"
    else:
        label_part = f"_{suite_label}" if suite_label else ""
        filename = f"quality_{safe_name}{label_part}_{timestamp}.json"
        suite_field = None
    filepath = out_dir / filename

    output = {
        "judge": judge_name,
        "model": os.getenv("LLM_MODEL", "unknown"),
        "threshold": METRIC_THRESHOLD,
        "timestamp": datetime.now().isoformat(),
        "scores": scores,
        "results": results,
    }
    if suite_field:
        output["suite"] = suite_field

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Resultados guardados en: {filepath}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_suite(
    client: httpx.Client,
    base_url: str,
    financial_context: dict,
    test_cases: list,
    judge: OllamaCloudJudge,
    label: str,
) -> tuple[list, float]:
    """Ejecuta una suite de test cases y retorna (resultados, wall_time_ms)."""
    results = []
    suite_start = time.time()
    print(f"--- {label} ({len(test_cases)} tests) ---")
    for test in test_cases:
        result = run_single_test(client, base_url, financial_context, test, judge)
        results.append(result)
    print()
    return results, (time.time() - suite_start) * 1000


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación de calidad de respuestas LLM con deepEval"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"URL base del servicio (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--suite",
        choices=["base", "enriched", "all"],
        default="all",
        help=(
            "Suite a ejecutar: "
            "'base' = quality_test_cases.json (sin additional_context), "
            "'enriched' = context_enriched_test_cases.json (con additional_context), "
            "'all' = ambas (default)"
        ),
    )
    parser.add_argument(
        "--base-ids",
        default=None,
        help="IDs de la suite base separados por coma. Sin valor = todos los casos.",
    )
    parser.add_argument(
        "--enriched-ids",
        default=None,
        help="IDs de la suite enriquecida separados por coma. Sin valor = todos los casos.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directorio donde guardar los JSON de resultados (default: tests/results)",
    )
    parser.add_argument(
        "--results-tag",
        choices=["full", "smoke"],
        default="full",
        help="Etiqueta de suite: 'smoke' añade el prefijo smoke_ al nombre del archivo.",
    )
    args = parser.parse_args()

    base_ids = set(args.base_ids.split(",")) if args.base_ids else None
    enriched_ids = set(args.enriched_ids.split(",")) if args.enriched_ids else None
    results_dir = Path(args.results_dir) if args.results_dir else None

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
    print(f"  Suite: {args.suite}")

    with httpx.Client() as client:
        # --- Suite base (sin additional_context) ---
        if args.suite in ("base", "all"):
            data = load_test_cases()
            base_cases = data["test_cases"]
            if base_ids:
                base_cases = [t for t in base_cases if t["id"] in base_ids]
            results_base, wall_base = run_suite(
                client, args.base_url,
                data["financial_context"], base_cases,
                judge, "CALIDAD BASE",
            )
            scores_base = calculate_scores(results_base, wall_base)
            print_summary(f"{judge_name} [base]", scores_base)
            save_results(judge_name, results_base, scores_base, suite_label="base", results_dir=results_dir, suite_tag=args.results_tag)

        # --- Suite enriquecida (con additional_context) ---
        if args.suite in ("enriched", "all"):
            enriched = load_enriched_test_cases()
            enriched_cases = enriched["test_cases"]
            if enriched_ids:
                enriched_cases = [t for t in enriched_cases if t["id"] in enriched_ids]
            results_enriched, wall_enriched = run_suite(
                client, args.base_url,
                enriched["financial_context"], enriched_cases,
                judge, "CALIDAD CON CONTEXTO ENRIQUECIDO",
            )
            scores_enriched = calculate_scores(results_enriched, wall_enriched)
            print_summary(f"{judge_name} [enriched]", scores_enriched)
            save_results(judge_name, results_enriched, scores_enriched, suite_label="enriched", results_dir=results_dir, suite_tag=args.results_tag)


if __name__ == "__main__":
    main()
