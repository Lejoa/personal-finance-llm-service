"""
Evaluación multi-turno de respuestas del LLM con DeepEval.
Usa ConversationalTestCase con métricas conversacionales de DeepEval.

Métricas evaluadas:
  - ConversationCompleteness: ¿el chatbot abordó todos los intents del usuario en la conversación?
  - TurnRelevancy:            ¿cada respuesta del asistente es relevante a su turno específico?
  - KnowledgeRetention:       ¿el chatbot retiene hechos mencionados a lo largo de los turnos?
  - RoleAdherence:            ¿el chatbot se mantiene en su rol de asistente financiero?

Prerrequisitos:
  - OLLAMA_API_KEY y LLM_MODEL definidas en .env
  - Servicio llm-service corriendo: docker compose up -d

Uso con Docker (recomendado):
  1. docker compose up -d
  2. docker compose -f docker-compose.test.yaml --profile test-multi-turn up \\
       --abort-on-container-exit llm-multi-turn-tests

Los resultados se guardan en tests/results/multi_turn_<modelo>_<timestamp>.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    TurnRelevancyMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, Turn
from langchain_openai import ChatOpenAI

SCRIPT_DIR = Path(__file__).parent
MULTI_TURN_TEST_CASES_PATH = SCRIPT_DIR / "multi_turn_test_cases.json"
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120.0

METRIC_THRESHOLD = 0.7

METRIC_LABELS = {
    "conversation_completeness": "ConversationCompletenessMetric",
    "turn_relevancy": "TurnRelevancyMetric",
    "knowledge_retention": "KnowledgeRetentionMetric",
    "role_adherence": "RoleAdherenceMetric",
}


# ---------------------------------------------------------------------------
# Juez: Ollama Cloud
# ---------------------------------------------------------------------------

class OllamaCloudJudge(DeepEvalBaseLLM):
    """Juez deepEval usando Ollama Cloud como backend."""

    def __init__(self):
        self._model = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-oss:120b-cloud"),
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
        return f"ollama-cloud/{os.getenv('LLM_MODEL', 'gpt-oss:120b-cloud')}"


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_test_cases() -> dict:
    with open(MULTI_TURN_TEST_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Contexto financiero como retrieval_context para turnos del asistente
# ---------------------------------------------------------------------------

def financial_context_to_retrieval_context(financial_context: dict) -> list[str]:
    """
    Convierte el financial_context estructurado en una lista de hechos textuales
    para usar como retrieval_context en los turnos del asistente.
    """
    uc = financial_context["user_context"]
    fs = financial_context["financial_summary"]
    categories = financial_context["categories"]
    budgets = financial_context["budgets"]

    absolute_savings = fs["total_income"] - fs["total_expenses"]

    facts = [
        (
            f"Moneda del usuario: {uc['currency']}. Locale: {uc['locale']}. "
            f"Nivel financiero: {uc['financial_level']}."
        ),
        (
            f"Período: {fs['period']}. Ingresos totales: {fs['total_income']:,.0f} {uc['currency']}. "
            f"Gastos totales: {fs['total_expenses']:,.0f} {uc['currency']}. "
            f"Tasa de ahorro: {fs['savings_rate']}%."
        ),
        f"Ahorro absoluto del período: {absolute_savings:,.0f} {uc['currency']}.",
    ]

    cat_parts = ", ".join(
        f"{c['name']}: {c['amount']:,.0f} {uc['currency']}" for c in categories
    )
    facts.append(f"Categorías de gasto del período: {cat_parts}.")

    for b in budgets:
        excess = b["spent"] - b["limit"]
        if excess > 0:
            status = f"EXCEDIDO en {excess:,.0f} {uc['currency']}"
        else:
            status = f"dentro del límite ({b['limit'] - b['spent']:,.0f} {uc['currency']} disponibles)"
        facts.append(
            f"Presupuesto '{b['name']}': límite {b['limit']:,.0f} {uc['currency']}, "
            f"gastado {b['spent']:,.0f} {uc['currency']} — {status}."
        )

    return facts


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
    except Exception:
        elapsed_ms = (time.time() - start) * 1000
        return None, round(elapsed_ms, 2)


# ---------------------------------------------------------------------------
# Recolección de turnos de la conversación
# ---------------------------------------------------------------------------

def collect_conversation_turns(
    client: httpx.Client,
    base_url: str,
    financial_context: dict,
    user_turns: list[str],
    retrieval_context: list[str],
) -> tuple[list[Turn], list[float], list[str]]:
    """
    Itera los turnos del usuario, llama a /llm/chat por cada uno y construye
    la lista de Turn objects para ConversationalTestCase.

    Retorna:
        turns: lista alternada [Turn(user), Turn(assistant), ...]
        response_times: lista de ms por cada llamada HTTP
        errors: lista de mensajes de error (vacía si todo OK)
    """
    turns: list[Turn] = []
    response_times: list[float] = []
    errors: list[str] = []

    for i, user_message in enumerate(user_turns):
        turns.append(Turn(role="user", content=user_message))

        actual_output, elapsed_ms = get_llm_response(
            client, base_url, financial_context, user_message
        )
        response_times.append(elapsed_ms)

        if actual_output is None:
            error_msg = f"Turno {i + 1}: sin respuesta del servicio ({elapsed_ms:.0f}ms)"
            errors.append(error_msg)
            turns.append(
                Turn(
                    role="assistant",
                    content="[ERROR: sin respuesta]",
                    retrieval_context=retrieval_context,
                )
            )
        else:
            turns.append(
                Turn(
                    role="assistant",
                    content=actual_output,
                    retrieval_context=retrieval_context,
                )
            )

    return turns, response_times, errors


# ---------------------------------------------------------------------------
# Construcción de métricas
# ---------------------------------------------------------------------------

def build_metrics(metric_keys: list[str], judge: OllamaCloudJudge) -> list:
    """Instancia solo las métricas declaradas en el test case."""
    builders = {
        "conversation_completeness": lambda: ConversationCompletenessMetric(
            threshold=METRIC_THRESHOLD, model=judge
        ),
        "turn_relevancy": lambda: TurnRelevancyMetric(
            threshold=METRIC_THRESHOLD, model=judge
        ),
        "knowledge_retention": lambda: KnowledgeRetentionMetric(
            threshold=METRIC_THRESHOLD, model=judge
        ),
        "role_adherence": lambda: RoleAdherenceMetric(
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
    retrieval_context: list[str],
) -> dict:
    preview = test["description"][:55]
    print(f"  [{test['id']}] {preview}...", end=" ", flush=True)

    turns, response_times, errors = collect_conversation_turns(
        client,
        base_url,
        financial_context,
        test["user_turns"],
        retrieval_context,
    )

    total_response_time = sum(response_times)

    if len(errors) == len(test["user_turns"]):
        print(f"ERROR ({total_response_time:.0f}ms)")
        return {
            "id": test["id"],
            "use_case": test["use_case"],
            "description": test.get("description", ""),
            "metrics_evaluated": test["metrics"],
            "total_response_time_ms": round(total_response_time, 2),
            "turn_response_times_ms": [round(t, 2) for t in response_times],
            "errors": errors,
            "metric_results": {},
            "passed": False,
        }

    conv_test_case = ConversationalTestCase(
        turns=turns,
        scenario=test.get("scenario", ""),
        expected_outcome=test.get("expected_outcome", ""),
        chatbot_role=test.get("chatbot_role", ""),
        name=test["id"],
    )

    metrics = build_metrics(test["metrics"], judge)
    metric_results = {}
    all_passed = True

    for metric in metrics:
        try:
            metric.measure(conv_test_case)
            passed = metric.score >= METRIC_THRESHOLD
            metric_results[metric.__class__.__name__] = {
                "score": round(metric.score, 4),
                "threshold": METRIC_THRESHOLD,
                "passed": passed,
                "reason": metric.reason,
            }
            if not passed:
                all_passed = False
        except Exception as e:
            metric_results[metric.__class__.__name__] = {
                "score": None,
                "passed": False,
                "error": str(e),
            }
            all_passed = False

    status = "PASS" if all_passed else "FAIL"
    print(f"{status} ({total_response_time:.0f}ms)")

    return {
        "id": test["id"],
        "use_case": test["use_case"],
        "description": test.get("description", ""),
        "turns_executed": len(test["user_turns"]),
        "actual_turns": [
            {"role": t.role, "content": t.content[:200]}
            for t in turns
        ],
        "metrics_evaluated": test["metrics"],
        "total_response_time_ms": round(total_response_time, 2),
        "turn_response_times_ms": [round(t, 2) for t in response_times],
        "errors": errors,
        "passed": all_passed,
        "metric_results": metric_results,
    }


# ---------------------------------------------------------------------------
# Cálculo de scores agregados
# ---------------------------------------------------------------------------

def calculate_scores(results: list) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    errors = sum(1 for r in results if r.get("errors"))

    all_times = [r["total_response_time_ms"] for r in results]
    times_sorted = sorted(all_times)

    timing = {}
    if all_times:
        p95_idx = min(int(len(times_sorted) * 0.95), len(times_sorted) - 1)
        timing = {
            "avg_ms": round(sum(all_times) / len(all_times), 2),
            "min_ms": round(min(all_times), 2),
            "max_ms": round(max(all_times), 2),
            "p95_ms": round(times_sorted[p95_idx], 2),
        }

    valid = [r for r in results if not r.get("errors")]

    per_metric: dict[str, dict] = {}
    for key, class_name in METRIC_LABELS.items():
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
                "label": class_name,
                "evaluated": count_for_metric,
                "passed": passes_for_metric,
                "avg_score": round(sum(scores_for_metric) / count_for_metric, 4),
                "pass_rate_pct": round(passes_for_metric / count_for_metric * 100, 1),
            }

    per_use_case: dict[str, dict] = {}
    for use_case in ("transaction", "financial_question", "insight"):
        subset = [r for r in results if r.get("use_case") == use_case]
        if subset:
            subset_passed = sum(1 for r in subset if r.get("passed", False))
            per_use_case[use_case] = {
                "total": len(subset),
                "passed": subset_passed,
                "score_pct": round(subset_passed / len(subset) * 100, 1),
            }

    return {
        "timing": timing,
        "overall": {
            "total": total,
            "passed": passed,
            "errors_with_partial_turns": errors,
            "score_pct": round(passed / total * 100, 1) if total else 0,
        },
        "per_metric": per_metric,
        "per_use_case": per_use_case,
    }


# ---------------------------------------------------------------------------
# Impresión del resumen
# ---------------------------------------------------------------------------

def print_summary(judge_name: str, scores: dict):
    print("\n" + "=" * 65)
    print(f"  RESUMEN MULTI-TURNO — Juez: {judge_name}")
    print("=" * 65)

    timing = scores.get("timing", {})
    print("\n  Tiempos de respuesta totales por conversación:")
    print(f"    Promedio: {timing.get('avg_ms', 0) / 1000:.1f}s")
    print(f"    Mínimo:   {timing.get('min_ms', 0) / 1000:.1f}s")
    print(f"    Máximo:   {timing.get('max_ms', 0) / 1000:.1f}s")
    print(f"    P95:      {timing.get('p95_ms', 0) / 1000:.1f}s")

    ov = scores["overall"]
    print(f"\n  Resultado general: {ov['passed']}/{ov['total']} conversaciones pasaron — {ov['score_pct']}%")
    if ov.get("errors_with_partial_turns", 0):
        print(f"  Con turnos fallidos: {ov['errors_with_partial_turns']} conversación(es)")

    METRIC_DESCRIPTIONS = {
        "ConversationCompletenessMetric": (
            "Evalúa si el chatbot abordó todos los intents del usuario "
            "a lo largo de la conversación completa. Un score bajo indica "
            "que el asistente dejó preguntas o solicitudes sin resolver."
        ),
        "TurnRelevancyMetric": (
            "Evalúa si cada respuesta del asistente es relevante al turno "
            "específico del usuario. Detecta respuestas genéricas o que no "
            "abordan directamente lo que se preguntó en ese turno."
        ),
        "KnowledgeRetentionMetric": (
            "Evalúa si el chatbot retiene y referencia correctamente hechos "
            "mencionados en turnos anteriores. Expone la limitación del "
            "servicio sin memoria de conversación."
        ),
        "RoleAdherenceMetric": (
            "Evalúa si el chatbot se mantiene en su rol de asistente "
            "financiero personal. Detecta respuestas fuera de rol o que "
            "abandonan el enfoque en finanzas personales."
        ),
    }

    print("\n  Desglose por métrica:")
    for _key, m in scores.get("per_metric", {}).items():
        desc = METRIC_DESCRIPTIONS.get(m["label"], "")
        print(
            f"\n    {m['label']:<35} "
            f"{m['passed']}/{m['evaluated']} pasaron "
            f"(avg score: {m['avg_score']:.2f}, pass rate: {m['pass_rate_pct']}%)"
        )
        if desc:
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

    print("\n  Desglose por caso de uso:")
    use_case_labels = {
        "transaction": "Registro de transacciones",
        "financial_question": "Preguntas financieras",
        "insight": "Preguntas de insights",
    }
    for use_case, data in scores.get("per_use_case", {}).items():
        label = use_case_labels.get(use_case, use_case)
        print(
            f"    {label:<30} "
            f"{data['passed']}/{data['total']} pasaron — {data['score_pct']}%"
        )

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# Guardado de resultados
# ---------------------------------------------------------------------------

def save_results(judge_name: str, results: list, scores: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = judge_name.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multi_turn_{safe_name}_{timestamp}.json"
    filepath = RESULTS_DIR / filename

    output = {
        "judge": judge_name,
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
        description="Evaluación multi-turno de respuestas LLM con DeepEval"
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

    retrieval_context = financial_context_to_retrieval_context(financial_context)

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
    print(f"  Conversaciones: {len(test_cases)}")
    print()

    results = []
    with httpx.Client() as client:
        print(f"--- MULTI-TURNO ({len(test_cases)} conversaciones) ---")
        for test in test_cases:
            result = run_single_test(
                client, args.base_url, financial_context, test, judge, retrieval_context
            )
            results.append(result)
        print()

    scores = calculate_scores(results)
    print_summary(judge_name, scores)
    save_results(judge_name, results, scores)


if __name__ == "__main__":
    main()