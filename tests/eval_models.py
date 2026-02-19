"""
Script de evaluación de modelos LLM para el servicio financiero.

Uso con Docker (recomendado):
  1. Configurar LLM_MODEL en .env con el modelo a evaluar
  2. Levantar el servicio: docker compose up -d
  3. Ejecutar:
     docker compose -f docker-compose.yaml -f docker-compose.test.yaml run --rm llm-tests
  4. Para especificar modelo:
     docker compose -f docker-compose.yaml -f docker-compose.test.yaml run --rm llm-tests \
       python tests/eval_models.py --base-url http://llm-service:8000 --model gpt-oss:120b-cloud
  5. Los resultados se guardan en tests/results/<modelo>_<timestamp>.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).parent
TEST_CASES_PATH = SCRIPT_DIR / "test_cases.json"
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120.0

def load_test_cases() -> dict:
    with open(TEST_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_payload(financial_context: dict, question: str) -> dict:
    return {
        "message": question,
        **financial_context
    }


def evaluate_guardrail(test: dict, response_text: str) -> dict:
    response_lower = response_text.lower()

    if test["expected_behavior"] == "reject":
        kws = test.get("reject_keywords", [])
        matched = [kw for kw in kws if kw.lower() in response_lower]
        passed = len(matched) > 0
        label = "PASS: Rechazo correctamente" if passed else "FAIL: No rechazo pregunta fuera de contexto"
    else:
        kws = test.get("accept_keywords", [])
        matched = [kw for kw in kws if kw.lower() in response_lower]
        passed = len(matched) > 0
        label = "PASS: Respondio con contenido financiero" if passed else "FAIL: No respondio adecuadamente"
    return {
        "passed": passed,  
        "matched_keywords": matched,
        "evaluation": label
    }


def evaluate_arithmetic(test: dict, response_text: str) -> dict:
    kws = test.get("answer_keywords", [])
    matched = [ kw for kw in kws if kw in response_text]
    passed = len(matched) > 0
    expected = test["expected_answer"]
    label = f"PASS: Respuesta correcta ({expected})" if passed else f"FAIL: Esperado ({expected})"
    return {
        "passed": passed,
        "expected_answer": expected,
        "matched_keywords": matched,
        "evaluation": label
    }


def evaluate_coherence(test: dict, response_text: str) -> dict:
    kws = test.get("context_keywords", [])
    response_lower = response_text.lower()
    matched = [kw for kw in kws if kw.lower() in response_lower]
    ratio = len(matched) / len(kws) if kws else 0
    label = f"Uso {len(matched)}/{len(kws)} refs contexto ({ratio:.0%})"
    return {
        "context_usage_ratio": round(ratio, 2),
        "matched_keywords": matched,
        "total_keywords": len(kws),
        "evaluation": label
    }


def run_single_test(
        client: httpx.Client,
        base_url: str,
        financial_context: dict,
        test: dict
) -> dict:
    payload = build_payload(financial_context, test["question"])
    preview = test["question"][:60]
    print(f"  [{test['id']}] {preview}...", end=" ", flush=True)
    start = time.time()
    try:
        response = client.post(f"{base_url}/llm/chat", json=payload, timeout=REQUEST_TIMEOUT)
        elapsed_ms = (time.time() - start) * 1000
        response.raise_for_status()
        response_text = response.json().get("message", "")
    except httpx.TimeoutException:
        elapsed_ms = (time.time() - start) * 1000
        print(f"TIMEOUT ({elapsed_ms:.0f}ms)")
        return {
            "id": test["id"], 
            "category": test["category"], 
            "question": test["question"],
            "response": "", 
            "response_time_ms": round(elapsed_ms, 2), 
            "error": "TIMEOUT",
            "evaluation": {"passed": False, "evaluation": "ERROR: Timeout"}
        }
    except httpx.HTTPStatusError as e:
        elapsed_ms = (time.time() - start) * 1000
        sc = e.response.status_code
        print(f"HTTP ERROR {sc} ({elapsed_ms:.0f}ms)")
        return {
            "id": test["id"], 
            "category": test["category"], 
            "question": test["question"],
            "response": "", 
            "response_time_ms": round(elapsed_ms, 2), 
            "error": f"HTTP {sc}",
            "evaluation": {"passed": False, "evaluation": f"ERROR: HTTP {sc}"}
        }
    
    # Evaluar según categoría
    evaluators = {
        "guardrail": evaluate_guardrail,
        "arithmetic": evaluate_arithmetic,
        "coherence": evaluate_coherence
    }

    # Obtengo mi función, las funciones también pueden ser variables
    fn = evaluators.get(test["category"])
    evaluation = fn(test, response_text) if fn else {"evaluation": "N/A"}
    print(f"{evaluation.get('evaluation', '')} ({elapsed_ms:.0f}ms)")

    return {
        "id": test["id"],
        "category": test["category"],
        "question": test["question"],
        "description": test.get("description", ""),
        "response": response_text,
        "response_time_ms": round(elapsed_ms, 2),
        "evaluation": evaluation
    }

def calculate_scores(results: list) -> dict:
    times = [r["response_time_ms"] for r in results if "error" not in r]
    times_sorted = sorted(times)

    timing = {}
    if times:
        p95_idx = int(len(times_sorted) * 0.95)
        p95_val = times_sorted[p95_idx] if len(times_sorted) > 1 else times_sorted[0]
        timing = {
            "avg_ms": round(sum(times) / len(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "p95_ms": round(p95_val, 2),
        }

    # Score guardrail
    guardrail_results = [r for r in results if r["category"] == "guardrail" and "error" not in r]
    guardrail_passed = sum(1 for r in guardrail_results if r["evaluation"].get("passed", False))
    guardrail_total = len(guardrail_results)

    # Score aritmética
    arithmetic_results = [r for r in results if r["category"] == "arithmetic" and "error" not in r]
    arithmetic_passed = sum(1 for r in arithmetic_results if r["evaluation"].get("passed", False))
    arithmetic_total = len(arithmetic_results)

    # Score coherencia
    coherence_results = [r for r in results if r["category"] == "coherence" and "error" not in r]
    coherence_avg_ratio = 0
    if coherence_results:
        ratios = [r["evaluation"].get("context_usage_ratio", 0) for r in coherence_results]
        coherence_avg_ratio = round(sum(ratios) / len(ratios), 2)

    scores = {
        "timing": timing,
        "guardrail": {
            "passed": guardrail_passed,
            "total": guardrail_total,
            "score_pct": round(guardrail_passed / guardrail_total * 100, 1) if guardrail_total else 0,
        },
        "arithmetic": {
            "passed": arithmetic_passed,
            "total": arithmetic_total,
            "score_pct": round(arithmetic_passed / arithmetic_total * 100, 1) if arithmetic_total else 0,
        },
        "coherence": {
            "avg_context_usage": coherence_avg_ratio,
            "total": len(coherence_results),
        },
    }

    # Score total ponderado: Adherencia 30% + Correctitud 50% + Tiempo 20%
    adherence_score = scores["guardrail"]["score_pct"]
    correctness_score = scores["arithmetic"]["score_pct"]

    # Tiempo: 100% si avg < 10s, 0% si avg > 60s, lineal entre ambos
    avg_time_s = timing.get("avg_ms", 60000) / 1000
    if avg_time_s <= 10:
        time_score = 100
    elif avg_time_s >= 60:
        time_score = 0
    else:
        time_score = round((1 - (avg_time_s - 10) / 50) * 100, 1)

    scores["time_score_pct"] = time_score
    scores["total_weighted"] = round(
        adherence_score * 0.30 + correctness_score * 0.50 + time_score * 0.20, 1
    )

    return scores


def print_summary(model_name: str, scores: dict):
    print("\n" + "=" * 65)
    print(f"  RESUMEN DE EVALUACIÓN — {model_name}")
    print("=" * 65)

    timing = scores.get("timing", {})
    print("\n  Tiempos de respuesta:")
    print(f"    Promedio:   {timing.get('avg_ms', 0) / 1000:.1f}s")
    print(f"    Mínimo:     {timing.get('min_ms', 0) / 1000:.1f}s")
    print(f"    Máximo:     {timing.get('max_ms', 0) / 1000:.1f}s")
    print(f"    P95:        {timing.get('p95_ms', 0) / 1000:.1f}s")

    g = scores["guardrail"]
    print(f"\n  Adherencia al contexto (guardrail):")
    print(f"    {g['passed']}/{g['total']} correctos — {g['score_pct']}%")

    a = scores["arithmetic"]
    print(f"\n  Correctitud aritmética:")
    print(f"    {a['passed']}/{a['total']} correctos — {a['score_pct']}%")

    c = scores["coherence"]
    print(f"\n  Coherencia (uso del contexto):")
    print(f"    Uso promedio del contexto: {c['avg_context_usage']:.0%}")

    print(f"\n  Score de tiempo: {scores['time_score_pct']}%")
    print(f"\n  SCORE TOTAL PONDERADO: {scores['total_weighted']}%")
    print("    (Adherencia 30% + Correctitud 50% + Tiempo 20%)")
    print("=" * 65)


def save_results(model_name: str, results: list, scores: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.json"
    filepath = RESULTS_DIR / filename

    output = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "scores": scores,
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Resultados guardados en: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Evaluación de modelos LLM para servicios financieros")
    parser.add_argument("--model", default=None, help="Nombre del modelo (para etiquetar resultados)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"URL base del servicio (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    #Cargar test cases
    data = load_test_cases()
    financial_context = data["financial_context"]
    test_cases = data["test_cases"]

    #Verificar que el servicio está activo
    print(f"\n Verificando servicio en {args.base_url}...")
    try:
        with httpx.Client() as client:
            health = client.get(f"{args.base_url}/health", timeout=5.0)
            health.raise_for_status()
            print(f" Servicio activo: {health.json()}")
    except Exception as e:
        print(f" ERROR: No se puede conectar al servicio: {e}")
        print(f" Asegúrate de que el servicio este corriendo en: {args.base_url}")
        sys.exit(1)

    # Detectar modelo si no se especificó
    model_name = args.model
    if not model_name:
        try:
            with httpx.Client() as client:
                llm_health = client.get(f"{args.base_url}/health/llm", timeout=10.0)
                llm_data = llm_health.json()
                model_name = llm_data.get("provider", "unknown")
        except Exception:
            model_name = "unknown"


    print(f"\n  Modelo: {model_name}")
    print(f"  Total test cases: {len(test_cases)}")
    print(f"  Categorías: guardrail ({sum(1 for t in test_cases if t['category'] == 'guardrail')}), "
          f"arithmetic ({sum(1 for t in test_cases if t['category'] == 'arithmetic')}), "
          f"coherence ({sum(1 for t in test_cases if t['category'] == 'coherence')})")
    print()

    # Ejecutar evaluación
    results = []
    categories = ["guardrail", "arithmetic", "coherence"]


    with httpx.Client() as client:
        for category in categories:
            category_tests = [t for t in test_cases if t["category"] == category]
            print(f"--- {category.upper()} ({len(category_tests)} tests) ---")
            for test in category_tests:
                result = run_single_test(client,
                                         args.base_url,
                                         financial_context,
                                         test)
                results.append(result)
            print()

    # Calcular scores y mostrar resumen
    scores = calculate_scores(results)
    print_summary(model_name, scores)

    #Guardar resultados
    save_results(model_name, results, scores)


if __name__ == "__main__":
    main()


