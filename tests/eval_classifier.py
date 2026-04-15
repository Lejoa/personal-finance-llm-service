"""
Script de evaluación del clasificador de contexto (/llm/classify-context).

Verifica que el LLM clasifique correctamente el tipo de contexto financiero
adicional que necesita cada mensaje del usuario:
  - trends:     comparaciones con meses anteriores, historial
  - budget:     estado de presupuestos, límites, margen disponible
  - categories: desglose por categoría, ranking de gastos
  - savings:    proyección de ahorro, velocidad de gasto
  - none:       registro de transacciones, preguntas respondibles con contexto base

Uso con Docker (recomendado):
  1. Levantar el servicio: docker compose up -d
  2. Ejecutar:
     docker compose -f docker-compose.yaml -f docker-compose.test.yaml run --rm llm-tests \\
       python tests/eval_classifier.py --base-url http://llm-service:8000
  3. Para especificar modelo:
     docker compose -f docker-compose.yaml -f docker-compose.test.yaml run --rm llm-tests \\
       python tests/eval_classifier.py --base-url http://llm-service:8000 --model gpt-4o

Los resultados se guardan en tests/results/classifier_<modelo>_<timestamp>.json
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
CLASSIFIER_TEST_CASES_PATH = SCRIPT_DIR / "classifier_test_cases.json"
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


def load_test_cases() -> list:
    with open(CLASSIFIER_TEST_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)["test_cases"]


def run_single_test(client: httpx.Client, base_url: str, test: dict) -> dict:
    """Llama a /llm/classify-context y verifica que retorne el context_type esperado."""
    preview = test["message"][:60]
    print(f"  [{test['id']}] {preview}...", end=" ", flush=True)
    start = time.time()

    try:
        response = client.post(
            f"{base_url}/llm/classify-context",
            json={"message": test["message"]},
            timeout=30.0,
        )
        elapsed_ms = (time.time() - start) * 1000
        status_code = response.status_code
        try:
            body = response.json()
        except Exception:
            body = {}
    except httpx.TimeoutException:
        elapsed_ms = (time.time() - start) * 1000
        print(f"TIMEOUT ({elapsed_ms:.0f}ms)")
        return {
            "id": test["id"],
            "message": test["message"],
            "description": test["description"],
            "expected_context_type": test["context_type"],
            "actual_context_type": None,
            "status_code": None,
            "response_time_ms": round(elapsed_ms, 2),
            "error": "TIMEOUT",
            "passed": False,
        }
    except httpx.RequestError as exc:
        elapsed_ms = (time.time() - start) * 1000
        print(f"ERROR ({type(exc).__name__}) ({elapsed_ms:.0f}ms)")
        return {
            "id": test["id"],
            "message": test["message"],
            "description": test["description"],
            "expected_context_type": test["context_type"],
            "actual_context_type": None,
            "status_code": None,
            "response_time_ms": round(elapsed_ms, 2),
            "error": str(exc),
            "passed": False,
        }

    actual = body.get("context_type", "")
    expected = test["context_type"]
    passed = status_code == 200 and actual == expected

    label = (
        f"PASS ({actual})"
        if passed
        else f"FAIL (esperado={expected!r}, obtenido={actual!r}, HTTP {status_code})"
    )
    print(f"{label} ({elapsed_ms:.0f}ms)")

    return {
        "id": test["id"],
        "message": test["message"],
        "description": test["description"],
        "expected_context_type": expected,
        "actual_context_type": actual,
        "status_code": status_code,
        "response_time_ms": round(elapsed_ms, 2),
        "passed": passed,
    }


def calculate_scores(results: list) -> dict:
    valid = [r for r in results if "error" not in r]
    passed = sum(1 for r in valid if r["passed"])
    total = len(valid)

    times = [r["response_time_ms"] for r in valid]
    timing = {}
    if times:
        times_sorted = sorted(times)
        p95_idx = min(int(len(times_sorted) * 0.95), len(times_sorted) - 1)
        timing = {
            "avg_ms": round(sum(times) / len(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "p95_ms": round(times_sorted[p95_idx], 2),
        }

    by_type = {}
    for ct in ("trends", "budget", "categories", "savings", "none"):
        subset = [r for r in valid if r.get("expected_context_type") == ct]
        p = sum(1 for r in subset if r["passed"])
        by_type[ct] = {
            "passed": p,
            "total": len(subset),
            "score_pct": round(p / len(subset) * 100, 1) if subset else 0,
        }

    return {
        "timing": timing,
        "classifier": {
            "passed": passed,
            "total": total,
            "score_pct": round(passed / total * 100, 1) if total else 0,
            "by_type": by_type,
        },
    }


def print_summary(model_name: str, scores: dict):
    print("\n" + "=" * 65)
    print(f"  RESUMEN CLASIFICADOR DE CONTEXTO — {model_name}")
    print("=" * 65)

    timing = scores.get("timing", {})
    print("\n  Tiempos de respuesta:")
    print(f"    Promedio: {timing.get('avg_ms', 0) / 1000:.2f}s")
    print(f"    Mínimo:   {timing.get('min_ms', 0) / 1000:.2f}s")
    print(f"    Máximo:   {timing.get('max_ms', 0) / 1000:.2f}s")
    print(f"    P95:      {timing.get('p95_ms', 0) / 1000:.2f}s")

    c = scores["classifier"]
    print(f"\n  Total: {c['passed']}/{c['total']} correctos — {c['score_pct']}%")
    print(f"\n  Desglose por context_type esperado:")
    for ct, s in c["by_type"].items():
        print(f"    {ct:<12} {s['passed']}/{s['total']} — {s['score_pct']}%")
    print("=" * 65)


def save_results(model_name: str, results: list, scores: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / f"classifier_{safe_name}_{timestamp}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "scores": scores,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  Resultados guardados en: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación del clasificador de contexto /llm/classify-context"
    )
    parser.add_argument(
        "--model", default=None, help="Nombre del modelo (para etiquetar resultados)"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"URL base del servicio (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()

    test_cases = load_test_cases()

    print(f"\n  Verificando servicio en {args.base_url}...")
    try:
        with httpx.Client() as client:
            health = client.get(f"{args.base_url}/health", timeout=5.0)
            health.raise_for_status()
            print(f"  Servicio activo: {health.json()}")
    except Exception as e:
        print(f"  ERROR: No se puede conectar al servicio: {e}")
        sys.exit(1)

    model_name = args.model
    if not model_name:
        try:
            with httpx.Client() as client:
                llm_health = client.get(f"{args.base_url}/health/llm", timeout=10.0)
                model_name = llm_health.json().get("provider", "unknown")
        except Exception:
            model_name = "unknown"

    by_type_count = {}
    for ct in ("trends", "budget", "categories", "savings", "none"):
        by_type_count[ct] = sum(1 for t in test_cases if t["context_type"] == ct)

    print(f"\n  Modelo: {model_name}")
    print(f"  Tests totales: {len(test_cases)}")
    print(f"  Por context_type: " + ", ".join(f"{k}={v}" for k, v in by_type_count.items()))
    print()

    results = []
    with httpx.Client() as client:
        print(f"--- CLASIFICADOR ({len(test_cases)} tests) ---")
        for test in test_cases:
            result = run_single_test(client, args.base_url, test)
            results.append(result)
        print()

    scores = calculate_scores(results)
    print_summary(model_name, scores)
    save_results(model_name, results, scores)


if __name__ == "__main__":
    main()