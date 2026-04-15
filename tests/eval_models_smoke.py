"""
Versión smoke (≈25%) de eval_models.py.

Ejecuta solo los casos de guardrail y clasificador indicados por --guardrail-ids
y --classifier-ids. Reutiliza toda la lógica de eval_models.py sin duplicar código.

Uso:
  python tests/eval_models_smoke.py \\
    --base-url http://llm-service:8000 \\
    --model gpt-oss:120b \\
    --guardrail-ids G1,G2,G3,T1,P1 \\
    --classifier-ids CL1,CL4,CL7,CL10,CL13,CL15,CL17
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import httpx

# Reutiliza toda la lógica del módulo principal
sys.path.insert(0, str(Path(__file__).parent))
from eval_models import (
    load_test_cases,
    load_classifier_test_cases,
    run_single_test,
    run_classifier_test,
    calculate_scores,
    calculate_classifier_scores,
    print_summary,
    print_classifier_summary,
    RESULTS_DIR,
    REQUEST_TIMEOUT,
)

DEFAULT_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

SMOKE_GUARDRAIL_IDS = {"G1", "G2", "G3", "T1", "P1"}
SMOKE_CLASSIFIER_IDS = {"CL1", "CL4", "CL7", "CL10", "CL13", "CL15", "CL17"}


def save_smoke_results(label: str, model_name: str, results: list, scores: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / f"smoke_{label}_{safe_name}_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "suite": f"smoke_{label}",
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
        description="Smoke evaluation (≈25%) de guardrails y clasificador"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--guardrail-ids",
        default=",".join(SMOKE_GUARDRAIL_IDS),
        help="IDs de guardrail separados por coma (default: %(default)s)",
    )
    parser.add_argument(
        "--classifier-ids",
        default=",".join(SMOKE_CLASSIFIER_IDS),
        help="IDs de clasificador separados por coma (default: %(default)s)",
    )
    args = parser.parse_args()

    guardrail_ids = set(args.guardrail_ids.split(","))
    classifier_ids = set(args.classifier_ids.split(","))

    base_url = args.base_url

    # Health check
    print(f"\n Verificando servicio en {base_url}...")
    try:
        with httpx.Client() as client:
            health = client.get(f"{base_url}/health", timeout=5.0)
            health.raise_for_status()
            print(f" Servicio activo: {health.json()}")
    except Exception as e:
        print(f" ERROR: No se puede conectar al servicio: {e}")
        sys.exit(1)

    # Obtener nombre del modelo
    model_name = args.model or "unknown"

    # Smoke test LLM
    print(f"\n  Smoke test del LLM ({base_url}/llm/smoke-test)...")
    try:
        with httpx.Client() as client:
            smoke = client.get(f"{base_url}/llm/smoke-test", timeout=REQUEST_TIMEOUT)
            if smoke.status_code == 200:
                print(f"  LLM responde OK → {smoke.json().get('model_response', '')!r}")
            else:
                print(f"  SMOKE TEST FALLÓ — HTTP {smoke.status_code}")
                sys.exit(1)
    except httpx.TimeoutException:
        print("  SMOKE TEST TIMEOUT")
        sys.exit(1)
    except Exception as e:
        print(f"  SMOKE TEST ERROR — {e}")
        sys.exit(1)

    # --- Guardrail subset ---
    data = load_test_cases()
    financial_context = data["financial_context"]
    all_guardrail = [t for t in data["test_cases"] if t["category"] == "guardrail"]
    guardrail_cases = [t for t in all_guardrail if t["id"] in guardrail_ids]

    off_count = sum(1 for t in guardrail_cases if t.get("subcategory") == "off_topic")
    on_count  = sum(1 for t in guardrail_cases if t.get("subcategory") == "on_topic")
    print(f"\n  Modelo: {model_name}")
    print(f"  Guardrail smoke: {len(guardrail_cases)} casos ({off_count} off_topic, {on_count} on_topic)")
    print(f"  IDs: {', '.join(t['id'] for t in guardrail_cases)}\n")

    guardrail_results = []
    with httpx.Client() as client:
        print(f"--- GUARDRAIL SMOKE ({len(guardrail_cases)} tests) ---")
        for test in guardrail_cases:
            result = run_single_test(client, base_url, financial_context, test)
            guardrail_results.append(result)
        print()

    guardrail_scores = calculate_scores(guardrail_results)
    print_summary(model_name, guardrail_scores)
    save_smoke_results("guardrail", model_name, guardrail_results, guardrail_scores)

    # --- Classifier subset ---
    all_classifier = load_classifier_test_cases()
    classifier_cases = [t for t in all_classifier if t["id"] in classifier_ids]

    print(f"\n  Classifier smoke: {len(classifier_cases)} casos")
    print(f"  IDs: {', '.join(t['id'] for t in classifier_cases)}\n")

    classifier_results = []
    with httpx.Client() as client:
        print(f"--- CLASIFICADOR SMOKE ({len(classifier_cases)} tests) ---")
        for test in classifier_cases:
            result = run_classifier_test(client, base_url, test)
            classifier_results.append(result)
        print()

    classifier_scores = calculate_classifier_scores(classifier_results)
    print_classifier_summary(model_name, classifier_scores)
    save_smoke_results("classifier", model_name, classifier_results, classifier_scores)


if __name__ == "__main__":
    main()
