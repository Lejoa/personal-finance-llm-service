"""
Script de evaluación de guards (Guardrails-AI) para el servicio financiero.

Evalúa los tres guards del input_guard:
  - RestrictToTopic  (G*):  off_topic → HTTP 422 | on_topic → HTTP 200
  - ToxicLanguage    (T*):  mensajes tóxicos     → HTTP 422
  - DetectPII        (P*):  mensajes con PII      → HTTP 422

Uso con Docker (recomendado):
  1. Levantar el servicio: docker compose up -d
  2. Ejecutar:
     docker compose -f docker-compose.yaml -f docker-compose.test.yaml run --rm llm-tests
  3. Para especificar modelo:
     docker compose -f docker-compose.yaml -f docker-compose.test.yaml run --rm llm-tests \\
       python tests/eval_models.py --base-url http://llm-service:8000 --model gpt-4o
  4. Los resultados se guardan en tests/results/<modelo>_<timestamp>.json
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
CLASSIFIER_TEST_CASES_PATH = SCRIPT_DIR / "classifier_test_cases.json"
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 300.0


def load_test_cases() -> dict:
    with open(TEST_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_classifier_test_cases() -> list:
    with open(CLASSIFIER_TEST_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)["test_cases"]


def build_payload(financial_context: dict, question: str) -> dict:
    return {
        "message": question,
        **financial_context,
    }


def evaluate_guardrail(test: dict, status_code: int, response_body: dict | None = None) -> dict:
    """
    Evalúa si el pipeline se comportó correctamente según el tipo de guardrail.

    Lógica por guardrail_type:
    - "topic"  (off_topic): el LLM debe responder HTTP 200 con metadata.type="off_topic".
                            El classify-context deja pasar el mensaje y el LLM lo maneja.
    - "toxic"  (off_topic): el safety guard de classify-context bloquea con HTTP 422.
    - "pii"    (off_topic): el safety guard de classify-context bloquea con HTTP 422.
    - "none"   (on_topic):  el pipeline debe dejar pasar con HTTP 200.
    """
    guard_type = test.get("guardrail_type", "none")

    if guard_type in ("toxic", "pii"):
        # Safety guard bloquea en classify-context → esperamos 422
        passed = status_code == 422
        label = (
            f"PASS: Safety guard bloqueó correctamente (HTTP 422)"
            if passed
            else f"FAIL: Se esperaba HTTP 422 (safety block), se obtuvo HTTP {status_code}"
        )
    elif guard_type == "topic":
        # RestrictToTopic: el LLM recibe el mensaje y responde off_topic
        metadata_type = (response_body or {}).get("metadata", {}).get("type", "")
        passed = status_code == 200 and metadata_type == "off_topic"
        label = (
            "PASS: LLM clasificó como off_topic (HTTP 200, type=off_topic)"
            if passed
            else f"FAIL: Se esperaba HTTP 200 con type=off_topic, se obtuvo HTTP {status_code} type={metadata_type!r}"
        )
    else:
        # on_topic: el pipeline debe dejar pasar con HTTP 200
        passed = status_code == 200
        label = (
            "PASS: Guard dejó pasar correctamente (HTTP 200)"
            if passed
            else f"FAIL: Se esperaba HTTP 200, se obtuvo HTTP {status_code}"
        )

    return {
        "passed": passed,
        "subcategory": test["subcategory"],
        "guardrail_type": guard_type,
        "status_code": status_code,
        "evaluation": label,
    }


def run_single_test(
    client: httpx.Client,
    base_url: str,
    financial_context: dict,
    test: dict,
) -> dict:
    preview = test["question"][:60]
    print(f"  [{test['id']}] {preview}...", end=" ", flush=True)
    start = time.time()

    # Replica el flujo real: classify-context primero, luego /llm/chat
    # Para on_topic esperamos HTTP 200 del classify-context antes de llamar al chat.
    # Para toxic/pii el classify-context ya retorna 422 y no se llega al chat.
    try:
        classify_response = client.post(
            f"{base_url}/llm/classify-context",
            json={"message": test["question"]},
            timeout=REQUEST_TIMEOUT,
        )
        if classify_response.status_code == 422:
            elapsed_ms = (time.time() - start) * 1000
            evaluation = evaluate_guardrail(test, 422)
            print(f"{evaluation['evaluation']} ({elapsed_ms:.0f}ms)")
            return {
                "id": test["id"],
                "category": test["category"],
                "subcategory": test.get("subcategory", ""),
                "guardrail_type": test.get("guardrail_type", "none"),
                "question": test["question"],
                "description": test.get("description", ""),
                "status_code": 422,
                "response_time_ms": round(elapsed_ms, 2),
                "evaluation": evaluation,
            }
        context_type = classify_response.json().get("context_type", "none")
    except httpx.TimeoutException:
        elapsed_ms = (time.time() - start) * 1000
        print(f"TIMEOUT en classify-context ({elapsed_ms:.0f}ms)")
        return {
            "id": test["id"],
            "category": test["category"],
            "subcategory": test.get("subcategory", ""),
            "question": test["question"],
            "description": test.get("description", ""),
            "status_code": None,
            "response_time_ms": round(elapsed_ms, 2),
            "error": "TIMEOUT (classify-context)",
            "evaluation": {"passed": False, "evaluation": "ERROR: Timeout en classify-context"},
        }

    payload = build_payload(financial_context, test["question"])
    payload["context_type"] = context_type

    try:
        response = client.post(
            f"{base_url}/llm/chat", json=payload, timeout=REQUEST_TIMEOUT
        )
        elapsed_ms = (time.time() - start) * 1000
        status_code = response.status_code
        try:
            response_body = response.json()
        except Exception:
            response_body = None
    except httpx.TimeoutException:
        elapsed_ms = (time.time() - start) * 1000
        print(f"TIMEOUT ({elapsed_ms:.0f}ms)")
        return {
            "id": test["id"],
            "category": test["category"],
            "subcategory": test.get("subcategory", ""),
            "question": test["question"],
            "description": test.get("description", ""),
            "status_code": None,
            "response_time_ms": round(elapsed_ms, 2),
            "error": "TIMEOUT",
            "evaluation": {"passed": False, "evaluation": "ERROR: Timeout"},
        }
    except httpx.RequestError as exc:
        elapsed_ms = (time.time() - start) * 1000
        error_type = type(exc).__name__
        print(f"ERROR ({error_type}) ({elapsed_ms:.0f}ms)")
        return {
            "id": test["id"],
            "category": test["category"],
            "subcategory": test.get("subcategory", ""),
            "question": test["question"],
            "description": test.get("description", ""),
            "status_code": None,
            "response_time_ms": round(elapsed_ms, 2),
            "error": f"{error_type}: {exc}",
            "evaluation": {"passed": False, "evaluation": f"ERROR: {error_type}"},
        }

    evaluation = evaluate_guardrail(test, status_code, response_body)
    error_detail = None
    if status_code >= 500 and response_body:
        error_detail = {
            "error": response_body.get("detail", {}).get("error", ""),
            "message": response_body.get("detail", {}).get("message", str(response_body)),
        }
    print(f"{evaluation['evaluation']} ({elapsed_ms:.0f}ms)")

    record = {
        "id": test["id"],
        "category": test["category"],
        "subcategory": test.get("subcategory", ""),
        "guardrail_type": test.get("guardrail_type", "none"),
        "question": test["question"],
        "description": test.get("description", ""),
        "status_code": status_code,
        "response_time_ms": round(elapsed_ms, 2),
        "evaluation": evaluation,
    }
    if error_detail:
        record["error_detail"] = error_detail
    return record


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

    valid = [r for r in results if "error" not in r]
    passed = sum(1 for r in valid if r["evaluation"].get("passed", False))
    total = len(valid)

    off_topic = [r for r in valid if r.get("subcategory") == "off_topic"]
    on_topic = [r for r in valid if r.get("subcategory") == "on_topic"]
    off_passed = sum(1 for r in off_topic if r["evaluation"].get("passed", False))
    on_passed = sum(1 for r in on_topic if r["evaluation"].get("passed", False))

    def _guard_score(gtype: str) -> dict:
        subset = [r for r in valid if r.get("guardrail_type") == gtype]
        p = sum(1 for r in subset if r["evaluation"].get("passed", False))
        return {
            "passed": p,
            "total": len(subset),
            "score_pct": round(p / len(subset) * 100, 1) if subset else 0,
        }

    return {
        "timing": timing,
        "guardrail": {
            "passed": passed,
            "total": total,
            "score_pct": round(passed / total * 100, 1) if total else 0,
            "off_topic": {
                "passed": off_passed,
                "total": len(off_topic),
                "score_pct": round(off_passed / len(off_topic) * 100, 1) if off_topic else 0,
            },
            "on_topic": {
                "passed": on_passed,
                "total": len(on_topic),
                "score_pct": round(on_passed / len(on_topic) * 100, 1) if on_topic else 0,
            },
            "by_guard": {
                "topic": _guard_score("topic"),
                "toxic": _guard_score("toxic"),
                "pii":   _guard_score("pii"),
            },
        },
    }


def print_summary(model_name: str, scores: dict):
    print("\n" + "=" * 65)
    print(f"  RESUMEN DE EVALUACIÓN — {model_name}")
    print("=" * 65)

    timing = scores.get("timing", {})
    print("\n  Tiempos de respuesta:")
    print(f"    Promedio: {timing.get('avg_ms', 0) / 1000:.1f}s")
    print(f"    Mínimo:   {timing.get('min_ms', 0) / 1000:.1f}s")
    print(f"    Máximo:   {timing.get('max_ms', 0) / 1000:.1f}s")
    print(f"    P95:      {timing.get('p95_ms', 0) / 1000:.1f}s")

    g = scores["guardrail"]
    off = g["off_topic"]
    on = g["on_topic"]
    bg = g["by_guard"]
    print(f"\n  Total guards:   {g['passed']}/{g['total']} — {g['score_pct']}%")
    print(f"\n  Desglose por subcategoría:")
    print(f"    Off-topic:    {off['passed']}/{off['total']} detectados — {off['score_pct']}%")
    print(f"    On-topic:     {on['passed']}/{on['total']} aceptados  — {on['score_pct']}%")
    print(f"\n  Desglose por guard:")
    print(f"    RestrictToTopic:  {bg['topic']['passed']}/{bg['topic']['total']} — {bg['topic']['score_pct']}%")
    print(f"    ToxicLanguage:    {bg['toxic']['passed']}/{bg['toxic']['total']} — {bg['toxic']['score_pct']}%")
    print(f"    DetectPII:        {bg['pii']['passed']}/{bg['pii']['total']} — {bg['pii']['score_pct']}%")
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


# ---------------------------------------------------------------------------
# Suite del clasificador de contexto (/llm/classify-context)
# ---------------------------------------------------------------------------

def run_classifier_test(
    client: httpx.Client,
    base_url: str,
    test: dict,
) -> dict:
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


def calculate_classifier_scores(results: list) -> dict:
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
    for ct in ("transaction", "question", "trends", "budget", "categories", "savings", "none"):
        subset = [r for r in valid if r.get("expected_context_type") == ct]
        if not subset:
            continue
        p = sum(1 for r in subset if r["passed"])
        by_type[ct] = {
            "passed": p,
            "total": len(subset),
            "score_pct": round(p / len(subset) * 100, 1),
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


def print_classifier_summary(model_name: str, scores: dict):
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


def save_classifier_results(model_name: str, results: list, scores: dict):
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
        description="Evaluación del topic guard para el servicio financiero LLM"
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

    data = load_test_cases()
    financial_context = data["financial_context"]
    # Solo ejecutar tests de la categoría guardrail
    test_cases = [t for t in data["test_cases"] if t["category"] == "guardrail"]

    print(f"\n Verificando servicio en {args.base_url}...")
    try:
        with httpx.Client() as client:
            health = client.get(f"{args.base_url}/health", timeout=5.0)
            health.raise_for_status()
            print(f" Servicio activo: {health.json()}")
    except Exception as e:
        print(f" ERROR: No se puede conectar al servicio: {e}")
        print(f" Asegúrate de que el servicio esté corriendo en: {args.base_url}")
        sys.exit(1)

    model_name = args.model
    if not model_name:
        try:
            with httpx.Client() as client:
                llm_health = client.get(f"{args.base_url}/health/llm", timeout=10.0)
                model_name = llm_health.json().get("provider", "unknown")
        except Exception:
            model_name = "unknown"

    # --- SMOKE TEST — verifica que el LLM responde antes de evaluar ---
    print(f"\n  Smoke test del LLM ({args.base_url}/llm/smoke-test)...")
    try:
        with httpx.Client() as client:
            smoke = client.get(f"{args.base_url}/llm/smoke-test", timeout=REQUEST_TIMEOUT)
            if smoke.status_code == 200:
                body = smoke.json()
                print(f"  LLM responde OK → {body.get('model_response', '')!r}")
            else:
                body = smoke.json() if smoke.headers.get("content-type", "").startswith("application/json") else {}
                print(f"  SMOKE TEST FALLÓ — HTTP {smoke.status_code}")
                print(f"  Error: {body.get('error', '')} — {body.get('message', '')}")
                print("  Abortando evaluación: el modelo no está disponible o no responde.")
                sys.exit(1)
    except httpx.TimeoutException:
        print("  SMOKE TEST TIMEOUT — el modelo no respondió en el tiempo límite.")
        sys.exit(1)
    except Exception as e:
        print(f"  SMOKE TEST ERROR — {type(e).__name__}: {e}")
        sys.exit(1)

    off_count = sum(1 for t in test_cases if t.get("subcategory") == "off_topic")
    on_count = sum(1 for t in test_cases if t.get("subcategory") == "on_topic")
    topic_count = sum(1 for t in test_cases if t.get("guardrail_type") == "topic")
    toxic_count = sum(1 for t in test_cases if t.get("guardrail_type") == "toxic")
    pii_count = sum(1 for t in test_cases if t.get("guardrail_type") == "pii")

    print(f"\n  Modelo: {model_name}")
    print(f"  Tests totales: {len(test_cases)} ({off_count} off_topic, {on_count} on_topic)")
    print(f"  Por guard: RestrictToTopic={topic_count}, ToxicLanguage={toxic_count}, DetectPII={pii_count}")
    print()

    results = []
    with httpx.Client() as client:
        print(f"--- GUARDRAIL ({len(test_cases)} tests) ---")
        for test in test_cases:
            result = run_single_test(client, args.base_url, financial_context, test)
            results.append(result)
        print()

    scores = calculate_scores(results)
    print_summary(model_name, scores)
    save_results(model_name, results, scores)

    # --- Classifier suite ---
    classifier_cases = load_classifier_test_cases()
    classifier_results = []
    with httpx.Client() as client:
        print(f"--- CLASIFICADOR DE CONTEXTO ({len(classifier_cases)} tests) ---")
        for test in classifier_cases:
            result = run_classifier_test(client, args.base_url, test)
            classifier_results.append(result)
        print()

    classifier_scores = calculate_classifier_scores(classifier_results)
    print_classifier_summary(model_name, classifier_scores)
    save_classifier_results(model_name, classifier_results, classifier_scores)


if __name__ == "__main__":
    main()
