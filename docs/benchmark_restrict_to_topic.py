"""
Benchmark de modelos zero-shot para RestrictToTopic.

Evalúa 5 modelos de HuggingFace con 8 casos de prueba (4 ALLOW + 4 BLOCK),
midiendo latencia de carga (cold start) y latencia por inferencia.

Ejecutar dentro del contenedor llm-service:
    docker compose -f personal-finance-infra/docker-compose.dev.yml \\
        exec -e PYTHONPATH=/app llm-service \\
        python /tmp/benchmark_restrict_to_topic.py

Los modelos se descargan desde HuggingFace Hub (~primera ejecución).
Los resultados se guardan en /app/tests/results/benchmark_restrict_<ts>.json
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/app")

from guardrails import Guard
from guardrails.hub import RestrictToTopic

# ── Topics (idénticos a guardrails_service.py) ────────────────────────────────

FINANCIAL_TOPICS = [
    "personal finance",
    "savings",
    "investment",
    "budget",
    "expenses",
    "income",
    "debt",
    "loans",
    "financial education",
    "cryptocurrency",
    "stock market",
    "money management",
    "spending money",
    "buying something",
    "purchasing goods",
    "paying for something",
    "expense tracking",
    "recording a transaction",
    "logging an expense",
    "logging income",
    "received payment",
    "earned money",
    "spent money",
    "paid for",
    "transaction record",
    "financial transaction",
    "purchase amount",
    "money spent",
    "money received",
]

INVALID_TOPICS = [
    "cooking recipes",
    "sports scores",
    "politics",
    "software programming",
    "entertainment news",
]

# ── Modelos a evaluar ──────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "cross-encoder/nli-deberta-v3-small",
        "label": "DeBERTa-v3-small (baseline actual)",
        "type": "cross-encoder",
    },
    {
        "name": "cross-encoder/nli-MiniLM2-L6-H768",
        "label": "MiniLM2-L6 (cross-encoder ligero)",
        "type": "cross-encoder",
    },
    {
        "name": "typeform/distilbart-mnli-12-3",
        "label": "DistilBART-mnli-12-3 (Typeform)",
        "type": "zero-shot-classification",
    },
    {
        "name": "valhalla/distilbart-mnli-12-3",
        "label": "DistilBART-mnli-12-3 (Valhalla)",
        "type": "zero-shot-classification",
    },
    {
        "name": "moritzlaurer/mDeBERTa-v3-base-mnli-xnli",
        "label": "mDeBERTa-v3-base multilingual",
        "type": "zero-shot-classification",
    },
]

# ── Casos de prueba ───────────────────────────────────────────────────────────

TEST_CASES = [
    # ALLOW — queries financieras en español
    {
        "id": "A1",
        "text": "¿Cómo puedo organizar mi presupuesto mensual y reducir mis gastos?",
        "lang": "ES",
        "expect": "ALLOW",
    },
    {
        "id": "A2",
        "text": "Gasté 45 000 pesos en el mercado hoy",
        "lang": "ES",
        "expect": "ALLOW",
    },
    # ALLOW — queries financieras en inglés
    {
        "id": "A3",
        "text": "How should I allocate my savings across different investment funds?",
        "lang": "EN",
        "expect": "ALLOW",
    },
    {
        "id": "A4",
        "text": "What is the difference between a savings account and a CD?",
        "lang": "EN",
        "expect": "ALLOW",
    },
    # BLOCK — temas inválidos en español
    {
        "id": "B1",
        "text": "¿Cuál es la mejor receta para hacer pasta carbonara?",
        "lang": "ES",
        "expect": "BLOCK",
        "invalid_topic": "cooking recipes",
    },
    {
        "id": "B2",
        "text": "¿Quién ganó el partido de fútbol anoche?",
        "lang": "ES",
        "expect": "BLOCK",
        "invalid_topic": "sports scores",
    },
    # BLOCK — temas inválidos en inglés
    {
        "id": "B3",
        "text": "What's the latest news in Hollywood?",
        "lang": "EN",
        "expect": "BLOCK",
        "invalid_topic": "entertainment news",
    },
    {
        "id": "B4",
        "text": "How do I write a Python function?",
        "lang": "EN",
        "expect": "BLOCK",
        "invalid_topic": "software programming",
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_model_benchmark(model_info: dict) -> dict:
    model_name = model_info["name"]
    label = model_info["label"]

    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"  [{model_name}]")
    print(f"{'=' * 65}")

    # Cold start — medir carga del modelo
    print("  Cargando modelo...", end=" ", flush=True)
    t_load = time.time()
    try:
        guard = Guard().use(
            RestrictToTopic(
                valid_topics=FINANCIAL_TOPICS,
                invalid_topics=INVALID_TOPICS,
                disable_classifier=False,
                disable_llm=True,
                model=model_name,
                on_fail="exception",
            )
        )
        load_time = time.time() - t_load
        print(f"OK ({load_time:.1f}s)")
        load_error = None
    except Exception as e:
        load_time = time.time() - t_load
        load_error = str(e)[:300]
        print(f"ERROR: {load_error}")
        return {
            "model": model_name,
            "label": label,
            "load_time_s": round(load_time, 2),
            "load_error": load_error,
            "cases": [],
            "accuracy": 0,
            "avg_latency_s": None,
            "avg_allow_s": None,
            "avg_block_s": None,
        }

    # Warm-up pass (primera inferencia real — carga pesos del modelo)
    print("  Warm-up...", end=" ", flush=True)
    t_warm = time.time()
    try:
        guard.validate("How much should I save per month?")
        warmup_time = time.time() - t_warm
        print(f"OK ({warmup_time:.1f}s)")
    except Exception:
        warmup_time = time.time() - t_warm
        print(f"done ({warmup_time:.1f}s)")

    # Ejecutar los 8 casos de prueba
    cases = []
    print()
    for tc in TEST_CASES:
        t = time.time()
        try:
            guard.validate(tc["text"])
            actual = "ALLOW"
            error = None
        except Exception as e:
            actual = "BLOCK"
            error = str(e)[:150]
        elapsed = round(time.time() - t, 2)

        correct = actual == tc["expect"]
        mark = "✅" if correct else "❌"
        print(f"  {mark} [{tc['id']}] {tc['lang']} {tc['expect']:5s} → {actual:5s} ({elapsed:.2f}s)  {tc['text'][:60]}")

        cases.append({
            "id": tc["id"],
            "lang": tc["lang"],
            "text": tc["text"],
            "expected": tc["expect"],
            "actual": actual,
            "correct": correct,
            "elapsed_s": elapsed,
            "error": error,
        })

    # Métricas agregadas
    correct_count = sum(1 for c in cases if c["correct"])
    all_latencies = [c["elapsed_s"] for c in cases]
    allow_latencies = [c["elapsed_s"] for c in cases if c["expected"] == "ALLOW"]
    block_latencies = [c["elapsed_s"] for c in cases if c["expected"] == "BLOCK"]

    avg = lambda lst: round(sum(lst) / len(lst), 2) if lst else None

    result = {
        "model": model_name,
        "label": label,
        "type": model_info["type"],
        "load_time_s": round(load_time, 2),
        "warmup_time_s": round(warmup_time, 2),
        "load_error": None,
        "cases": cases,
        "accuracy": f"{correct_count}/{len(cases)}",
        "accuracy_pct": round(correct_count / len(cases) * 100, 1),
        "avg_latency_s": avg(all_latencies),
        "avg_allow_s": avg(allow_latencies),
        "avg_block_s": avg(block_latencies),
        "max_latency_s": round(max(all_latencies), 2),
    }

    print(f"\n  Accuracy: {result['accuracy']}  "
          f"avg={result['avg_latency_s']}s  "
          f"allow={result['avg_allow_s']}s  "
          f"block={result['avg_block_s']}s  "
          f"max={result['max_latency_s']}s")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nBenchmark RestrictToTopic — modelo zero-shot")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Modelos a evaluar: {len(MODELS)}")
    print(f"Casos de prueba: {len(TEST_CASES)} (4 ALLOW + 4 BLOCK)")

    results = []
    for model_info in MODELS:
        result = run_model_benchmark(model_info)
        results.append(result)

    # Tabla comparativa final
    print(f"\n\n{'=' * 80}")
    print("TABLA COMPARATIVA")
    print(f"{'=' * 80}")
    header = f"{'Modelo':<45} {'Acc':>5} {'Avg':>6} {'ALLOW':>6} {'BLOCK':>6} {'Max':>6} {'Load':>6}"
    print(header)
    print("-" * 80)
    for r in results:
        if r.get("load_error"):
            print(f"  {r['label']:<43} ERROR: {r['load_error'][:40]}")
        else:
            acc = r["accuracy_pct"]
            flag = "✅" if acc == 100 and r["avg_latency_s"] < 3 else ("⚠" if acc == 100 else "❌")
            print(
                f"{flag} {r['label']:<43} "
                f"{r['accuracy']:>5}  "
                f"{r['avg_latency_s']:>5.2f}s "
                f"{r['avg_allow_s']:>5.2f}s "
                f"{r['avg_block_s']:>5.2f}s "
                f"{r['max_latency_s']:>5.2f}s "
                f"{r['load_time_s']:>5.1f}s"
            )
    print(f"{'=' * 80}")
    print("Criterio: accuracy 8/8 Y latencia avg < 3s")

    # Guardar JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("/app/tests/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"benchmark_restrict_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "models_tested": len(MODELS),
                "test_cases": len(TEST_CASES),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResultados guardados en: {out_path}")

    # Recomendación automática
    candidates = [
        r for r in results
        if not r.get("load_error")
        and r["accuracy_pct"] == 100
        and r["avg_latency_s"] is not None
        and r["avg_latency_s"] < 3
    ]
    if candidates:
        winner = min(candidates, key=lambda r: r["avg_latency_s"])
        print(f"\n🏆 Modelo recomendado: {winner['model']}")
        print(f"   accuracy=8/8  avg={winner['avg_latency_s']}s  max={winner['max_latency_s']}s")
    else:
        print("\n⚠ Ningún modelo cumple accuracy=8/8 Y avg<3s. Revisar resultados.")


if __name__ == "__main__":
    main()
