"""
Versión smoke (≈25%) de test_llm_quality.py.

Ejecuta solo los casos indicados por --base-ids y --enriched-ids.
Reutiliza toda la lógica de test_llm_quality.py sin duplicar código.

Uso:
  python tests/test_llm_quality_smoke.py \\
    --base-url http://llm-service:8000 \\
    --base-ids Q5,Q3 \\
    --enriched-ids CE1,CE3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from test_llm_quality import (
    OllamaCloudJudge,
    load_test_cases,
    load_enriched_test_cases,
    run_suite,
    run_single_test,
    calculate_scores,
    print_summary,
    RESULTS_DIR,
    REQUEST_TIMEOUT,
)

DEFAULT_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

SMOKE_BASE_IDS     = {"Q5", "Q3"}
SMOKE_ENRICHED_IDS = {"CE1", "CE3"}


def save_smoke_results(label: str, judge_name: str, results: list, scores: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = judge_name.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / f"smoke_quality_{label}_{safe}_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "suite": f"smoke_quality_{label}",
                "judge": judge_name,
                "model": os.getenv("LLM_MODEL", "unknown"),
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
        description="Smoke quality evaluation (≈25%)"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--base-ids",
        default=",".join(SMOKE_BASE_IDS),
        help="IDs de la suite base separados por coma (default: %(default)s)",
    )
    parser.add_argument(
        "--enriched-ids",
        default=",".join(SMOKE_ENRICHED_IDS),
        help="IDs de la suite enriquecida separados por coma (default: %(default)s)",
    )
    args = parser.parse_args()

    base_ids     = set(args.base_ids.split(","))
    enriched_ids = set(args.enriched_ids.split(","))
    base_url     = args.base_url

    # Health check
    print(f"\n  Verificando servicio en {base_url}...")
    try:
        with httpx.Client() as client:
            health = client.get(f"{base_url}/health", timeout=5.0)
            health.raise_for_status()
            print(f"  Servicio activo: {health.json()}")
    except Exception as e:
        print(f"  ERROR: No se puede conectar al servicio: {e}")
        sys.exit(1)

    judge = OllamaCloudJudge()
    judge_name = judge.get_model_name()
    print(f"\n  Juez: {judge_name}")
    print(f"  Base IDs:     {', '.join(sorted(base_ids))}")
    print(f"  Enriched IDs: {', '.join(sorted(enriched_ids))}")

    with httpx.Client() as client:
        # --- Suite base (subconjunto) ---
        data = load_test_cases()
        base_cases = [t for t in data["test_cases"] if t["id"] in base_ids]
        print(f"\n  Quality base smoke: {len(base_cases)} casos\n")

        results_base, wall_base = run_suite(
            client, base_url,
            data["financial_context"], base_cases,
            judge, "CALIDAD BASE SMOKE",
        )
        scores_base = calculate_scores(results_base, wall_base)
        print_summary(f"{judge_name} [base-smoke]", scores_base)
        save_smoke_results("base", judge_name, results_base, scores_base)

        # --- Suite enriquecida (subconjunto) ---
        enriched = load_enriched_test_cases()
        enriched_cases = [t for t in enriched["test_cases"] if t["id"] in enriched_ids]
        print(f"\n  Quality enriched smoke: {len(enriched_cases)} casos\n")

        results_enriched, wall_enriched = run_suite(
            client, base_url,
            enriched["financial_context"], enriched_cases,
            judge, "CALIDAD ENRIQUECIDA SMOKE",
        )
        scores_enriched = calculate_scores(results_enriched, wall_enriched)
        print_summary(f"{judge_name} [enriched-smoke]", scores_enriched)
        save_smoke_results("enriched", judge_name, results_enriched, scores_enriched)


if __name__ == "__main__":
    main()
