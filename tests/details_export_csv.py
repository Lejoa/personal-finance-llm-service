"""
details_export_csv.py
Genera dos CSV detallados a partir de los archivos de resultados individuales
encontrados en los subdirectorios de tests/results/:

  1. details_guardrail.csv — una fila por caso de guardrail evaluado
  2. details_quality.csv   — una fila por (caso × métrica) evaluado

Uso:
  python tests/details_export_csv.py
  python tests/details_export_csv.py --results-dir tests/results --output-dir tests/results
"""

import argparse
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Columnas
# ---------------------------------------------------------------------------

GUARDRAIL_COLS = [
    "modelo",
    "carpeta",
    "fecha_evaluacion",
    # Caso
    "caso_id",
    "guardrail_type",
    "subcategory",
    "description",
    "question",
    # Resultado
    "passed",
    "status_code",
    "response_time_ms",
    "evaluation",
    "error",
    "error_detail_type",
    "error_detail_message",
]

QUALITY_COLS = [
    "modelo",
    "carpeta",
    "fecha_evaluacion",
    "juez",
    # Caso
    "caso_id",
    "description",
    "input",
    "expected_output",
    "actual_output",
    # Tiempos
    "llm_time_ms",
    "judge_time_ms",
    "total_time_ms",
    # Caso overall
    "caso_passed",
    # Métrica
    "metric",
    "metric_score",
    "metric_threshold",
    "metric_passed",
    "metric_reason",
    "metric_error",
    "metric_judge_time_ms",
]

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, bool):
        return "SI" if v else "NO"
    return str(v)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Guardrail rows
# ---------------------------------------------------------------------------

def _scan_dirs(results_dir: Path) -> list[tuple[Path, str]]:
    """
    Devuelve únicamente la raíz de results_dir para escanear.
    Los subdirectorios históricos (gemini/, gpt-oss-20-120/, qwen/, etc.) se ignoran.
    """
    return [(results_dir, "results")]


def extract_guardrail_rows(results_dir: Path) -> list[dict]:
    rows = []
    seen = set()  # evitar duplicados si un archivo aparece en raíz y subdir

    for folder_dir, folder in _scan_dirs(results_dir):
        # Archivos de guardrail: NO comienzan con quality_, multi_turn_, comparison_
        for jpath in sorted(folder_dir.glob("*.json")):
            if jpath.name.startswith(("quality_", "multi_turn_", "comparison_")):
                continue
            if jpath.suffix != ".json" or jpath.name.startswith("."):
                continue
            if jpath.resolve() in seen:
                continue
            seen.add(jpath.resolve())
            try:
                data = load_json(jpath)
            except Exception as e:
                print(f"  WARN: no se pudo leer {jpath.name}: {e}")
                continue
            if "results" not in data:
                continue

            model = data.get("model", jpath.stem)
            timestamp = data.get("timestamp", "")

            for r in data.get("results", []):
                ed = r.get("error_detail") or {}
                row = {
                    "modelo": model,
                    "carpeta": folder,
                    "fecha_evaluacion": timestamp,
                    "caso_id": r.get("id", ""),
                    "guardrail_type": r.get("guardrail_type", ""),
                    "subcategory": r.get("subcategory", ""),
                    "description": r.get("description", ""),
                    "question": r.get("question", ""),
                    "passed": _fmt(r.get("evaluation", {}).get("passed")),
                    "status_code": _fmt(r.get("status_code")),
                    "response_time_ms": _fmt(r.get("response_time_ms")),
                    "evaluation": r.get("evaluation", {}).get("evaluation", ""),
                    "error": r.get("error", ""),
                    "error_detail_type": ed.get("error", ""),
                    "error_detail_message": ed.get("message", ""),
                }
                rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Quality rows (una fila por caso × métrica)
# ---------------------------------------------------------------------------

METRIC_CLASS_TO_KEY = {
    "AnswerRelevancyMetric":    "answer_relevancy",
    "FaithfulnessMetric":       "faithfulness",
    "ContextualRelevancyMetric":"contextual_relevancy",
    "ContextualRecallMetric":   "contextual_recall",
    "ContextualPrecisionMetric":"contextual_precision",
}


def extract_quality_rows(results_dir: Path) -> list[dict]:
    rows = []
    seen: set[Path] = set()

    for folder_dir, folder in _scan_dirs(results_dir):
        for jpath in sorted(folder_dir.glob("quality_*.json")):
            if jpath.resolve() in seen:
                continue
            seen.add(jpath.resolve())
            try:
                data = load_json(jpath)
            except Exception as e:
                print(f"  WARN: no se pudo leer {jpath.name}: {e}")
                continue

            model = data.get("model", "unknown")
            judge = data.get("judge", "")
            timestamp = data.get("timestamp", "")

            for r in data.get("results", []):
                base = {
                    "modelo": model,
                    "carpeta": folder,
                    "fecha_evaluacion": timestamp,
                    "juez": judge,
                    "caso_id": r.get("id", ""),
                    "description": r.get("description", ""),
                    "input": r.get("input", ""),
                    "expected_output": r.get("expected_output", ""),
                    "actual_output": r.get("actual_output", ""),
                    "llm_time_ms": _fmt(r.get("llm_time_ms", r.get("response_time_ms"))),
                    "judge_time_ms": _fmt(r.get("judge_time_ms")),
                    "total_time_ms": _fmt(r.get("total_time_ms")),
                    "caso_passed": _fmt(r.get("passed")),
                }

                metric_results = r.get("metric_results", {})

                if not metric_results:
                    # Caso con error: emitir una sola fila sin métrica
                    row = {**base,
                           "metric": "",
                           "metric_score": "",
                           "metric_threshold": "",
                           "metric_passed": "NO",
                           "metric_reason": r.get("error", "Sin respuesta del servicio"),
                           "metric_error": r.get("error", ""),
                           "metric_judge_time_ms": ""}
                    rows.append(row)
                    continue

                for class_name, mdata in metric_results.items():
                    metric_key = METRIC_CLASS_TO_KEY.get(class_name, class_name)
                    row = {
                        **base,
                        "metric": metric_key,
                        "metric_score": _fmt(mdata.get("score")),
                        "metric_threshold": _fmt(mdata.get("threshold")),
                        "metric_passed": _fmt(mdata.get("passed")),
                        "metric_reason": mdata.get("reason", ""),
                        "metric_error": mdata.get("error", ""),
                        "metric_judge_time_ms": _fmt(mdata.get("judge_time_ms")),
                    }
                    rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Impresión de resumen en consola
# ---------------------------------------------------------------------------

def print_guardrail_summary(rows: list[dict]):
    from collections import defaultdict, Counter
    by_model = defaultdict(list)
    for r in rows:
        key = f"{r['carpeta']}/{r['modelo']}"
        by_model[key].append(r)

    print("\n=== GUARDRAIL — Resumen por modelo ===")
    print(f"  {'Modelo':<40} {'Total':>6} {'Pass':>6} {'%':>7}")
    print("  " + "-" * 62)
    for key in sorted(by_model):
        cases = by_model[key]
        total = len(cases)
        passed = sum(1 for c in cases if c["passed"] == "SI")
        pct = passed / total * 100 if total else 0
        print(f"  {key:<40} {total:>6} {passed:>6} {pct:>6.1f}%")


def print_quality_summary(rows: list[dict]):
    from collections import defaultdict
    by_model_metric = defaultdict(list)
    for r in rows:
        if not r["metric"]:
            continue
        key = (f"{r['carpeta']}/{r['modelo']}", r["metric"])
        by_model_metric[key].append(r)

    models = sorted({k[0] for k in by_model_metric})
    metrics = ["answer_relevancy", "faithfulness", "contextual_relevancy",
               "contextual_recall", "contextual_precision"]

    print("\n=== QUALITY — Resumen por modelo × métrica (avg score) ===")
    header = f"  {'Modelo':<35}" + "".join(f" {m[:12]:>13}" for m in metrics)
    print(header)
    print("  " + "-" * (35 + 13 * len(metrics)))
    for model in models:
        line = f"  {model:<35}"
        for metric in metrics:
            subset = by_model_metric.get((model, metric), [])
            if not subset:
                line += f" {'N/A':>13}"
            else:
                scores = [float(r["metric_score"]) for r in subset if r["metric_score"]]
                avg = sum(scores) / len(scores) if scores else 0.0
                line += f" {avg:>13.4f}"
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exporta resultados detallados por caso a CSV"
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directorio raíz con subcarpetas de resultados (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directorio donde se guardan los CSV (default: {DEFAULT_RESULTS_DIR})",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Buscando resultados en: {results_dir}")

    # --- Guardrail ---
    g_rows = extract_guardrail_rows(results_dir)
    g_path = output_dir / "details_guardrail.csv"
    with open(g_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=GUARDRAIL_COLS)
        writer.writeheader()
        writer.writerows(g_rows)
    print(f"\nGuardrail CSV: {g_path}  ({len(g_rows)} filas)")
    print_guardrail_summary(g_rows)

    # --- Quality ---
    q_rows = extract_quality_rows(results_dir)
    q_path = output_dir / "details_quality.csv"
    with open(q_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=QUALITY_COLS)
        writer.writeheader()
        writer.writerows(q_rows)
    print(f"\nQuality CSV:   {q_path}  ({len(q_rows)} filas)")
    print_quality_summary(q_rows)

    print()


if __name__ == "__main__":
    main()