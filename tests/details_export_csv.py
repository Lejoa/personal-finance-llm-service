"""
details_export_csv.py
Genera CSV detallados a partir de los archivos de resultados en tests/results/.
Todos los archivos (full y smoke) se encuentran en la misma raíz.

  - Full tests:  {model}_{ts}.json, classifier_{ts}.json, quality_{ts}.json
  - Smoke tests: smoke_guardrail_{model}_{ts}.json, smoke_classifier_{model}_{ts}.json,
                 smoke_quality_{suite}_{judge}_{ts}.json

Archivos CSV generados:
  1. details_guardrail.csv   — una fila por caso de guardrail evaluado
  2. details_classifier.csv  — una fila por caso de clasificador evaluado
  3. details_quality.csv     — una fila por (caso × métrica) evaluado

Uso:
  python tests/details_export_csv.py              # full + smoke
  python tests/details_export_csv.py --smoke      # solo smoke
  python tests/details_export_csv.py --full       # solo full (tests completos)
  python tests/details_export_csv.py --results-dir tests/results --output-dir tests/results
"""

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Columnas
# ---------------------------------------------------------------------------

GUARDRAIL_COLS = [
    "suite",
    "modelo",
    "fecha_evaluacion",
    "caso_id",
    "guardrail_type",
    "subcategory",
    "description",
    "question",
    "passed",
    "status_code",
    "response_time_ms",
    "evaluation",
    "error",
    "error_detail_type",
    "error_detail_message",
]

CLASSIFIER_COLS = [
    "suite",
    "modelo",
    "fecha_evaluacion",
    "caso_id",
    "description",
    "message",
    "expected_context_type",
    "actual_context_type",
    "passed",
    "status_code",
    "response_time_ms",
]

QUALITY_COLS = [
    "suite",
    "modelo",
    "fecha_evaluacion",
    "juez",
    "caso_id",
    "description",
    "input",
    "expected_output",
    "actual_output",
    "llm_time_ms",
    "judge_time_ms",
    "total_time_ms",
    "caso_passed",
    "metric",
    "metric_score",
    "metric_threshold",
    "metric_passed",
    "metric_reason",
    "metric_error",
    "metric_judge_time_ms",
]

METRIC_CLASS_TO_KEY = {
    "AnswerRelevancyMetric":     "answer_relevancy",
    "FaithfulnessMetric":        "faithfulness",
    "ContextualRelevancyMetric": "contextual_relevancy",
    "ContextualRecallMetric":    "contextual_recall",
    "ContextualPrecisionMetric": "contextual_precision",
}

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


def _bool_str(v) -> str:
    """Normaliza passed que puede llegar como bool, 'True'/'False', o str."""
    if isinstance(v, bool):
        return "SI" if v else "NO"
    if isinstance(v, str):
        return "SI" if v.strip().lower() == "true" else "NO"
    return ""


def _parse_dict(v):
    """Deserializa un dict que puede haber sido serializado como string."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return {}
    return {}


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Selección de archivos por modo
# ---------------------------------------------------------------------------

def _collect_files(results_dir: Path, mode: str, patterns: list[str]) -> list[Path]:
    """
    Devuelve los archivos de results_dir que coincidan con los patrones
    y sean del tipo correcto según el modo (full, smoke, all).
    """
    files = []
    seen: set[Path] = set()

    for pattern in patterns:
        for jpath in sorted(results_dir.glob(pattern)):
            if jpath.resolve() in seen:
                continue
            seen.add(jpath.resolve())
            is_smoke = jpath.name.startswith("smoke_")
            if mode == "smoke" and not is_smoke:
                continue
            if mode == "full" and is_smoke:
                continue
            files.append(jpath)

    return files


# ---------------------------------------------------------------------------
# Guardrail — tests completos
# ---------------------------------------------------------------------------

def _guardrail_row(data: dict, r: dict) -> dict:
    evaluation = _parse_dict(r.get("evaluation", {}))
    ed = r.get("error_detail") or {}
    return {
        "suite": data.get("suite", "guardrail"),
        "modelo": data.get("model", ""),
        "fecha_evaluacion": data.get("timestamp", ""),
        "caso_id": r.get("id", ""),
        "guardrail_type": r.get("guardrail_type", ""),
        "subcategory": r.get("subcategory", ""),
        "description": r.get("description", ""),
        "question": r.get("question", ""),
        "passed": _bool_str(evaluation.get("passed")),
        "status_code": _fmt(r.get("status_code")),
        "response_time_ms": _fmt(r.get("response_time_ms")),
        "evaluation": evaluation.get("evaluation", ""),
        "error": r.get("error", ""),
        "error_detail_type": ed.get("error", ""),
        "error_detail_message": ed.get("message", ""),
    }


def extract_full_guardrail_rows(results_dir: Path) -> list[dict]:
    """Extrae filas de guardrail de los tests completos ({model}_{ts}.json)."""
    rows = []
    # Archivos full de guardrail: no empiezan con prefijos conocidos de otras suites
    excluded = ("quality_", "multi_turn_", "comparison_", "classifier_",
                "smoke_", "details_")
    for jpath in sorted(results_dir.glob("*.json")):
        if any(jpath.name.startswith(p) for p in excluded):
            continue
        try:
            data = load_json(jpath)
        except Exception as e:
            print(f"  WARN: {jpath.name}: {e}")
            continue
        if "results" not in data:
            continue
        for r in data["results"]:
            rows.append(_guardrail_row(data, r))
    return rows


def extract_smoke_guardrail_rows(results_dir: Path) -> list[dict]:
    """Extrae filas de guardrail de los smoke tests (smoke_guardrail_*.json)."""
    rows = []
    for jpath in _collect_files(results_dir, "smoke", ["smoke_guardrail_*.json"]):
        try:
            data = load_json(jpath)
        except Exception as e:
            print(f"  WARN: {jpath.name}: {e}")
            continue
        if "results" not in data:
            continue
        for r in data["results"]:
            rows.append(_guardrail_row(data, r))
    return rows


# ---------------------------------------------------------------------------
# Classifier — tests completos y smoke
# ---------------------------------------------------------------------------

def _classifier_row(data: dict, r: dict) -> dict:
    return {
        "suite": data.get("suite", "classifier"),
        "modelo": data.get("model", ""),
        "fecha_evaluacion": data.get("timestamp", ""),
        "caso_id": r.get("id", ""),
        "description": r.get("description", ""),
        "message": r.get("message", ""),
        "expected_context_type": r.get("expected_context_type", ""),
        "actual_context_type": r.get("actual_context_type", ""),
        "passed": _bool_str(r.get("passed")),
        "status_code": _fmt(r.get("status_code")),
        "response_time_ms": _fmt(r.get("response_time_ms")),
    }


def extract_full_classifier_rows(results_dir: Path) -> list[dict]:
    """Extrae filas de clasificador de los tests completos (classifier_*.json)."""
    rows = []
    for jpath in _collect_files(results_dir, "full", ["classifier_*.json"]):
        try:
            data = load_json(jpath)
        except Exception as e:
            print(f"  WARN: {jpath.name}: {e}")
            continue
        if "results" not in data:
            continue
        for r in data["results"]:
            rows.append(_classifier_row(data, r))
    return rows


def extract_smoke_classifier_rows(results_dir: Path) -> list[dict]:
    """Extrae filas de clasificador de los smoke tests (smoke_classifier_*.json)."""
    rows = []
    for jpath in _collect_files(results_dir, "smoke", ["smoke_classifier_*.json"]):
        try:
            data = load_json(jpath)
        except Exception as e:
            print(f"  WARN: {jpath.name}: {e}")
            continue
        if "results" not in data:
            continue
        for r in data["results"]:
            rows.append(_classifier_row(data, r))
    return rows


# ---------------------------------------------------------------------------
# Quality — tests completos y smoke
# ---------------------------------------------------------------------------

def _quality_rows_from_file(data: dict) -> list[dict]:
    suite = data.get("suite", "quality")
    model = data.get("model", "unknown")
    judge = data.get("judge", "")
    timestamp = data.get("timestamp", "")
    rows = []

    for r in data.get("results", []):
        metric_results = _parse_dict(r.get("metric_results", {}))
        base = {
            "suite": suite,
            "modelo": model,
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
            "caso_passed": _bool_str(r.get("passed")),
        }

        if not metric_results:
            rows.append({
                **base,
                "metric": "",
                "metric_score": "",
                "metric_threshold": "",
                "metric_passed": "NO",
                "metric_reason": r.get("error", "Sin respuesta del servicio"),
                "metric_error": r.get("error", ""),
                "metric_judge_time_ms": "",
            })
            continue

        for class_name, mdata in metric_results.items():
            rows.append({
                **base,
                "metric": METRIC_CLASS_TO_KEY.get(class_name, class_name),
                "metric_score": _fmt(mdata.get("score")),
                "metric_threshold": _fmt(mdata.get("threshold")),
                "metric_passed": _bool_str(mdata.get("passed")),
                "metric_reason": mdata.get("reason", ""),
                "metric_error": mdata.get("error", ""),
                "metric_judge_time_ms": _fmt(mdata.get("judge_time_ms")),
            })

    return rows


def extract_full_quality_rows(results_dir: Path) -> list[dict]:
    """Extrae filas de quality de los tests completos (quality_*.json)."""
    rows = []
    for jpath in _collect_files(results_dir, "full", ["quality_*.json"]):
        try:
            data = load_json(jpath)
        except Exception as e:
            print(f"  WARN: {jpath.name}: {e}")
            continue
        rows.extend(_quality_rows_from_file(data))
    return rows


def extract_smoke_quality_rows(results_dir: Path) -> list[dict]:
    """Extrae filas de quality de los smoke tests (smoke_quality_*.json)."""
    rows = []
    for jpath in _collect_files(results_dir, "smoke", ["smoke_quality_*.json"]):
        try:
            data = load_json(jpath)
        except Exception as e:
            print(f"  WARN: {jpath.name}: {e}")
            continue
        rows.extend(_quality_rows_from_file(data))
    return rows


# ---------------------------------------------------------------------------
# Agregadores por modo
# ---------------------------------------------------------------------------

def collect_guardrail_rows(results_dir: Path, mode: str) -> list[dict]:
    rows = []
    if mode in ("all", "full"):
        rows.extend(extract_full_guardrail_rows(results_dir))
    if mode in ("all", "smoke"):
        rows.extend(extract_smoke_guardrail_rows(results_dir))
    return rows


def collect_classifier_rows(results_dir: Path, mode: str) -> list[dict]:
    rows = []
    if mode in ("all", "full"):
        rows.extend(extract_full_classifier_rows(results_dir))
    if mode in ("all", "smoke"):
        rows.extend(extract_smoke_classifier_rows(results_dir))
    return rows


def collect_quality_rows(results_dir: Path, mode: str) -> list[dict]:
    rows = []
    if mode in ("all", "full"):
        rows.extend(extract_full_quality_rows(results_dir))
    if mode in ("all", "smoke"):
        rows.extend(extract_smoke_quality_rows(results_dir))
    return rows


# ---------------------------------------------------------------------------
# Impresión de resumen en consola
# ---------------------------------------------------------------------------

def _print_pass_summary(title: str, rows: list[dict]):
    by_model: dict[str, list] = defaultdict(list)
    for r in rows:
        key = f"{r['suite']}/{r['modelo']}"
        by_model[key].append(r)

    print(f"\n=== {title} — Resumen por modelo ===")
    print(f"  {'Suite/Modelo':<50} {'Total':>6} {'Pass':>6} {'%':>7}")
    print("  " + "-" * 72)
    for key in sorted(by_model):
        cases = by_model[key]
        total = len(cases)
        passed = sum(1 for c in cases if c["passed"] == "SI")
        pct = passed / total * 100 if total else 0
        print(f"  {key:<50} {total:>6} {passed:>6} {pct:>6.1f}%")


def print_quality_summary(rows: list[dict]):
    by_model_metric: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if not r["metric"]:
            continue
        key = (f"{r['suite']}/{r['modelo']}", r["metric"])
        by_model_metric[key].append(r)

    models = sorted({k[0] for k in by_model_metric})
    metrics = [
        "answer_relevancy", "faithfulness", "contextual_relevancy",
        "contextual_recall", "contextual_precision",
    ]

    print("\n=== QUALITY — Resumen por modelo × métrica (avg score) ===")
    header = f"  {'Suite/Modelo':<45}" + "".join(f" {m[:12]:>13}" for m in metrics)
    print(header)
    print("  " + "-" * (45 + 13 * len(metrics)))
    for model in models:
        line = f"  {model:<45}"
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
        help=f"Directorio raíz de resultados (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directorio donde se guardan los CSV (default: {DEFAULT_RESULTS_DIR})",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--smoke",
        action="store_true",
        help="Procesar solo archivos smoke (smoke_*.json)",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Procesar solo archivos de tests completos",
    )
    args = parser.parse_args()

    mode = "smoke" if args.smoke else ("full" if args.full else "all")
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modo: {mode.upper()}")
    print(f"Buscando resultados en: {results_dir}")

    # --- Guardrail ---
    g_rows = collect_guardrail_rows(results_dir, mode)
    g_path = output_dir / "details_guardrail.csv"
    with open(g_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=GUARDRAIL_COLS)
        w.writeheader()
        w.writerows(g_rows)
    print(f"\nGuardrail CSV:   {g_path}  ({len(g_rows)} filas)")
    if g_rows:
        _print_pass_summary("GUARDRAIL", g_rows)

    # --- Classifier ---
    c_rows = collect_classifier_rows(results_dir, mode)
    c_path = output_dir / "details_classifier.csv"
    with open(c_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CLASSIFIER_COLS)
        w.writeheader()
        w.writerows(c_rows)
    print(f"\nClassifier CSV:  {c_path}  ({len(c_rows)} filas)")
    if c_rows:
        _print_pass_summary("CLASSIFIER", c_rows)

    # --- Quality ---
    q_rows = collect_quality_rows(results_dir, mode)
    q_path = output_dir / "details_quality.csv"
    with open(q_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=QUALITY_COLS)
        w.writeheader()
        w.writerows(q_rows)
    print(f"\nQuality CSV:     {q_path}  ({len(q_rows)} filas)")
    if q_rows:
        print_quality_summary(q_rows)

    print()


if __name__ == "__main__":
    main()