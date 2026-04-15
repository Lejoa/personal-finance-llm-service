"""
compare_results.py
Genera un reporte comparativo a partir de los JSON producidos por:
  - eval_models.py / eval_models_smoke.py  → guardrail + classifier
  - test_llm_quality.py / test_llm_quality_smoke.py → quality

Para cada modelo recoge la ejecución más reciente de cada suite y calcula
un score compuesto (50% guardrails + 50% quality).

Detección de modelos: automática — escanea results_dir y extrae los modelos
de los archivos JSON encontrados. No requiere lista hardcodeada.

Uso:
  python tests/compare_results.py                         # full + smoke, auto-detect modelos
  python tests/compare_results.py --smoke                 # solo archivos smoke_*
  python tests/compare_results.py --full                  # solo tests completos
  python tests/compare_results.py --models gemini-3-flash-preview:cloud deepseek-v3.2:cloud
  python tests/compare_results.py --results-dir tests/results --output tests/results/comparison_report.json
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "comparison_report.json"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


# ---------------------------------------------------------------------------
# Auto-detección de modelos presentes en results_dir
# ---------------------------------------------------------------------------

def discover_models(results_dir: Path, mode: str) -> list[str]:
    """
    Lee todos los JSON relevantes y extrae el campo 'model' único.
    mode: 'all' | 'smoke' | 'full'
    """
    models: set[str] = set()
    for jpath in sorted(results_dir.glob("*.json")):
        name = jpath.name
        is_smoke = name.startswith("smoke_")
        if mode == "smoke" and not is_smoke:
            continue
        if mode == "full" and is_smoke:
            continue
        if name.startswith(("comparison_", "details_")):
            continue
        try:
            data = load_json(jpath)
            m = data.get("model")
            if m:
                models.add(m)
        except Exception:
            continue
    return sorted(models)


# ---------------------------------------------------------------------------
# Búsqueda del archivo más reciente por tipo y modelo
# ---------------------------------------------------------------------------

def _latest_file(results_dir: Path, mode: str, patterns: list[str], model: str) -> Path | None:
    """
    Devuelve el archivo más reciente que:
    - coincida con alguno de los patrones glob
    - contenga 'model' == model en su JSON
    - sea del tipo correcto (smoke vs full según mode)
    """
    candidates: list[Path] = []
    for pattern in patterns:
        for jpath in sorted(results_dir.glob(pattern)):
            is_smoke = jpath.name.startswith("smoke_")
            if mode == "smoke" and not is_smoke:
                continue
            if mode == "full" and is_smoke:
                continue
            try:
                data = load_json(jpath)
                if data.get("model") == model:
                    candidates.append(jpath)
            except Exception:
                continue
    return candidates[-1] if candidates else None


def latest_guardrail_file(results_dir: Path, mode: str, model: str) -> Path | None:
    """
    Para full: busca {safe_model}_*.json (sin prefijos de otras suites).
    Para smoke: busca smoke_guardrail_*.json con model==model.
    """
    safe = _safe_name(model)
    excluded_prefixes = ("quality_", "multi_turn_", "comparison_", "classifier_",
                         "smoke_", "details_")

    if mode in ("all", "full"):
        for jpath in sorted(results_dir.glob(f"{safe}_*.json")):
            if any(jpath.name.startswith(p) for p in excluded_prefixes):
                continue
            try:
                data = load_json(jpath)
                if data.get("model") == model:
                    return jpath  # tomar el más reciente (sorted)
            except Exception:
                continue

    if mode in ("all", "smoke"):
        match = _latest_file(results_dir, "smoke", ["smoke_guardrail_*.json"], model)
        if match:
            return match

    return None


def latest_classifier_file(results_dir: Path, mode: str, model: str) -> Path | None:
    full_pattern = ["classifier_*.json"]
    smoke_pattern = ["smoke_classifier_*.json"]

    if mode == "smoke":
        return _latest_file(results_dir, "smoke", smoke_pattern, model)
    if mode == "full":
        return _latest_file(results_dir, "full", full_pattern, model)
    # all: preferir el más reciente entre ambos tipos
    f = _latest_file(results_dir, "full", full_pattern, model)
    s = _latest_file(results_dir, "smoke", smoke_pattern, model)
    candidates = [p for p in [f, s] if p is not None]
    return sorted(candidates)[-1] if candidates else None


def latest_quality_file(results_dir: Path, mode: str, model: str) -> Path | None:
    full_pattern = ["quality_*.json"]
    smoke_pattern = ["smoke_quality_*.json"]

    if mode == "smoke":
        return _latest_file(results_dir, "smoke", smoke_pattern, model)
    if mode == "full":
        return _latest_file(results_dir, "full", full_pattern, model)
    f = _latest_file(results_dir, "full", full_pattern, model)
    s = _latest_file(results_dir, "smoke", smoke_pattern, model)
    candidates = [p for p in [f, s] if p is not None]
    return sorted(candidates)[-1] if candidates else None


# ---------------------------------------------------------------------------
# Extracción de scores
# ---------------------------------------------------------------------------

def extract_guardrail(data: dict) -> dict:
    s = data.get("scores", {})
    g = s.get("guardrail", {})
    by_guard = g.get("by_guard", {})
    timing = s.get("timing", {})
    return {
        "overall_pct":   g.get("score_pct"),
        "topic_pct":     by_guard.get("topic", {}).get("score_pct"),
        "toxic_pct":     by_guard.get("toxic", {}).get("score_pct"),
        "pii_pct":       by_guard.get("pii", {}).get("score_pct"),
        "off_topic_pct": g.get("off_topic", {}).get("score_pct"),
        "on_topic_pct":  g.get("on_topic", {}).get("score_pct"),
        "avg_ms":        timing.get("avg_ms"),
        "p95_ms":        timing.get("p95_ms"),
        "source_file":   data.get("_source_file", ""),
    }


def extract_classifier(data: dict) -> dict:
    s = data.get("scores", {})
    c = s.get("classifier", {})
    by_type = c.get("by_type", {})
    timing = s.get("timing", {})
    return {
        "overall_pct": c.get("score_pct"),
        "by_type":     {t: v.get("score_pct") for t, v in by_type.items()},
        "avg_ms":      timing.get("avg_ms"),
        "p95_ms":      timing.get("p95_ms"),
        "source_file": data.get("_source_file", ""),
    }


def extract_quality(data: dict) -> dict:
    s = data.get("scores", {})
    ov = s.get("overall", {})
    pm = s.get("per_metric", {})
    timing = s.get("timing", {})

    def metric_avg(key: str):
        return pm.get(key, {}).get("avg_score")

    def metric_pass(key: str):
        return pm.get(key, {}).get("pass_rate_pct")

    return {
        "overall_pct":              ov.get("score_pct"),
        "answer_relevancy_avg":     metric_avg("answer_relevancy"),
        "faithfulness_avg":         metric_avg("faithfulness"),
        "contextual_relevancy_avg": metric_avg("contextual_relevancy"),
        "contextual_recall_avg":    metric_avg("contextual_recall"),
        "contextual_precision_avg": metric_avg("contextual_precision"),
        "answer_relevancy_pass":    metric_pass("answer_relevancy"),
        "faithfulness_pass":        metric_pass("faithfulness"),
        "avg_ms":      timing.get("avg_ms"),
        "p95_ms":      timing.get("p95_ms"),
        "judge":       data.get("judge", ""),
        "source_file": data.get("_source_file", ""),
    }


# ---------------------------------------------------------------------------
# Score compuesto
# ---------------------------------------------------------------------------

def composite_score(guardrail: dict, quality: dict) -> float | None:
    parts = []
    if guardrail.get("overall_pct") is not None:
        parts.append(guardrail["overall_pct"] / 100)
    if quality.get("overall_pct") is not None:
        parts.append(quality["overall_pct"] / 100)
    return round(sum(parts) / len(parts), 4) if parts else None


# ---------------------------------------------------------------------------
# Impresión de tabla comparativa
# ---------------------------------------------------------------------------

def print_report(report: dict):
    models = report["models_compared"]
    results = report["results"]

    col = 32
    mcol = 14

    header_row = f"{'Métrica':<{col}}" + "".join(f"{m[-mcol:]:>{mcol}}" for m in models)
    sep = "-" * len(header_row)

    def row(label: str, extractor, fmt=".1f"):
        vals = []
        for m in models:
            v = extractor(results.get(m, {}))
            vals.append(f"{v:{fmt}}" if v is not None else "N/A")
        return f"  {label:<{col-2}}" + "".join(f"{v:>{mcol}}" for v in vals)

    def section(title: str):
        print(f"\n  [{title}]")
        print(sep)

    print("\n" + "=" * len(header_row))
    print("  REPORTE COMPARATIVO DE MODELOS LLM — FINANZAS")
    print("=" * len(header_row))
    print(f"\n  {'':.<{col}}" + "".join(f"{m[-mcol:]:>{mcol}}" for m in models))
    print(sep)

    section("GUARDRAILS")
    print(row("Overall %",          lambda r: r.get("guardrail", {}).get("overall_pct")))
    print(row("  Topic guard %",    lambda r: r.get("guardrail", {}).get("topic_pct")))
    print(row("  Toxic guard %",    lambda r: r.get("guardrail", {}).get("toxic_pct")))
    print(row("  PII guard %",      lambda r: r.get("guardrail", {}).get("pii_pct")))
    print(row("  Off-topic det. %", lambda r: r.get("guardrail", {}).get("off_topic_pct")))
    print(row("  On-topic acc. %",  lambda r: r.get("guardrail", {}).get("on_topic_pct")))
    print(row("Avg response (ms)",  lambda r: r.get("guardrail", {}).get("avg_ms"), fmt=".0f"))

    section("CLASSIFIER")
    print(row("Overall %",           lambda r: r.get("classifier", {}).get("overall_pct")))
    for ctx_type in ("transaction", "question", "trends", "budget", "categories", "savings", "none"):
        print(row(f"  {ctx_type} %",
                  lambda r, t=ctx_type: r.get("classifier", {}).get("by_type", {}).get(t)))
    print(row("Avg response (ms)",   lambda r: r.get("classifier", {}).get("avg_ms"), fmt=".0f"))

    section("QUALITY (DeepEval)")
    print(row("Overall %",               lambda r: r.get("quality", {}).get("overall_pct")))
    print(row("  AnswerRelevancy avg",   lambda r: r.get("quality", {}).get("answer_relevancy_avg"), fmt=".4f"))
    print(row("  Faithfulness avg",      lambda r: r.get("quality", {}).get("faithfulness_avg"), fmt=".4f"))
    print(row("  CtxRelevancy avg",      lambda r: r.get("quality", {}).get("contextual_relevancy_avg"), fmt=".4f"))
    print(row("  CtxRecall avg",         lambda r: r.get("quality", {}).get("contextual_recall_avg"), fmt=".4f"))
    print(row("  CtxPrecision avg",      lambda r: r.get("quality", {}).get("contextual_precision_avg"), fmt=".4f"))
    print(row("Avg response (ms)",       lambda r: r.get("quality", {}).get("avg_ms"), fmt=".0f"))

    section("RANKING FINAL (score compuesto)")
    for i, m in enumerate(report["ranking"], 1):
        score = results.get(m, {}).get("composite_score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"    {i}. {m}  →  {score_str}")

    print("\n" + "=" * len(header_row) + "\n")


# ---------------------------------------------------------------------------
# Exportación a CSV consolidado (una fila por modelo)
# ---------------------------------------------------------------------------

SUMMARY_COLS = [
    # Identificación
    "ranking",
    "modelo",
    "modo",
    "fecha_evaluacion",
    "juez",
    # Score compuesto
    "composite_score",
    # Guardrail
    "guardrail_overall_pct",
    "guardrail_topic_pct",
    "guardrail_toxic_pct",
    "guardrail_pii_pct",
    "guardrail_off_topic_pct",
    "guardrail_on_topic_pct",
    "guardrail_avg_ms",
    "guardrail_p95_ms",
    # Classifier
    "classifier_overall_pct",
    "classifier_transaction_pct",
    "classifier_question_pct",
    "classifier_trends_pct",
    "classifier_budget_pct",
    "classifier_categories_pct",
    "classifier_savings_pct",
    "classifier_none_pct",
    "classifier_avg_ms",
    "classifier_p95_ms",
    # Quality
    "quality_overall_pct",
    "quality_answer_relevancy_avg",
    "quality_faithfulness_avg",
    "quality_contextual_relevancy_avg",
    "quality_contextual_recall_avg",
    "quality_contextual_precision_avg",
    "quality_answer_relevancy_pass_pct",
    "quality_faithfulness_pass_pct",
    "quality_avg_ms",
    "quality_p95_ms",
]


def _fmt_csv(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def export_summary_csv(report: dict, output_path: Path) -> None:
    """Genera un CSV con una fila por modelo con todos los scores consolidados."""
    rows = []
    ranking_list: list[str] = report["ranking"]
    results: dict = report["results"]

    for rank, model in enumerate(ranking_list, 1):
        r = results.get(model, {})
        g = r.get("guardrail", {})
        c = r.get("classifier", {})
        by_type = c.get("by_type", {})
        q = r.get("quality", {})

        row = {
            "ranking":              rank,
            "modelo":               model,
            "modo":                 report.get("mode", ""),
            "fecha_evaluacion":     report.get("generated_at", ""),
            "juez":                 report.get("judge_model", ""),
            "composite_score":      _fmt_csv(r.get("composite_score")),
            # Guardrail
            "guardrail_overall_pct":   _fmt_csv(g.get("overall_pct")),
            "guardrail_topic_pct":     _fmt_csv(g.get("topic_pct")),
            "guardrail_toxic_pct":     _fmt_csv(g.get("toxic_pct")),
            "guardrail_pii_pct":       _fmt_csv(g.get("pii_pct")),
            "guardrail_off_topic_pct": _fmt_csv(g.get("off_topic_pct")),
            "guardrail_on_topic_pct":  _fmt_csv(g.get("on_topic_pct")),
            "guardrail_avg_ms":        _fmt_csv(g.get("avg_ms")),
            "guardrail_p95_ms":        _fmt_csv(g.get("p95_ms")),
            # Classifier
            "classifier_overall_pct":    _fmt_csv(c.get("overall_pct")),
            "classifier_transaction_pct":_fmt_csv(by_type.get("transaction")),
            "classifier_question_pct":   _fmt_csv(by_type.get("question")),
            "classifier_trends_pct":     _fmt_csv(by_type.get("trends")),
            "classifier_budget_pct":     _fmt_csv(by_type.get("budget")),
            "classifier_categories_pct": _fmt_csv(by_type.get("categories")),
            "classifier_savings_pct":    _fmt_csv(by_type.get("savings")),
            "classifier_none_pct":       _fmt_csv(by_type.get("none")),
            "classifier_avg_ms":         _fmt_csv(c.get("avg_ms")),
            "classifier_p95_ms":         _fmt_csv(c.get("p95_ms")),
            # Quality
            "quality_overall_pct":                  _fmt_csv(q.get("overall_pct")),
            "quality_answer_relevancy_avg":          _fmt_csv(q.get("answer_relevancy_avg")),
            "quality_faithfulness_avg":              _fmt_csv(q.get("faithfulness_avg")),
            "quality_contextual_relevancy_avg":      _fmt_csv(q.get("contextual_relevancy_avg")),
            "quality_contextual_recall_avg":         _fmt_csv(q.get("contextual_recall_avg")),
            "quality_contextual_precision_avg":      _fmt_csv(q.get("contextual_precision_avg")),
            "quality_answer_relevancy_pass_pct":     _fmt_csv(q.get("answer_relevancy_pass")),
            "quality_faithfulness_pass_pct":         _fmt_csv(q.get("faithfulness_pass")),
            "quality_avg_ms":  _fmt_csv(q.get("avg_ms")),
            "quality_p95_ms":  _fmt_csv(q.get("p95_ms")),
        }
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"Comparativa CSV: {output_path}  ({len(rows)} modelos)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Genera reporte comparativo de modelos LLM")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directorio con los JSON de resultados (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Ruta del JSON de salida (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Modelos a comparar. Si se omite, se detectan automáticamente.",
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
    output_path = Path(args.output)

    models = args.models or discover_models(results_dir, mode)
    if not models:
        print("No se encontraron modelos en el directorio de resultados.")
        return

    print(f"Modo: {mode.upper()}")
    print(f"Modelos detectados: {', '.join(models)}")

    comparison: dict[str, dict] = {}

    for model in models:
        print(f"\nProcesando: {model}")

        # Guardrail
        gf = latest_guardrail_file(results_dir, mode, model)
        if gf:
            gdata = load_json(gf)
            gdata["_source_file"] = gf.name
            guardrail = extract_guardrail(gdata)
            print(f"  Guardrail:  {gf.name}")
        else:
            guardrail = {}
            print(f"  Guardrail:  (no encontrado)")

        # Classifier
        cf = latest_classifier_file(results_dir, mode, model)
        if cf:
            cdata = load_json(cf)
            cdata["_source_file"] = cf.name
            classifier = extract_classifier(cdata)
            print(f"  Classifier: {cf.name}")
        else:
            classifier = {}
            print(f"  Classifier: (no encontrado)")

        # Quality
        qf = latest_quality_file(results_dir, mode, model)
        if qf:
            qdata = load_json(qf)
            qdata["_source_file"] = qf.name
            quality = extract_quality(qdata)
            print(f"  Quality:    {qf.name}")
        else:
            quality = {}
            print(f"  Quality:    (no encontrado)")

        cs = composite_score(guardrail, quality)
        comparison[model] = {
            "guardrail":       guardrail,
            "classifier":      classifier,
            "quality":         quality,
            "composite_score": cs,
        }

    ranking = sorted(
        models,
        key=lambda m: comparison[m].get("composite_score") or -1,
        reverse=True,
    )

    report = {
        "generated_at":    datetime.now().isoformat(),
        "mode":            mode,
        "models_compared": models,
        "judge_model":     os.getenv("JUDGE_MODEL", "gpt-oss:120b"),
        "results":         comparison,
        "ranking":         ranking,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(report)
    print(f"Reporte JSON guardado en: {output_path}")

    csv_path = output_path.with_name(output_path.stem + ".csv")
    export_summary_csv(report, csv_path)


if __name__ == "__main__":
    main()