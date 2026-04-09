"""
compare_results.py
Genera un reporte comparativo a partir de los JSON producidos por:
  - eval_models.py       → tests/results/{model}_{ts}.json
  - test_llm_quality.py  → tests/results/quality_{judge}_{ts}.json
  - test_multi_turn.py   → tests/results/multi_turn_{judge}_{ts}.json

Para cada modelo recoge la ejecución más reciente de cada suite y calcula
un score compuesto (33% guardrails + 33% quality + 33% multi-turn).

Uso:
  python tests/compare_results.py
  python tests/compare_results.py --results-dir tests/results --output tests/results/comparison_report.json
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "comparison_report.json"

MODELS = [
    "gemini-3-flash-preview",
    "gpt-oss:120b",
    "qwen3.5:397b"
]


# ---------------------------------------------------------------------------
# Carga de archivos por tipo
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def latest_guardrail_file(results_dir: Path, model: str) -> Path | None:
    """Devuelve el archivo de guardrail más reciente para el modelo dado."""
    safe = model.replace("/", "_").replace(":", "_")
    candidates = sorted(results_dir.glob(f"{safe}_*.json"))
    # Excluir archivos que no son de guardrail (quality_, multi_turn_, comparison_)
    guardrail_files = [
        p for p in candidates
        if not p.name.startswith("quality_")
        and not p.name.startswith("multi_turn_")
        and not p.name.startswith("comparison_")
    ]
    return guardrail_files[-1] if guardrail_files else None


def latest_typed_file(results_dir: Path, prefix: str, model: str) -> Path | None:
    """Devuelve el archivo quality_ o multi_turn_ más reciente que corresponda al modelo."""
    candidates = sorted(results_dir.glob(f"{prefix}*.json"))
    # Filtrar por campo "model" dentro del JSON
    matches = []
    for path in candidates:
        try:
            data = load_json(path)
            if data.get("model") == model:
                matches.append(path)
        except Exception:
            continue
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Extracción de scores
# ---------------------------------------------------------------------------

def extract_guardrail(data: dict) -> dict:
    s = data.get("scores", {})
    g = s.get("guardrail", {})
    by_guard = g.get("by_guard", {})
    timing = s.get("timing", {})
    return {
        "overall_pct": g.get("score_pct", None),
        "topic_pct":   by_guard.get("topic", {}).get("score_pct", None),
        "toxic_pct":   by_guard.get("toxic", {}).get("score_pct", None),
        "pii_pct":     by_guard.get("pii", {}).get("score_pct", None),
        "off_topic_pct": g.get("off_topic", {}).get("score_pct", None),
        "on_topic_pct":  g.get("on_topic", {}).get("score_pct", None),
        "avg_ms":      timing.get("avg_ms", None),
        "p95_ms":      timing.get("p95_ms", None),
        "source_file": data.get("_source_file", ""),
    }


def extract_quality(data: dict) -> dict:
    s = data.get("scores", {})
    ov = s.get("overall", {})
    pm = s.get("per_metric", {})
    timing = s.get("timing", {})

    def metric_avg(key: str) -> float | None:
        return pm.get(key, {}).get("avg_score", None)

    def metric_pass(key: str) -> float | None:
        return pm.get(key, {}).get("pass_rate_pct", None)

    return {
        "overall_pct":             ov.get("score_pct", None),
        "answer_relevancy_avg":    metric_avg("answer_relevancy"),
        "faithfulness_avg":        metric_avg("faithfulness"),
        "contextual_relevancy_avg":metric_avg("contextual_relevancy"),
        "contextual_recall_avg":   metric_avg("contextual_recall"),
        "contextual_precision_avg":metric_avg("contextual_precision"),
        "answer_relevancy_pass":   metric_pass("answer_relevancy"),
        "faithfulness_pass":       metric_pass("faithfulness"),
        "avg_ms":  timing.get("avg_ms", None),
        "p95_ms":  timing.get("p95_ms", None),
        "judge":   data.get("judge", ""),
        "source_file": data.get("_source_file", ""),
    }


def extract_multi_turn(data: dict) -> dict:
    s = data.get("scores", {})
    ov = s.get("overall", {})
    pm = s.get("per_metric", {})
    puc = s.get("per_use_case", {})
    timing = s.get("timing", {})

    def metric_avg(key: str) -> float | None:
        return pm.get(key, {}).get("avg_score", None)

    def metric_pass(key: str) -> float | None:
        return pm.get(key, {}).get("pass_rate_pct", None)

    def uc_pct(key: str) -> float | None:
        return puc.get(key, {}).get("score_pct", None)

    return {
        "overall_pct":                    ov.get("score_pct", None),
        "conversation_completeness_avg":  metric_avg("conversation_completeness"),
        "turn_relevancy_avg":             metric_avg("turn_relevancy"),
        "knowledge_retention_avg":        metric_avg("knowledge_retention"),
        "role_adherence_avg":             metric_avg("role_adherence"),
        "conversation_completeness_pass": metric_pass("conversation_completeness"),
        "turn_relevancy_pass":            metric_pass("turn_relevancy"),
        "use_case_transaction_pct":       uc_pct("transaction"),
        "use_case_financial_question_pct":uc_pct("financial_question"),
        "use_case_insight_pct":           uc_pct("insight"),
        "avg_ms":  timing.get("avg_ms", None),
        "p95_ms":  timing.get("p95_ms", None),
        "judge":   data.get("judge", ""),
        "source_file": data.get("_source_file", ""),
    }


# ---------------------------------------------------------------------------
# Score compuesto
# ---------------------------------------------------------------------------

def composite_score(guardrail: dict, quality: dict, multi_turn: dict) -> float | None:
    parts = []
    if guardrail.get("overall_pct") is not None:
        parts.append(guardrail["overall_pct"] / 100)
    if quality.get("overall_pct") is not None:
        parts.append(quality["overall_pct"] / 100)
    if multi_turn.get("overall_pct") is not None:
        parts.append(multi_turn["overall_pct"] / 100)
    return round(sum(parts) / len(parts), 4) if parts else None


# ---------------------------------------------------------------------------
# Impresión de tabla comparativa
# ---------------------------------------------------------------------------

def print_report(report: dict):
    models = report["models_compared"]
    results = report["results"]

    col = 30
    mcol = 10

    header = f"{'Métrica':<{col}}" + "".join(f"{m[-mcol:]:>{mcol}}" for m in models)
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  REPORTE COMPARATIVO DE MODELOS LLM — FINANZAS")
    print("=" * len(header))

    def row(label: str, extractor, fmt=".1f"):
        vals = []
        for m in models:
            v = extractor(results.get(m, {}))
            vals.append(f"{v:{fmt}}" if v is not None else "N/A")
        return f"  {label:<{col-2}}" + "".join(f"{v:>{mcol}}" for v in vals)

    def section(title: str):
        print(f"\n  [{title}]")
        print(sep)

    # Header row
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

    section("QUALITY (DeepEval)")
    print(row("Overall %",              lambda r: r.get("quality", {}).get("overall_pct")))
    print(row("  AnswerRelevancy avg",  lambda r: r.get("quality", {}).get("answer_relevancy_avg"), fmt=".4f"))
    print(row("  Faithfulness avg",     lambda r: r.get("quality", {}).get("faithfulness_avg"), fmt=".4f"))
    print(row("  CtxRelevancy avg",     lambda r: r.get("quality", {}).get("contextual_relevancy_avg"), fmt=".4f"))
    print(row("  CtxRecall avg",        lambda r: r.get("quality", {}).get("contextual_recall_avg"), fmt=".4f"))
    print(row("  CtxPrecision avg",     lambda r: r.get("quality", {}).get("contextual_precision_avg"), fmt=".4f"))
    print(row("Avg response (ms)",      lambda r: r.get("quality", {}).get("avg_ms"), fmt=".0f"))

    # MULTI-TURN (deshabilitado temporalmente)
    # section("MULTI-TURN (DeepEval)")
    # print(row("Overall %",                   lambda r: r.get("multi_turn", {}).get("overall_pct")))
    # print(row("  Completeness avg",          lambda r: r.get("multi_turn", {}).get("conversation_completeness_avg"), fmt=".4f"))
    # print(row("  TurnRelevancy avg",         lambda r: r.get("multi_turn", {}).get("turn_relevancy_avg"), fmt=".4f"))
    # print(row("  KnowledgeRetention avg",    lambda r: r.get("multi_turn", {}).get("knowledge_retention_avg"), fmt=".4f"))
    # print(row("  RoleAdherence avg",         lambda r: r.get("multi_turn", {}).get("role_adherence_avg"), fmt=".4f"))
    # print(row("  Use-case: transaction %",   lambda r: r.get("multi_turn", {}).get("use_case_transaction_pct")))
    # print(row("  Use-case: fin.question %",  lambda r: r.get("multi_turn", {}).get("use_case_financial_question_pct")))
    # print(row("  Use-case: insight %",       lambda r: r.get("multi_turn", {}).get("use_case_insight_pct")))
    # print(row("Avg response (ms)",           lambda r: r.get("multi_turn", {}).get("avg_ms"), fmt=".0f"))

    section("RANKING FINAL (score compuesto)")
    ranking = report["ranking"]
    for i, m in enumerate(ranking, 1):
        score = results.get(m, {}).get("composite_score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"    {i}. {m}  →  {score_str}")

    print("\n" + "=" * len(header) + "\n")


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
        default=MODELS,
        help="Modelos a comparar (default: los 3 modelos financieros)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    comparison: dict[str, dict] = {}

    for model in args.models:
        print(f"Procesando modelo: {model}")

        # Guardrail
        gf = latest_guardrail_file(results_dir, model)
        if gf:
            gdata = load_json(gf)
            gdata["_source_file"] = gf.name
            guardrail = extract_guardrail(gdata)
            print(f"  Guardrail: {gf.name}")
        else:
            guardrail = {}
            print(f"  Guardrail: (no encontrado)")

        # Quality
        qf = latest_typed_file(results_dir, "quality_", model)
        if qf:
            qdata = load_json(qf)
            qdata["_source_file"] = qf.name
            quality = extract_quality(qdata)
            print(f"  Quality:   {qf.name}")
        else:
            quality = {}
            print(f"  Quality:   (no encontrado)")

        # Multi-turn (deshabilitado temporalmente)
        # mf = latest_typed_file(results_dir, "multi_turn_", model)
        # if mf:
        #     mdata = load_json(mf)
        #     mdata["_source_file"] = mf.name
        #     multi_turn = extract_multi_turn(mdata)
        #     print(f"  Multi-turn:{mf.name}")
        # else:
        #     multi_turn = {}
        #     print(f"  Multi-turn:(no encontrado)")
        multi_turn = {}

        cs = composite_score(guardrail, quality, multi_turn)

        comparison[model] = {
            "guardrail":       guardrail,
            "quality":         quality,
            "multi_turn":      multi_turn,
            "composite_score": cs,
        }

    # Ranking por composite_score descendente
    ranking = sorted(
        args.models,
        key=lambda m: comparison[m].get("composite_score") or -1,
        reverse=True,
    )

    report = {
        "generated_at":   datetime.now().isoformat(),
        "models_compared": args.models,
        "judge_model":    os.getenv("JUDGE_MODEL", "gpt-oss:120b-cloud"),
        "results":        comparison,
        "ranking":        ranking,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(report)
    print(f"Reporte guardado en: {output_path}")


if __name__ == "__main__":
    main()