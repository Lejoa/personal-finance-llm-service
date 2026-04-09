"""
export_csv.py
Carga todos los comparison_report.json encontrados en los subdirectorios de
tests/results/ y genera un CSV consolidado con una fila por modelo.

Uso:
  python tests/export_csv.py
  python tests/export_csv.py --results-dir tests/results --output tests/results/comparativa.csv
"""

import argparse
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "comparativa.csv"

# Orden de columnas en el CSV
COLUMNS = [
    # Identificación
    "modelo",
    "carpeta",
    "fecha_evaluacion",
    "juez",
    # Score compuesto
    "score_compuesto",
    # Guardrails — resumen
    "guard_overall_pct",
    "guard_topic_pct",
    "guard_toxic_pct",
    "guard_pii_pct",
    "guard_off_topic_pct",
    "guard_on_topic_pct",
    "guard_avg_ms",
    "guard_p95_ms",
    # Quality — resumen
    "quality_overall_pct",
    "quality_answer_relevancy_avg",
    "quality_faithfulness_avg",
    "quality_ctx_relevancy_avg",
    "quality_ctx_recall_avg",
    "quality_ctx_precision_avg",
    "quality_answer_relevancy_pass_pct",
    "quality_faithfulness_pass_pct",
    "quality_avg_ms",
    "quality_p95_ms",
    # Archivos fuente
    "guardrail_source_file",
    "quality_source_file",
]


def _fmt(value) -> str:
    """Convierte None en cadena vacía y floats a 4 decimales."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_all_reports(results_dir: Path) -> list[dict]:
    """
    Busca comparison_report.json únicamente en la raíz de results_dir.
    """
    rows = []

    report_path = results_dir / "comparison_report.json"
    if not report_path.exists():
        print(f"No se encontró comparison_report.json en {results_dir}")
        return rows

    report_files = [report_path]

    for report_path in report_files:
        folder = "results"
        try:
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
        except Exception as e:
            print(f"  WARN: no se pudo leer {report_path}: {e}")
            continue

        generated_at = report.get("generated_at", "")
        judge = report.get("judge_model", "")
        results = report.get("results", {})

        for model, data in results.items():
            g = data.get("guardrail", {})
            q = data.get("quality", {})
            cs = data.get("composite_score")

            row = {
                "modelo": model,
                "carpeta": folder,
                "fecha_evaluacion": generated_at,
                "juez": judge,
                "score_compuesto": _fmt(cs),
                # Guardrails
                "guard_overall_pct":   _fmt(g.get("overall_pct")),
                "guard_topic_pct":     _fmt(g.get("topic_pct")),
                "guard_toxic_pct":     _fmt(g.get("toxic_pct")),
                "guard_pii_pct":       _fmt(g.get("pii_pct")),
                "guard_off_topic_pct": _fmt(g.get("off_topic_pct")),
                "guard_on_topic_pct":  _fmt(g.get("on_topic_pct")),
                "guard_avg_ms":        _fmt(g.get("avg_ms")),
                "guard_p95_ms":        _fmt(g.get("p95_ms")),
                # Quality
                "quality_overall_pct":              _fmt(q.get("overall_pct")),
                "quality_answer_relevancy_avg":     _fmt(q.get("answer_relevancy_avg")),
                "quality_faithfulness_avg":         _fmt(q.get("faithfulness_avg")),
                "quality_ctx_relevancy_avg":        _fmt(q.get("contextual_relevancy_avg")),
                "quality_ctx_recall_avg":           _fmt(q.get("contextual_recall_avg")),
                "quality_ctx_precision_avg":        _fmt(q.get("contextual_precision_avg")),
                "quality_answer_relevancy_pass_pct":_fmt(q.get("answer_relevancy_pass")),
                "quality_faithfulness_pass_pct":    _fmt(q.get("faithfulness_pass")),
                "quality_avg_ms":  _fmt(q.get("avg_ms")),
                "quality_p95_ms":  _fmt(q.get("p95_ms")),
                # Fuentes
                "guardrail_source_file": g.get("source_file", ""),
                "quality_source_file":   q.get("source_file", ""),
            }
            rows.append(row)

    return rows


def print_table(rows: list[dict]):
    if not rows:
        return

    # Ordenar por score_compuesto descendente (vacíos al final)
    def sort_key(r):
        v = r["score_compuesto"]
        return float(v) if v else -1.0

    rows_sorted = sorted(rows, key=sort_key, reverse=True)

    col_w = 28
    score_w = 10
    header = f"{'Modelo':<{col_w}} {'Carpeta':<12} {'Compuesto':>{score_w}} {'Guard%':>{score_w}} {'Quality%':>{score_w}} {'Guard ms':>{score_w}} {'Quality ms':>{score_w}}"
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  CONSOLIDADO DE EVALUACIONES LLM")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in rows_sorted:
        print(
            f"{r['modelo']:<{col_w}} "
            f"{r['carpeta']:<12} "
            f"{r['score_compuesto']:>{score_w}} "
            f"{r['guard_overall_pct']:>{score_w}} "
            f"{r['quality_overall_pct']:>{score_w}} "
            f"{r['guard_avg_ms']:>{score_w}} "
            f"{r['quality_avg_ms']:>{score_w}}"
        )

    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Exporta resultados de evaluación a CSV")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directorio raíz con subcarpetas de resultados (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Ruta del CSV de salida (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    print(f"Buscando reportes en: {results_dir}")
    rows = load_all_reports(results_dir)

    if not rows:
        print("No se encontraron datos para exportar.")
        return

    print(f"  Modelos encontrados: {len(rows)}")
    for r in rows:
        print(f"    [{r['carpeta']}] {r['modelo']}")

    # Escribir CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        # utf-8-sig: agrega BOM para que Excel/LibreOffice reconozca UTF-8 automáticamente
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV guardado en: {output_path}")
    print_table(rows)


if __name__ == "__main__":
    main()