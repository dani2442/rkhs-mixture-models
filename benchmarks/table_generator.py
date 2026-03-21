"""
Generate the unified benchmark LaTeX table from JSON result files.

Reads all ``benchmarks/results/*.json``, combines them with the static
metadata in ``config.py``, and writes ``paper/sections/table_benchmark.tex``.

Can be run standalone::

    python -m benchmarks.table_generator
"""

import os

from benchmarks.config import BENCHMARK_COLUMNS, METHOD_REGISTRY
from benchmarks.runner import DEFAULT_RESULTS_DIR, load_all_results

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TABLE_PATH = os.path.join(
    _PROJECT_ROOT, "paper", "sections", "table_benchmark.tex"
)


def _format_cell(mean: float, std: float, is_best: bool) -> str:
    """Format a single ARI cell as ``$mean_{std}$`` with optional bold."""
    mean_s = f"{mean:.3f}"
    std_s = f"{std:.3f}"
    if is_best:
        return f"$\\mathbf{{{mean_s}}}_{{{std_s}}}$"
    return f"${mean_s}_{{{std_s}}}$"


def generate_table(
    results_dir: str = DEFAULT_RESULTS_DIR,
    output_path: str = DEFAULT_TABLE_PATH,
):
    """
    Build and write the unified benchmark table.

    Parameters
    ----------
    results_dir : str
        Directory containing the per-benchmark JSON files.
    output_path : str
        Path for the generated ``.tex`` file.
    """
    all_data = load_all_results(results_dir)

    # ── Determine the best mean ARI per column ──────────────────────
    best_per_col: dict[str, float] = {}
    for bench_id in BENCHMARK_COLUMNS:
        if bench_id not in all_data:
            continue
        bench_results = all_data[bench_id]["results"]
        best = -1.0
        for method_name, info in METHOD_REGISTRY.items():
            if bench_id not in info["benchmarks"]:
                continue
            if method_name in bench_results:
                val = bench_results[method_name].get("mean_ari")
                if val is not None and val > best:
                    best = val
        if best > -1.0:
            best_per_col[bench_id] = best

    # ── Build rows ──────────────────────────────────────────────────
    n_bench = len(BENCHMARK_COLUMNS)
    col_spec = "l l c c " + " ".join(["c"] * n_bench)

    rows: list[str] = []
    prev_space = None
    for method_name, info in METHOD_REGISTRY.items():
        # Add a thin separator when the space category changes
        if prev_space is not None and info["space"] != prev_space:
            rows.append("\\midrule")
        prev_space = info["space"]

        cells = [
            method_name,
            info["space"],
            info["train"],
            info["infer"],
        ]
        for bench_id in BENCHMARK_COLUMNS:
            if bench_id not in info["benchmarks"]:
                cells.append("---")
                continue
            if bench_id not in all_data:
                cells.append("---")
                continue
            bench_results = all_data[bench_id]["results"]
            if method_name not in bench_results:
                cells.append("---")
                continue
            entry = bench_results[method_name]
            mean_ari = entry.get("mean_ari")
            std_ari = entry.get("std_ari", 0.0)
            if mean_ari is None:
                cells.append("---")
                continue
            is_best = (
                bench_id in best_per_col
                and abs(mean_ari - best_per_col[bench_id]) < 1e-9
            )
            cells.append(_format_cell(mean_ari, std_ari, is_best))

        rows.append(" & ".join(cells) + r" \\")

    # ── Column headers ──────────────────────────────────────────────
    headers = ["Method", "Space", "Train", "Infer"]
    headers.extend(col["header"] for col in BENCHMARK_COLUMNS.values())
    header_line = " & ".join(headers) + r" \\"

    # ── Assemble table ──────────────────────────────────────────────
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Clustering comparison across spaces (ARI). "
        r"Each cell shows $\mathrm{mean}_{\mathrm{std}}$ over multiple "
        r"datasets/runs. Best per column in \textbf{bold}. "
        r"``---'' indicates the method cannot handle that space or "
        r"the benchmark has not been run yet.}",
        r"\label{tab:unified_benchmark}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\scriptsize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_line,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[table_generator] Wrote {output_path}")


if __name__ == "__main__":
    generate_table()
