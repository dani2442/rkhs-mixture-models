"""
Shared utilities for benchmark scripts.

Provides result saving/loading in a common JSON format and a hook
to regenerate the unified LaTeX table after each benchmark run.
"""

import json
import os
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "benchmarks", "results")


def save_benchmark_results(
    benchmark_id: str,
    config: dict,
    results: dict,
    results_dir: str = DEFAULT_RESULTS_DIR,
    regenerate: bool = True,
):
    """
    Save benchmark results to JSON and optionally regenerate the table.

    Parameters
    ----------
    benchmark_id : str
        Must match a key in ``BENCHMARK_COLUMNS`` (e.g. ``"l2"``).
    config : dict
        Experiment hyper-parameters (for reproducibility).
    results : dict
        ``{method_name: {"ari_scores": [...], "mean_ari": float, "std_ari": float}}``.
        Method names must match keys in ``METHOD_REGISTRY``.
    results_dir : str
        Directory where JSON files are stored.
    regenerate : bool
        If *True*, regenerate ``paper/sections/table_benchmark.tex`` after saving.
    """
    os.makedirs(results_dir, exist_ok=True)

    payload = {
        "benchmark_id": benchmark_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "results": results,
    }

    path = os.path.join(results_dir, f"{benchmark_id}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[runner] Saved results to {path}")

    if regenerate:
        from benchmarks.table_generator import generate_table

        generate_table(results_dir=results_dir)


def load_all_results(results_dir: str = DEFAULT_RESULTS_DIR) -> dict:
    """
    Load all JSON result files from *results_dir*.

    Returns
    -------
    dict
        ``{benchmark_id: payload}`` where payload is the full JSON dict.
    """
    all_results = {}
    if not os.path.isdir(results_dir):
        return all_results

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        all_results[data["benchmark_id"]] = data

    return all_results
