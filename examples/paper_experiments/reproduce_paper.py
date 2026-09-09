#!/usr/bin/env python
"""Check or reproduce the GPmix entry and component sweep retained in the paper.

``check`` and ``tables`` use only the standard library and saved results.
``gpmix`` refits the paper configuration twice per dataset to verify repeatability.
Historical measurements are never overwritten by this script.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT))
SCALAR_NAMES = ["Growth", "Waveform", "Phoneme", "Kneading", "ECG200"]
GP_CONFIG_NAME = "rl-fpc-12"


def recorded_rows():
    data = json.loads((RESULTS / "gpmix_comparison.json").read_text())
    scores = [data["gpmix_grid"][name][GP_CONFIG_NAME]["ari"] for name in SCALAR_NAMES]
    gp_cell = f"${statistics.mean(scores):.3f}_{{{statistics.pstdev(scores):.3f}}}$"
    scaling = json.loads((RESULTS / "largek_nfixed.json").read_text())
    if [r["K"] for r in scaling] != [5, 10, 20, 50, 100, 200]:
        raise ValueError("Unexpected component sweep")
    if any(r["n"] != 100000 or r["M"] != 20 for r in scaling):
        raise ValueError("Expected n=100000 and M=20")
    rows = [
        "Components $K$ & " + " & ".join(str(r["K"]) for r in scaling),
        r"ARI (mean $\pm$ SD) & " + " & ".join(
            f"${r['ari_mean']:.3f}\\pm{r['ari_std']:.3f}$" for r in scaling
        ),
        "Time/fit (s) & " + " & ".join(f"{r['runtime_s']:.2f}" for r in scaling),
        "Memory (MiB) & " + " & ".join(f"{r['peak_extra_mb']:.0f}" for r in scaling),
    ]
    return gp_cell, [row + r" \\" for row in rows]


def check_paper():
    gp_cell, scaling_rows = recorded_rows()
    for name in ["main.tex", "preprint.tex"]:
        text = (ROOT / "paper" / name).read_text()
        gp = next(line for line in text.splitlines() if line.startswith("GPmix"))
        em = next(line for line in text.splitlines() if line.startswith("Gaussian Mixture (EM)"))
        medoids = next(line for line in text.splitlines() if line.startswith("K-Medoids"))
        if gp_cell not in gp or em.split(" & ")[6] != "---":
            raise ValueError(f"{name}: incorrect GPmix or L2 EM cell")
        if r"$\mathbf{0.458}_{0.201}$" not in medoids:
            raise ValueError(f"{name}: missing original K-Medoids emphasis")
    for name in ["supplemental.tex", "supplemental_preprint.tex"]:
        text = (ROOT / "paper" / name).read_text()
        for row in scaling_rows:
            if row not in text:
                raise ValueError(f"{name}: missing or changed row: {row}")
        if "tab:large_n" in text:
            raise ValueError(f"{name}: removed sample-size table is still present")
    print("Both papers match the saved GPmix and component-sweep results.")


def refit_gpmix(args):
    import platform
    import numpy as np
    import sklearn
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    from sklearn.metrics import adjusted_rand_score
    from benchmarks.bench_l2_realdata import PAPER_DATASET_SPECS, PAPER_GPMIX_CONFIG
    from src.competitors import GPmixClustering

    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    recorded = json.loads((RESULTS / "gpmix_comparison.json").read_text())
    scores = {}
    with threadpool_limits(limits=args.threads):
        for loader, _ in PAPER_DATASET_SPECS:
            curves, labels, k, name = loader()
            first = GPmixClustering(n_clusters=k, **PAPER_GPMIX_CONFIG).fit_predict(curves)
            # Change ambient RNG states; the estimator's own seed must suffice.
            np.random.seed(12345)
            torch.manual_seed(12345)
            second = GPmixClustering(n_clusters=k, **PAPER_GPMIX_CONFIG).fit_predict(curves)
            if not np.array_equal(first, second):
                raise ValueError(f"{name}: seeded GPmix fits differ")
            score = adjusted_rand_score(labels, first)
            expected = recorded["gpmix_grid"][name][GP_CONFIG_NAME]["ari"]
            if abs(score - expected) > 1e-12:
                raise ValueError(f"{name}: ARI {score} differs from recorded {expected}")
            scores[name] = score
            print(f"{name}: ARI={score:.6f}, repeated fit and saved result agree")
        environment = {
            "python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "threadpools": threadpool_info(),
        }
    values = list(scores.values())
    result = {"configuration": PAPER_GPMIX_CONFIG, "scores": scores,
              "mean_ari": statistics.mean(values), "std_ari": statistics.pstdev(values),
              "environment": environment}
    print(f"Five-dataset mean ± SD: {result['mean_ari']:.3f} ± {result['std_ari']:.3f}")
    if args.output:
        path = Path(args.output).resolve()
        if path == RESULTS or RESULTS in path.parents:
            raise ValueError("Use an output outside the historical results directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    sub.add_parser("tables")
    gp = sub.add_parser("gpmix")
    gp.add_argument("--threads", type=int, default=1)
    gp.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        check_paper()
    elif args.command == "tables":
        gp_cell, rows = recorded_rows()
        print("GPmix L2 cell:", gp_cell)
        print("\n".join(rows))
    else:
        refit_gpmix(args)


if __name__ == "__main__":
    main()
