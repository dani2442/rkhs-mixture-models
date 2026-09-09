"""Regression checks for bandwidth randomness and independent seeded fits."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.paper_experiments.synthetic_experiments import make_mixture, median_sigma


class ReproducibilityTests(unittest.TestCase):
    def test_bandwidth_is_independent_of_global_random_state(self):
        torch.set_num_threads(1)
        x, _ = make_mixture(5, 6, 200, seed=3)  # exceeds the 800-point subsample
        before = torch.get_rng_state().clone()
        first = median_sigma(x, seed=17)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        torch.manual_seed(999)
        torch.rand(500)
        self.assertEqual(first, median_sigma(x, seed=17))

    def test_fresh_process_fits_reproduce_statistics(self):
        command = [
            sys.executable, str(ROOT / "examples/paper_experiments/synthetic_experiments.py"),
            "largek-single", "--K", "5", "--n_per", "200", "--M", "6",
            "--epochs", "10", "--restarts", "2", "--seed", "3", "--threads", "1",
        ]
        with tempfile.TemporaryDirectory() as workdir:
            command += ["--results-dir", workdir]
            rows = [json.loads(subprocess.check_output(command, text=True, cwd=workdir))
                    for _ in range(2)]
        # Time and peak memory are measurements, not deterministic estimator outputs.
        for key in ["sigma", "ari_per_restart", "objective_per_restart", "ari_mean", "ari_std"]:
            self.assertEqual(rows[0][key], rows[1][key], key)
        self.assertEqual(rows[0]["initialization_seeds"], [3, 1003])
        self.assertEqual(rows[0]["bandwidth_seed"], 3)
        self.assertEqual(rows[0]["environment"]["torch_threads"], 1)


if __name__ == "__main__":
    unittest.main()
