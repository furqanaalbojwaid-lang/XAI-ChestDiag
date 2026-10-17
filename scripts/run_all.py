"""Cross-platform (Windows / Linux / macOS) runner for the full experiment.

    python scripts/run_all.py                         # seed 42
    python scripts/run_all.py --seeds 42 43 44        # three seeds (recommended)
    python scripts/run_all.py --models densenet121    # only the proposed model
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    cmd = [sys.executable, "-m", *args]
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--skip-training", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    models = a.models or cfg["models"]
    c = ["--config", a.config]

    run("src.prepare_splits", *c)
    for seed in a.seeds:
        for m in models:
            if not a.skip_training:
                run("src.train", *c, "--model", m, "--seed", str(seed))
            run("src.evaluate", *c, "--model", m, "--seed", str(seed))
    s0 = str(a.seeds[0])
    run("src.throughput", *c, "--batch-sizes", "1", "64")
    run("src.compare", *c, "--seed", s0)
    run("src.pointing_game", *c, "--model", "densenet121", "--seed", s0)
    run("src.plots", *c, "--seed", s0)
    run("src.gradcam_figure", *c, "--model", "densenet121", "--seed", s0)
    print("\nDone. Tables: outputs/summary   Figures: outputs/figures")


if __name__ == "__main__":
    main()
