"""Run the whole experiment in dependency order.

  python -m src.pipeline                 every stage
  python -m src.pipeline --from compare  resume at a stage

Training skips runs that already exist, so deleting artifacts/reports/runs forces a full rerun.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

STAGES = [
    ("verify", ["src.data.verify_download"]),
    ("quality", ["src.data.quality"]),
    ("graph", ["src.data.build_graph"]),
    ("eda", ["src.eval.eda"]),
    ("leakage", ["src.eval.leakage_checks"]),
    ("baseline", ["src.train.baseline"]),
    ("gnn", ["src.train.gnn"]),
    ("compare", ["src.eval.compare"]),
    ("anomaly", ["src.eval.anomaly"]),
    ("errors", ["src.eval.error_analysis"]),
    ("explain", ["src.explain.run"]),
    ("index", ["src.api.build_index"]),
]


def main() -> None:
    names = [name for name, _ in STAGES]
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", choices=names, default=names[0])
    args = parser.parse_args()
    for name, modules in STAGES[names.index(args.start):]:
        for module in modules:
            print(f"== {name}: {module}", flush=True)
            subprocess.run([sys.executable, "-W", "ignore", "-m", module], check=True)
    print("pipeline complete")


if __name__ == "__main__":
    main()
