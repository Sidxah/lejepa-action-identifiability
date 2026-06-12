#!/usr/bin/env python3
"""Phase-2 extraction: notebook module cells -> src/*.py and scripts/sweep.py.

Mechanical by design (the addendum's contract): the BODIES of the module-tagged
cells are copied verbatim from the notebook; this script only adds curated
import headers, package glue, and the CLI mains. Re-run it after rebuilding the
notebook to keep notebook and repo in sync.

Usage: python tools/extract_modules.py NOTEBOOK.ipynb [--root .]
"""
import argparse
import pathlib

import nbformat

# Notebook-only lines stripped from the setup header (matplotlib inline backend
# and IPython display are meaningless outside a kernel; drivers are not extracted).
SETUP_STRIP_START = "import matplotlib"
SETUP_STRIP_END = "from IPython.display import Markdown, display"

HEADERS = {
    "config": '''"""Experiment configuration, seeding, device selection, output paths."""
import os
from dataclasses import dataclass

import numpy as np
import torch
from pathlib import Path
''',
    "world": '''"""Synthetic world: ground-truth latent dynamics + frozen observation map."""
import math

import torch
import torch.nn.functional as F
''',
    "model": '''"""Encoder and action-conditioned linear predictor."""
import math

import torch
import torch.nn as nn
''',
    "sigreg": '''"""SIGReg: sliced Epps-Pulley characteristic-function match to N(0, I)."""
import torch
''',
    "train": '''"""Training loop: LeJEPA recipe (no EMA, no stop-gradient), AdamW + warmup-cosine."""
import math

import numpy as np
import torch

from .config import set_seed
from .model import Encoder, LinearPredictor
from .sigreg import sigreg
''',
    "metrics": '''"""Metrics M1/M2/M3, embedding diagnostics, and the pre-registered verdict."""
import math

import numpy as np
import torch
''',
    "run": '''"""One experimental condition: world + datasets once, then train/measure per seed."""
import time

from .metrics import compute_all_metrics
from .train import train_run
from .world import World, make_datasets
''',
}

SWEEP_HEADER = '''"""Ablation sweep: run the grid, save results/metrics.csv, render figures.

Usage (from the repo root):
    python scripts/sweep.py            # FULL grid (the spec's experiment)
    python scripts/sweep.py --quick    # smoke-test grid, finishes in ~a minute
"""
import argparse
import dataclasses
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIG_DIR, RESULTS_DIR, ensure_dirs, make_config, pick_device
from src.metrics import render_verdict, verdict
from src.run import run_condition
from src.train import train_run
from src.world import World, make_datasets
'''

SWEEP_FOOTER = '''

def _stats(df, cond, col):
    vals = df.loc[df.condition == cond, col].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    n_inf = int(np.isinf(vals).sum())
    if len(finite) == 0:
        return float("nan"), 0.0, n_inf
    return float(finite.mean()), float(finite.std()), n_inf


def make_figures(df):
    """The four figures from the spec, saved to results/figures/."""
    cond_order = list(dict.fromkeys(df.condition))

    # (a) R2_orth vs R2_lin across the nonlinearity sweep
    nl = df[df.condition.str.startswith("nonlin_") | (df.condition == "base")]
    ns = nl.groupby("nonlinearity_strength")[["R2_lin", "R2_orth"]].agg(["mean", "std"]).sort_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for col, color in [("R2_lin", "tab:blue"), ("R2_orth", "tab:red")]:
        ax.errorbar(ns.index, ns[(col, "mean")], yerr=ns[(col, "std")].fillna(0.0),
                    marker="o", capsize=3, color=color, label=f"$R^2_{{{col[3:]}}}$")
    ax.set_xlabel("nonlinearity strength $s$ of the observation map")
    ax.set_ylabel("$R^2$ (state recovery, eval split)")
    ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.02)
    ax.legend(); ax.set_title("M1: linear vs orthogonal state recovery", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig1_r2_vs_nonlinearity.png", dpi=150); plt.close(fig)

    # (b) headline cond_m across conditions, with the V-JEPA 2 reference line
    conds_b = [c for c in cond_order
               if np.isfinite(df.loc[df.condition == c, "cond_m"]).any()
               or np.isinf(df.loc[df.condition == c, "cond_m"]).any()]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.62 * len(conds_b)), 3.8))
    for i, c in enumerate(conds_b):
        mu, sd, n_inf = _stats(df, c, "cond_m")
        if np.isfinite(mu):
            ax.errorbar([i], [mu], yerr=[sd], fmt="o", capsize=4, color="tab:blue")
        if n_inf:
            ax.annotate(f"{n_inf} run(s) at inf", (i, ax.get_ylim()[1]), fontsize=7,
                        ha="center", color="tab:red", rotation=90, va="top")
    ax.axhline(1.0, color="gray", lw=0.8, label="exact rotation (cond = 1)")
    ax.axhline(1.5, color="tab:red", lw=1.0, ls="--", label="V-JEPA 2 reported $\\\\approx$ 1.5")
    ax.set_yscale("log")
    ax.set_xticks(range(len(conds_b)), conds_b, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("cond$_m(L)$  (1 = rotation)")
    ax.legend(fontsize=8)
    ax.set_title("M2 headline: action-axis condition number across conditions", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig2_cond_errorbars.png", dpi=150); plt.close(fig)

    # (c) dynamics recovery error across conditions
    fig, ax = plt.subplots(figsize=(max(6.0, 0.62 * len(cond_order)), 3.4))
    for i, c in enumerate(cond_order):
        mu, sd, _ = _stats(df, c, "D_rel")
        ax.errorbar([i], [mu], yerr=[sd], fmt="s", capsize=4, color="tab:green")
    ax.set_xticks(range(len(cond_order)), cond_order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("$\\\\|Q^T \\\\hat{R} Q - \\\\rho I\\\\|_F / (\\\\rho \\\\sqrt{n})$")
    ax.set_title("M3: dynamics recovery error", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig3_dynamics_error.png", dpi=150); plt.close(fig)

    # (d) symmetric vs action-conditioned state recovery
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    width = 0.35
    for j, cond in enumerate(["base", "symmetric"]):
        mus, sds = zip(*[_stats(df, cond, col)[:2] for col in ("R2_lin", "R2_orth")])
        ax.bar(np.arange(2) + (j - 0.5) * width, mus, width, yerr=sds, capsize=4,
               color="tab:blue" if cond == "base" else "tab:gray",
               label="action-conditioned" if cond == "base" else "symmetric (B = 0)")
    ax.set_xticks([0, 1], ["$R^2_{lin}$", "$R^2_{orth}$"])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("does the ambiguity survive the passage?", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig4_symmetric_vs_action.png", dpi=150); plt.close(fig)
    print(f"figures saved to {FIG_DIR}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smoke-test grid (NOT science)")
    args = ap.parse_args()
    cfg = make_config(args.quick)
    device = pick_device()
    ensure_dirs()
    df = run_sweep(cfg, device)
    print(aggregate(df).to_string())
    make_figures(df)
    print()
    print(render_verdict(verdict(df), cfg.quick))


if __name__ == "__main__":
    main()
'''

RUN_FOOTER = '''

def main():
    """Run one experimental condition from the command line (config-driven via argparse)."""
    import argparse
    import dataclasses

    from .config import make_config, pick_device

    ap = argparse.ArgumentParser(description="Run one experiment condition (all training seeds).")
    ap.add_argument("--quick", action="store_true", help="smoke-test dimensions (NOT science)")
    ap.add_argument("--label", default="custom")
    for name, typ in [("n", int), ("m", int), ("rho", float), ("K", int), ("D", int),
                      ("steps", int), ("lam", float), ("nonlinearity-strength", float),
                      ("action-scale", float), ("action-rank", int), ("noise-mode", str),
                      ("world-seed", int)]:
        ap.add_argument(f"--{name}", type=typ, default=None)
    ap.add_argument("--symmetric", action="store_true",
                    help="B = 0 world with a no-action predictor")
    args = ap.parse_args()

    cfg = make_config(args.quick)
    overrides = {k.replace("-", "_"): v for k, v in vars(args).items()
                 if v is not None and k not in ("quick", "label", "symmetric")}
    if args.symmetric:
        overrides["action_conditioned"] = False
    cfg = dataclasses.replace(cfg, **overrides)
    rows = run_condition(args.label, cfg, pick_device())
    import pandas as pd

    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebook")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = pathlib.Path(args.root)

    nb = nbformat.read(args.notebook, as_version=4)
    bodies = {}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        first = cell.source.splitlines()[0].strip()
        if not first.startswith("# -- module:"):
            continue
        tag = first.removeprefix("# -- module:").removesuffix("--").strip()
        body = "\n".join(cell.source.splitlines()[1:]).strip()  # drop the tag line
        bodies.setdefault(tag, []).append(body)

    # setup -> src/config.py, with the notebook-only display block stripped
    setup_cells = bodies.pop("setup")
    lines = setup_cells[0].splitlines()
    keep, skipping = [], False
    for ln in lines:
        if ln.startswith(SETUP_STRIP_START):
            skipping = True
        if not skipping and not (ln.startswith("import ") or ln.startswith("from ")):
            keep.append(ln)
        if skipping and ln.startswith(SETUP_STRIP_END):
            skipping = False
    config_body = "\n".join(keep).strip() + "\n\n\n" + "\n\n".join(setup_cells[1:])

    (root / "src").mkdir(exist_ok=True)
    (root / "scripts").mkdir(exist_ok=True)
    (root / "src" / "__init__.py").write_text("")

    outputs = {
        root / "src" / "config.py": HEADERS["config"] + "\n" + config_body + "\n",
        root / "src" / "world.py": HEADERS["world"] + "\n" + "\n\n".join(bodies["world"]) + "\n",
        root / "src" / "model.py": HEADERS["model"] + "\n" + "\n\n".join(bodies["model"]) + "\n",
        root / "src" / "sigreg.py": HEADERS["sigreg"] + "\n" + "\n\n".join(bodies["sigreg"]) + "\n",
        root / "src" / "train.py": HEADERS["train"] + "\n" + "\n\n".join(bodies["train"]) + "\n",
        root / "src" / "metrics.py": HEADERS["metrics"] + "\n" + "\n\n".join(bodies["metrics"]) + "\n",
        root / "src" / "run.py": HEADERS["run"] + "\n" + "\n\n".join(bodies["run"]) + RUN_FOOTER,
        root / "scripts" / "sweep.py": SWEEP_HEADER + "\n" + "\n\n".join(bodies["sweep"]) + SWEEP_FOOTER,
    }
    for path, text in outputs.items():
        compile(text, str(path), "exec")
        path.write_text(text)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
