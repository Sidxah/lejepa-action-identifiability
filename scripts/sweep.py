"""Ablation sweep: run the grid, save results/metrics.csv, render figures.

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

def build_grid(cfg):
    """The ablation grid: list of (label, config). QUICK keeps a representative
    subset that exercises every code path; FULL is the spec's grid."""
    grid = []

    def add(label, **kw):
        grid.append((label, dataclasses.replace(cfg, **kw)))

    add("base")
    for s in ([0.0] if cfg.quick else [0.0, 0.25, 0.5, 0.75]):  # s=1.0 IS base
        add(f"nonlin_s={s:.2f}", nonlinearity_strength=s)
    add("sigreg_off", lam=0.0)
    if not cfg.quick:
        add("lam=0.05(paper)", lam=0.05)  # paper-default enforcement strength, measured
    add("symmetric", action_conditioned=False)
    if not cfg.quick:
        add("excite_scale=0.5", action_scale=0.5)
        add("excite_scale=0.1", action_scale=0.1)
        add("excite_rank1", action_rank=1)
    add("literal_noise", noise_mode="literal")
    if not cfg.quick:  # sensitivity, one factor at a time (K follows n unless probed)
        add("n=4", n=4, K=4)
        add("n=16", n=16, K=16)
        add("m=1", m=1)
        add("m=4", m=4)
        add("rho=0.5", rho=0.5)
        add("rho=0.99", rho=0.99)
        add("K=16", K=16)
    return grid


def project_sweep_time(grid, device, probe_steps=30):
    """Time a short real probe and print the projected sweep duration BEFORE
    launching, so the cost of the full grid is auditable up front."""
    _, c0 = grid[0]
    probe = dataclasses.replace(c0, steps=probe_steps, warmup=min(5, probe_steps))
    world = World(probe)
    train_data, _ = make_datasets(world, probe)
    t0 = time.time()
    train_run(probe, train_data, device, probe.train_seeds[0])
    per_step = (time.time() - t0) / probe_steps  # includes setup -> conservative
    total_steps = sum(c.steps * len(c.train_seeds) for _, c in grid)
    est = total_steps * per_step
    n_runs = sum(len(c.train_seeds) for _, c in grid)
    print(f"sweep: {len(grid)} conditions x seeds = {n_runs} runs, "
          f"~{per_step * 1000:.0f} ms/step -> projected ~{est / 60:.1f} min on {device}")
    return est


def run_sweep(cfg, device):
    """Run the full grid; save results/metrics.csv; return the per-run DataFrame."""
    grid = build_grid(cfg)
    project_sweep_time(grid, device)
    t0 = time.time()
    rows = []
    for i, (label, c) in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] {label}", flush=True)
        rows.extend(run_condition(label, c, device))
    df = pd.DataFrame(rows)
    shutil.rmtree(FIG_DIR, ignore_errors=True)  # no stale figures from earlier grids
    ensure_dirs()
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print(f"sweep done in {(time.time() - t0) / 60:.1f} min; "
          f"wrote {RESULTS_DIR / 'metrics.csv'} ({len(df)} runs)")
    return df


def aggregate(df):
    """Mean/std per condition. inf (degenerate cond_m) is excluded from the means
    and counted separately in n_inf_cond -- silent truncation would misreport."""
    d = df.replace([np.inf, -np.inf], np.nan)
    cols = ["R2_lin", "R2_orth", "gap", "cond_m", "theta_max_deg", "rho_hat", "D_rel"]
    agg = d.groupby("condition", sort=False)[cols].agg(["mean", "std"]).round(4)
    g = df.groupby("condition", sort=False)
    agg[("seeds", "")] = g["seed"].count()
    agg[("conv_frac", "")] = g["converged"].mean().round(2)
    agg[("n_inf_cond", "")] = g["cond_m"].apply(lambda s: int(np.isinf(s).sum()))
    return agg

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
    ax.axhline(1.5, color="tab:red", lw=1.0, ls="--", label="V-JEPA 2 reported $\\approx$ 1.5")
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
    ax.set_ylabel("$\\|Q^T \\hat{R} Q - \\rho I\\|_F / (\\rho \\sqrt{n})$")
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
