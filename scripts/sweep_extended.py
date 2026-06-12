#!/usr/bin/env python3
"""Extended robustness sweep — answers to the review of the main experiment.

PRE-REGISTRATION NOTE. This script, including every threshold and expectation
below, is committed and pushed BEFORE the extended run is executed; the commit
timestamp is the tamper-evident record that the main sweep lacked. The five
questions it answers, each raised by review of the main results:

  Q1  MULTI-WORLD.  The main sweep used one world seed. Here: 4 new worlds
      (different B, different observation map) x {base, symmetric, rank-1
      excitation} x 5 training seeds.
      Pre-registered: >= 4 of 5 worlds (incl. 1234) CONFIRM on the base
      condition (gate R2_lin >= 0.90 on converged seeds; gap <= 0.05,
      cond_m <= 2.0, theta_max <= 15 deg).
  Q2  LAMBDA TRANSITION.  lam in {0.1, 0.2, 0.35, 0.65, 0.8} at base
      (0.05 and 0.5 come from the main sweep).
      Pre-registered: the smallest lambda with mean emb_cov_err <= 0.20 AND
      mean R2_lin >= 0.90 lies in [0.2, 0.5] (the collapse-equilibrium
      argument of NOTES.md predicts shrinkage ~ (1-lam)/lam).
  Q3  BUDGET vs WALL.  n=16 and rho=0.99 rerun at steps=8000 (2x).
      Pre-registered decision rule: delta R2_lin >= +0.15 vs the 4000-step
      main rows => "optimization-limited"; <= +0.05 => "identifiability/budget
      wall at this scale"; in between => mixed.
  Q4  OVERCOMPLETENESS.  K in {10, 12, 24} (8 and 16 from the main sweep).
      Pre-registered: the lin-orth gap grows monotonically with K - n.
  Q5  BASIN FREQUENCY.  5 extra base training seeds (world 1234) + the
      multi-world base runs => 30 base runs total. A run is "in the basin"
      iff rho_hat < 0.75 (clear bimodal separation in the main data:
      0.894 vs 0.567). Report frequency and the spectral signature
      (rho_mod - rho_hat large => complex eigenvalue pairs: a rotating
      suboptimal solution).

Outputs: results/metrics_ext.csv, results/figures/fig5..8_*.png, and a
printed report applying the rules above. Main-sweep rows are READ from
results/metrics.csv (never re-written).

Usage:
    python scripts/sweep_extended.py            # the real thing (~115 runs)
    python scripts/sweep_extended.py --smoke    # minutes-long code-path check
"""
import argparse
import dataclasses
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
from src.run import run_condition

# ---- pre-registered constants (see module docstring) ----
WORLD_SEEDS_EXT = [2024, 31337, 777, 9001]      # + 1234 from the main sweep
LAMBDAS_EXT = [0.1, 0.2, 0.35, 0.65, 0.8]       # + 0.05, 0.5 from the main sweep
K_EXT = [10, 12, 24]                            # + 8 (base), 16 from the main sweep
BUDGET_STEPS = 8000                             # vs 4000 in the main sweep
EXTRA_BASE_SEEDS = (5, 6, 7, 8, 9)              # + 0..4 from the main sweep
MW_GATE_R2 = 0.90
MW_GAP_OK, MW_COND_OK, MW_THETA_OK = 0.05, 2.0, 15.0
MW_PASS_REQUIRED = 4                            # of 5 worlds
LAM_COV_OK, LAM_R2_OK = 0.20, 0.90
LAM_PRED_LO, LAM_PRED_HI = 0.2, 0.5
BUDGET_OPT_LIMITED, BUDGET_WALL = 0.15, 0.05    # delta R2_lin thresholds
BASIN_RHO = 0.75                                # rho_hat below => basin membership


def build_grid_ext(cfg):
    grid = []

    def add(label, **kw):
        grid.append((label, dataclasses.replace(cfg, **kw)))

    for ws in WORLD_SEEDS_EXT:                       # Q1
        add(f"mw{ws}|base", world_seed=ws)
        add(f"mw{ws}|symmetric", world_seed=ws, action_conditioned=False)
        add(f"mw{ws}|excite_rank1", world_seed=ws, action_rank=1)
    for lam in LAMBDAS_EXT:                          # Q2
        add(f"lam={lam}", lam=lam)
    add("n=16@8000", n=16, K=16, steps=BUDGET_STEPS)  # Q3
    add("rho=0.99@8000", rho=0.99, steps=BUDGET_STEPS)
    for K in K_EXT:                                  # Q4
        add(f"K={K}", K=K)
    add("base_seeds5-9", train_seeds=EXTRA_BASE_SEEDS)  # Q5
    return grid


def _mean(df, mask, col):
    vals = df.loc[mask, col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def report(ext, main):
    """Apply the pre-registered rules. Every number comes from the two CSVs."""
    lines = ["", "=" * 72, "EXTENDED SWEEP REPORT (rules pre-registered in this script's header)",
             "=" * 72]

    # ---- Q1 multi-world ----
    worlds = ([1234] if main is not None else []) + WORLD_SEEDS_EXT
    passes, details = 0, []
    for ws in worlds:
        if ws == 1234:
            rows = main[(main.condition == "base") & (main.converged)]
        else:
            rows = ext[(ext.condition == f"mw{ws}|base") & (ext.converged)]
        r2 = _mean(rows, rows.index == rows.index, "R2_lin") if len(rows) else float("nan")
        gap = _mean(rows, rows.index == rows.index, "gap") if len(rows) else float("nan")
        cond = _mean(rows, rows.index == rows.index, "cond_m") if len(rows) else float("nan")
        th = _mean(rows, rows.index == rows.index, "theta_max_deg") if len(rows) else float("nan")
        ok = (r2 >= MW_GATE_R2) and (gap <= MW_GAP_OK) and (cond <= MW_COND_OK) and (th <= MW_THETA_OK)
        passes += bool(ok)
        details.append(f"    world {ws}: R2_lin={r2:.3f} gap={gap:.4f} "
                       f"cond_m={cond:.3f} theta={th:.2f} -> {'CONFIRM' if ok else 'fail'}")
    lines.append(f"[Q1 multi-world] {passes}/{len(worlds)} worlds confirm "
                 f"(pre-registered requirement: >= {MW_PASS_REQUIRED}/5) -> "
                 f"{'PASS' if passes >= MW_PASS_REQUIRED else 'FAIL'}")
    lines += details

    # ---- Q2 lambda transition ----
    lam_rows = []
    if main is not None:
        lam_rows.append((0.05, main[main.condition == "lam=0.05(paper)"]))
        lam_rows.append((0.5, main[main.condition == "base"]))
    for lam in LAMBDAS_EXT:
        lam_rows.append((lam, ext[ext.condition == f"lam={lam}"]))
    lam_rows.sort(key=lambda t: t[0])
    lam_star = None
    lines.append("[Q2 lambda transition]")
    for lam, rows in lam_rows:
        if len(rows) == 0:
            continue
        ce = rows.emb_cov_err.mean()
        r2 = rows.R2_lin.mean()
        cm = _mean(rows, rows.index == rows.index, "cond_m")
        lines.append(f"    lam={lam:<5} R2_lin={r2:.3f} emb_cov_err={ce:.3f} cond_m={cm:.3f}")
        if lam_star is None and ce <= LAM_COV_OK and r2 >= LAM_R2_OK:
            lam_star = lam
    if lam_star is None:
        lines.append("    lam* not reached in the sweep -> pre-registered prediction FAIL")
    else:
        ok = LAM_PRED_LO <= lam_star <= LAM_PRED_HI
        lines.append(f"    lam* = {lam_star} (smallest lam with cov_err<={LAM_COV_OK} and "
                     f"R2_lin>={LAM_R2_OK}); predicted in [{LAM_PRED_LO}, {LAM_PRED_HI}] -> "
                     f"{'PASS' if ok else 'FAIL'}")

    # ---- Q3 budget vs wall ----
    lines.append("[Q3 budget vs wall]")
    for label_main, label_ext in [("n=16", "n=16@8000"), ("rho=0.99", "rho=0.99@8000")]:
        r4 = main[main.condition == label_main].R2_lin.mean() if main is not None else float("nan")
        r8 = ext[ext.condition == label_ext].R2_lin.mean()
        d = r8 - r4
        verdict = ("optimization-limited" if d >= BUDGET_OPT_LIMITED
                   else "budget/identifiability wall at this scale" if d <= BUDGET_WALL
                   else "mixed")
        lines.append(f"    {label_main}: R2_lin {r4:.3f} (4000 steps) -> {r8:.3f} (8000) "
                     f"delta={d:+.3f} -> {verdict}")

    # ---- Q4 overcompleteness ----
    lines.append("[Q4 overcompleteness: gap vs K (n=8)]")
    k_rows = []
    if main is not None:
        k_rows.append((8, main[main.condition == "base"]))
        k_rows.append((16, main[main.condition == "K=16"]))
    for K in K_EXT:
        k_rows.append((K, ext[ext.condition == f"K={K}"]))
    k_rows.sort(key=lambda t: t[0])
    gaps = []
    for K, rows in k_rows:
        if len(rows) == 0:
            continue
        g, lk = rows.gap.mean(), rows.leak.mean()
        gaps.append((K, g))
        lines.append(f"    K={K:<3} gap={g:.4f} leak={lk:.4f}")
    mono = all(gaps[i][1] <= gaps[i + 1][1] + 1e-3 for i in range(len(gaps) - 1))
    lines.append(f"    monotone increase predicted -> {'PASS' if mono else 'FAIL'}")

    # ---- Q5 basin frequency ----
    base_rows = [ext[ext.condition == "base_seeds5-9"]]
    base_rows += [ext[ext.condition == f"mw{ws}|base"] for ws in WORLD_SEEDS_EXT]
    if main is not None:
        base_rows.append(main[main.condition == "base"])
    allb = pd.concat(base_rows, ignore_index=True)
    in_basin = allb[allb.rho_hat < BASIN_RHO]
    lines.append(f"[Q5 basin] {len(in_basin)}/{len(allb)} base runs in the suboptimal basin "
                 f"(rho_hat < {BASIN_RHO})")
    if len(in_basin):
        lines.append(f"    basin spectral signature: mean rho_hat={in_basin.rho_hat.mean():.3f} "
                     f"vs mean rho_mod={in_basin.rho_mod.mean():.3f} "
                     f"(large gap => complex eigenvalue pairs: a rotating suboptimal solution)")
        lines.append(f"    basin runs keep cond_m={_mean(in_basin, in_basin.index == in_basin.index, 'cond_m'):.3f} "
                     f"(action axis stays rotational even inside the basin)")
    return "\n".join(lines)


def figures(ext, main, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)

    # fig5: multi-world robustness
    worlds = ([1234] if main is not None else []) + WORLD_SEEDS_EXT
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for i, ws in enumerate(worlds):
        rows = (main[main.condition == "base"] if ws == 1234 and main is not None
                else ext[ext.condition == f"mw{ws}|base"])
        if len(rows) == 0:
            continue
        for ax, col in zip(axes, ("gap", "cond_m")):
            v = rows[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            ax.errorbar([i], [v.mean()], yerr=[v.std()], fmt="o", capsize=4, color="tab:blue")
    for ax, col, thr in zip(axes, ("gap", "cond_m"), (MW_GAP_OK, MW_COND_OK)):
        ax.axhline(thr, color="tab:red", ls="--", lw=1, label=f"confirm bound {thr}")
        ax.set_xticks(range(len(worlds)), [str(w) for w in worlds], fontsize=8)
        ax.set_xlabel("world seed"); ax.set_ylabel(col); ax.legend(fontsize=8)
    axes[0].set_title("state gap across worlds", fontsize=10)
    axes[1].set_title("action-axis cond_m across worlds", fontsize=10)
    plt.tight_layout(); plt.savefig(fig_dir / "fig5_multiworld.png", dpi=150); plt.close(fig)

    # fig6: lambda transition
    pts = []
    if main is not None:
        pts += [(0.05, main[main.condition == "lam=0.05(paper)"]),
                (0.5, main[main.condition == "base"])]
    pts += [(l, ext[ext.condition == f"lam={l}"]) for l in LAMBDAS_EXT]
    pts = sorted([p for p in pts if len(p[1])], key=lambda t: t[0])
    xs = [p[0] for p in pts]
    fig, ax1 = plt.subplots(figsize=(5.6, 3.4))
    ax1.errorbar(xs, [p[1].R2_lin.mean() for p in pts],
                 yerr=[p[1].R2_lin.std() for p in pts], marker="o", capsize=3,
                 color="tab:blue", label="R2_lin")
    ax1.set_xlabel("lambda (SIGReg weight)"); ax1.set_ylabel("R2_lin", color="tab:blue")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.errorbar(xs, [p[1].emb_cov_err.mean() for p in pts],
                 yerr=[p[1].emb_cov_err.std() for p in pts], marker="s", capsize=3,
                 color="tab:orange", label="emb_cov_err")
    ax2.set_ylabel("embedding cov error", color="tab:orange")
    ax1.set_title("the collapse transition: enforcement strength", fontsize=10)
    plt.tight_layout(); plt.savefig(fig_dir / "fig6_lambda_transition.png", dpi=150); plt.close(fig)

    # fig7: gap vs K
    pts = []
    if main is not None:
        pts += [(8, main[main.condition == "base"]), (16, main[main.condition == "K=16"])]
    pts += [(K, ext[ext.condition == f"K={K}"]) for K in K_EXT]
    pts = sorted([p for p in pts if len(p[1])], key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.errorbar([p[0] for p in pts], [p[1].gap.mean() for p in pts],
                yerr=[p[1].gap.std() for p in pts], marker="o", capsize=3, color="tab:purple")
    ax.set_xlabel("embedding dimension K (true n = 8)")
    ax.set_ylabel("gap = R2_lin - R2_orth")
    ax.set_title("overcompleteness weakens the rotational signature", fontsize=10)
    plt.tight_layout(); plt.savefig(fig_dir / "fig7_overcompleteness.png", dpi=150); plt.close(fig)

    # fig8: budget
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    labels, deltas = [], []
    for label_main, label_ext in [("n=16", "n=16@8000"), ("rho=0.99", "rho=0.99@8000")]:
        r4 = main[main.condition == label_main].R2_lin if main is not None else pd.Series(dtype=float)
        r8 = ext[ext.condition == label_ext].R2_lin
        if len(r8) == 0:
            continue
        x = len(labels)
        ax.bar([x - 0.18], [r4.mean()], 0.36, yerr=[r4.std()], capsize=4, color="tab:gray",
               label="4000 steps" if x == 0 else None)
        ax.bar([x + 0.18], [r8.mean()], 0.36, yerr=[r8.std()], capsize=4, color="tab:green",
               label="8000 steps" if x == 0 else None)
        labels.append(label_main)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("R2_lin"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title("budget vs wall", fontsize=10)
    plt.tight_layout(); plt.savefig(fig_dir / "fig8_budget.png", dpi=150); plt.close(fig)
    print(f"figures saved to {fig_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny dimensions, 1 seed, minutes -- code-path check only")
    args = ap.parse_args()

    device = pick_device()
    ensure_dirs()
    if args.smoke:
        cfg = dataclasses.replace(make_config(quick=True), steps=60, warmup=10, train_seeds=(0,))
        global WORLD_SEEDS_EXT, LAMBDAS_EXT, K_EXT, BUDGET_STEPS, EXTRA_BASE_SEEDS
        WORLD_SEEDS_EXT = [2024]
        LAMBDAS_EXT = [0.1]
        K_EXT = [6]
        BUDGET_STEPS = 120
        EXTRA_BASE_SEEDS = (1,)
        out_csv = RESULTS_DIR / "metrics_ext_smoke.csv"
        fig_dir = Path("/tmp/lejepa_ext_smoke_figs")
        main_df = None  # never mix smoke dims with the real main CSV
    else:
        cfg = make_config(quick=False)
        out_csv = RESULTS_DIR / "metrics_ext.csv"
        fig_dir = FIG_DIR
        main_csv = RESULTS_DIR / "metrics.csv"
        main_df = pd.read_csv(main_csv) if main_csv.exists() else None
        if main_df is None:
            print("WARNING: results/metrics.csv not found -- comparisons vs the main sweep "
                  "will be skipped; clone/run the main sweep first for the full report.")

    grid = build_grid_ext(cfg)
    n_runs = sum(len(c.train_seeds) for _, c in grid)
    print(f"extended sweep: {len(grid)} conditions, {n_runs} runs, device {device}")
    t0 = time.time()
    rows = []
    for i, (label, c) in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] {label}", flush=True)
        rows.extend(run_condition(label, c, device))
    ext = pd.DataFrame(rows)
    ext.to_csv(out_csv, index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min; wrote {out_csv} ({len(ext)} runs)")

    figures(ext, main_df, fig_dir)
    print(report(ext, main_df))


if __name__ == "__main__":
    main()
