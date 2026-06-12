#!/usr/bin/env python3
"""Phase-2 extraction readiness check + single FULL-mode convergence probe.

1. Reads the built notebook and verifies every code cell starts with a module
   tag or the driver marker (the Phase-2 extraction contract).
2. Concatenates the module-tagged cells per tag (exactly what the Phase-2
   copy-out will do) and execs them in one namespace -- proving the module
   code is self-contained apart from the shared setup header.
3. Optionally (--full-probe) runs ONE full-mode base-condition training run
   and prints its metrics row, validating the FULL hyperparameters without
   committing to the whole sweep.

Usage: python tools/extract_check.py NOTEBOOK.ipynb [--full-probe] [--seed 0]
"""
import argparse
import time

import nbformat

MODULE_ORDER = ["setup", "world", "model", "sigreg", "train", "metrics", "run", "sweep"]
DRIVER = "# -- driver (notebook-only) --"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebook")
    ap.add_argument("--full-probe", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    nb = nbformat.read(args.notebook, as_version=4)
    modules = {m: [] for m in MODULE_ORDER}
    n_driver = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        first = cell.source.splitlines()[0].strip()
        if first == DRIVER:
            n_driver += 1
            continue
        tag = None
        for m in MODULE_ORDER:
            if first == f"# -- module: {m} --":
                tag = m
                break
        assert tag is not None, f"code cell {i} has no module tag: {first!r}"
        modules[tag].append(cell.source)

    print("extraction map:")
    for m in MODULE_ORDER:
        print(f"  {m:<8} {len(modules[m])} cell(s)")
    print(f"  driver   {n_driver} cell(s) (notebook-only, not extracted)")

    ns = {}
    for m in MODULE_ORDER:
        body = "\n\n".join(modules[m])
        exec(compile(body, f"src/{m}.py", "exec"), ns)  # noqa: S102 -- our own code
    print("module bodies exec cleanly in extraction order.")

    if not args.full_probe:
        return

    cfg = ns["Config"](quick=False)  # FULL defaults
    device = ns["pick_device"]()
    print(f"\nFULL probe: base condition, seed {args.seed}, device {device}")
    world = ns["World"](cfg)
    train_data, eval_data = ns["make_datasets"](world, cfg)
    t0 = time.time()
    enc, pred, hist = ns["train_run"](cfg, train_data, device, args.seed)
    print(f"trained {cfg.steps} steps in {time.time() - t0:.0f}s "
          f"(converged={hist['converged']}, "
          f"L_pred {hist['L_pred'][0]:.2f} -> {hist['ema_L_pred'][-1]:.3f})")
    row = ns["compute_all_metrics"](enc, pred, eval_data, world, cfg, device)
    for k in ("R2_lin", "R2_orth", "gap", "proc_scale", "cond_m", "cond_La",
              "theta_max_deg", "norm_ratio", "rho_hat", "rho_mod", "D_rel",
              "emb_mean_norm", "emb_cov_err", "pred_bias_norm"):
        print(f"  {k:<15} = {row[k]:.4f}")


if __name__ == "__main__":
    main()
