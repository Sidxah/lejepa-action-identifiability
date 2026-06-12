"""One experimental condition: world + datasets once, then train/measure per seed."""
import time

from .metrics import compute_all_metrics
from .train import train_run
from .world import World, make_datasets

def run_condition(label, cfg, device):
    """Train every seed of one experimental condition. The world and the datasets
    are built once per condition (they are training-seed independent by design),
    then each seed trains its own model and is measured on the shared eval split."""
    world = World(cfg)
    train_data, eval_data = make_datasets(world, cfg)
    rows = []
    for seed in cfg.train_seeds:
        t0 = time.time()
        enc, pred, hist = train_run(cfg, train_data, device, seed)
        row = compute_all_metrics(enc, pred, eval_data, world, cfg, device)
        row.update(
            condition=label, seed=seed, converged=hist["converged"],
            wall_s=round(time.time() - t0, 2),
            L_pred_final=float(hist["ema_L_pred"][-1]),
            L_sigreg_final=float(hist["L_sigreg"][-1]),
            quick=cfg.quick, n=cfg.n, m=cfg.m, rho=cfg.rho, K=cfg.K, D=cfg.D,
            nonlinearity_strength=cfg.nonlinearity_strength, lam=cfg.lam,
            action_scale=cfg.action_scale, action_rank=cfg.action_rank,
            noise_mode=cfg.noise_mode, action_conditioned=cfg.action_conditioned,
            steps=cfg.steps, world_seed=cfg.world_seed,
        )
        rows.append(row)
        print(f"    seed {seed}: R2_lin={row['R2_lin']:.3f} R2_orth={row['R2_orth']:.3f} "
              f"cond_m={row['cond_m']:.3g} rho_hat={row['rho_hat']:.3f} "
              f"conv={row['converged']} ({row['wall_s']:.0f}s)", flush=True)
    return rows

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
