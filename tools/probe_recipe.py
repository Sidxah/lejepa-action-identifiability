#!/usr/bin/env python3
"""Recipe-calibration probe: train one FULL-mode base run with overridable
hyperparameters, logging R2_lin / cov-isotropy / losses along the trajectory.

Usage:
  python tools/probe_recipe.py NOTEBOOK.ipynb --steps 3000 --lam 0.05 [--lr 2e-3]
         [--device cpu|mps] [--seed 0] [--snap 300]
"""
import argparse
import math
import time

import nbformat
import numpy as np

MODULE_ORDER = ["setup", "world", "model", "sigreg", "train", "metrics", "run", "sweep"]


def load_modules(path):
    nb = nbformat.read(path, as_version=4)
    ns = {}
    for m in MODULE_ORDER:
        srcs = [c.source for c in nb.cells if c.cell_type == "code"
                and c.source.splitlines()[0].strip() == f"# -- module: {m} --"]
        exec(compile("\n\n".join(srcs), f"src/{m}.py", "exec"), ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snap", type=int, default=300)
    args = ap.parse_args()

    import os
    if args.device:
        os.environ["LEJEPA_DEVICE"] = args.device
    ns = load_modules(args.notebook)
    import dataclasses
    import torch

    cfg = ns["Config"](quick=False)
    cfg = dataclasses.replace(cfg, steps=args.steps, lam=args.lam, lr=args.lr,
                              warmup=max(100, args.steps // 30))
    device = ns["pick_device"]()
    print(f"probe: steps={cfg.steps} lam={cfg.lam} lr={cfg.lr} device={device}")

    world = ns["World"](cfg)
    train_data, eval_data = ns["make_datasets"](world, cfg)
    ns["set_seed"](args.seed)
    enc = ns["Encoder"](cfg.D, cfg.K, cfg.hidden).to(device)
    pred = ns["LinearPredictor"](cfg.K, cfg.m, cfg.action_conditioned).to(device)
    decay = [p for p in enc.parameters() if p.ndim >= 2]
    no_decay = [p for p in enc.parameters() if p.ndim < 2] + list(pred.parameters())
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)

    x_t, a_t, x_t1 = train_data["x_t"], train_data["a_t"], train_data["x_t1"]
    Z_eval = eval_data["z_t"].double().numpy()
    Xe = eval_data["x_t"]
    n_train = x_t.shape[0]
    perm = torch.randperm(n_train)
    ptr = 0
    sigreg = ns["sigreg"]
    t0 = time.time()

    def snapshot(step, L_pred, L_sig):
        enc.eval()
        with torch.no_grad():
            Zh = enc(Xe.to(device)).cpu().double().numpy()
        enc.train()
        Zc = Z_eval - Z_eval.mean(0)
        Zhc = Zh - Zh.mean(0)
        W, *_ = np.linalg.lstsq(Zhc, Zc, rcond=None)
        r2 = 1.0 - ((Zc - Zhc @ W) ** 2).sum() / (Zc ** 2).sum()
        C = np.cov(Zh.T)
        cov_err = np.linalg.norm(C - np.eye(cfg.K)) / math.sqrt(cfg.K)
        ev = np.linalg.eigvalsh(C)
        print(f"  step {step:>5}  L_pred={L_pred:7.3f}  L_sig={L_sig:9.3f}  "
              f"R2_lin={r2:6.3f}  cov_err={cov_err:5.3f}  "
              f"cov_eig=[{ev.min():.2f},{ev.max():.2f}]  ({time.time()-t0:.0f}s)",
              flush=True)

    for step in range(cfg.steps):
        lr = ns["lr_at"](step, cfg.steps, cfg.warmup, cfg.lr)
        for grp in opt.param_groups:
            grp["lr"] = lr
        if ptr + cfg.batch_size > n_train:
            perm = torch.randperm(n_train)
            ptr = 0
        idx = perm[ptr:ptr + cfg.batch_size]
        ptr += cfg.batch_size
        xb, ab, yb = x_t[idx].to(device), a_t[idx].to(device), x_t1[idx].to(device)
        z1, z2 = enc(xb), enc(yb)
        L_pred = ((pred(z1, ab) - z2) ** 2).sum(dim=1).mean()
        L_sig = 0.5 * (sigreg(z1, step, cfg.num_slices) + sigreg(z2, step, cfg.num_slices))
        loss = (1.0 - cfg.lam) * L_pred + cfg.lam * L_sig
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.snap == 0 or step == cfg.steps - 1:
            snapshot(step, float(L_pred), float(L_sig))

    # final metrics row
    row = ns["compute_all_metrics"](enc, pred, eval_data, world, cfg, device)
    keys = ("R2_lin", "R2_orth", "gap", "proc_scale", "cond_m", "theta_max_deg",
            "rho_hat", "rho_mod", "D_rel", "emb_cov_err")
    print("final:", {k: round(row[k], 4) for k in keys})


if __name__ == "__main__":
    main()
