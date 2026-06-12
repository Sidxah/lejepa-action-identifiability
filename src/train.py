"""Training loop: LeJEPA recipe (no EMA, no stop-gradient), AdamW + warmup-cosine."""
import math

import numpy as np
import torch

from .config import set_seed
from .model import Encoder, LinearPredictor
from .sigreg import sigreg

def lr_at(step, total_steps, warmup, base_lr):
    """Linear warmup then cosine decay to base_lr/100."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * p)))


def train_run(cfg, train_data, device, train_seed):
    """One full training run. Returns (encoder, predictor, history).

    history: dict of per-step arrays (lr, L_pred, L_sigreg, L_total) plus a
    pre-registered boolean `converged` (EMA-smoothed L_pred changes < 2%
    over the final 20% of steps).
    """
    set_seed(train_seed)  # init + shuffling only; the world has its own generators
    enc = Encoder(cfg.D, cfg.K, cfg.hidden).to(device)
    pred = LinearPredictor(cfg.K, cfg.m, cfg.action_conditioned).to(device)

    # Weight decay on encoder weight MATRICES only. Decay on predictor.Rhat would
    # bias the recovered rho_hat low -- a metric contamination, not regularization.
    decay = [p for p in enc.parameters() if p.ndim >= 2]
    no_decay = [p for p in enc.parameters() if p.ndim < 2] + list(pred.parameters())
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )

    x_t, a_t, x_t1 = train_data["x_t"], train_data["a_t"], train_data["x_t1"]
    n_train = x_t.shape[0]
    perm = torch.randperm(n_train)  # epoch-wise shuffling (seeded by set_seed above)
    ptr = 0
    hist = {"lr": [], "L_pred": [], "L_sigreg": [], "L_total": []}

    for step in range(cfg.steps):
        lr = lr_at(step, cfg.steps, cfg.warmup, cfg.lr)
        for grp in opt.param_groups:
            grp["lr"] = lr
        if ptr + cfg.batch_size > n_train:
            perm = torch.randperm(n_train)
            ptr = 0
        idx = perm[ptr:ptr + cfg.batch_size]
        ptr += cfg.batch_size

        xb = x_t[idx].to(device)
        ab = a_t[idx].to(device)
        yb = x_t1[idx].to(device)

        z1 = enc(xb)
        z2 = enc(yb)              # NO stop-gradient on the target (LeJEPA recipe)
        L_pred = ((pred(z1, ab) - z2) ** 2).sum(dim=1).mean()
        if cfg.lam > 0:
            L_sig = 0.5 * (sigreg(z1, step, cfg.num_slices) + sigreg(z2, step, cfg.num_slices))
            loss = (1.0 - cfg.lam) * L_pred + cfg.lam * L_sig
        else:
            loss = L_pred          # SIGReg-off control: pure prediction
            with torch.no_grad():  # still logged, so collapse is visible in the curves
                L_sig = 0.5 * (sigreg(z1, step, cfg.num_slices) + sigreg(z2, step, cfg.num_slices))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        hist["lr"].append(lr)
        hist["L_pred"].append(float(L_pred.detach()))
        hist["L_sigreg"].append(float(L_sig.detach()))
        hist["L_total"].append(float(loss.detach()))

    for k in list(hist):
        hist[k] = np.asarray(hist[k])
    # Pre-registered convergence criterion: EMA-smoothed L_pred relative change
    # over the final 20% of steps < 2%. Non-converged runs are flagged, reported,
    # and excluded from the verdict aggregation (never silently dropped).
    span = max(10, cfg.steps // 20)
    alpha = 2.0 / (span + 1)
    ema = np.empty_like(hist["L_pred"])
    acc = hist["L_pred"][0]
    for i, v in enumerate(hist["L_pred"]):
        acc = alpha * v + (1 - alpha) * acc
        ema[i] = acc
    i80 = int(0.8 * (cfg.steps - 1))
    hist["converged"] = bool(abs(ema[i80] - ema[-1]) / max(abs(ema[i80]), 1e-8) < 0.02)
    hist["ema_L_pred"] = ema
    return enc, pred, hist
