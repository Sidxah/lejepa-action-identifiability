#!/usr/bin/env python3
"""End-to-end smoke test (spec §7): runs in seconds, validates every code path.

    python tests/smoke_test.py

Three layers, by increasing integration:
  1. SIGReg sanity -- near its O(1) null on N(0, I) samples, large on
     collapsed / anisotropic / shifted ones.
  2. Metrics self-test on CONSTRUCTED cases with known answers (pure rotation,
     known stretch, K > n embedding) -- validates the exact functions the
     results are computed with, no training involved.
  3. A tiny full experiment (n=4, ~40 steps, 1 seed): world -> train ->
     metrics, asserting the row is well-formed. Correctness of the pipeline,
     NOT science -- the numbers are meaningless at this scale.
"""
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import make_config, pick_device
from src.metrics import m1_state, m2_action_axis, m3_dynamics
from src.run import run_condition
from src.sigreg import sigreg


def test_sigreg():
    g = torch.Generator().manual_seed(0)
    N, K = 4096, 8
    x_good = torch.randn(N, K, generator=g)
    vals = {
        "N(0,I)": float(sigreg(x_good, step=0)),
        "collapsed": float(sigreg(0.01 * torch.randn(N, K, generator=g), step=0)),
        "anisotropic": float(sigreg(x_good * torch.tensor([3.0] + [1.0] * (K - 1)), step=0)),
        "shifted": float(sigreg(x_good + 1.5, step=0)),
    }
    for name, v in vals.items():
        print(f"  sigreg({name:<12}) = {v:10.3f}")
    assert vals["N(0,I)"] < vals["collapsed"]
    assert vals["N(0,I)"] < vals["anisotropic"]
    assert vals["N(0,I)"] < vals["shifted"]


def test_metrics():
    rng = np.random.default_rng(0)
    n, m, N, rho = 8, 2, 4000, 0.9
    Z = rng.standard_normal((N, n))
    O, _ = np.linalg.qr(rng.standard_normal((n, n)))
    B = rng.standard_normal((n, m))

    m1, Q = m1_state(Z @ O.T, Z)                     # pure rotation: perfect
    m2 = m2_action_axis(Q, O @ B, B)
    m3 = m3_dynamics(Q, O @ (rho * np.eye(n)) @ O.T, rho)
    assert m1["R2_orth"] > 0.999 and m2["cond_m"] < 1.001
    assert m2["theta_max_deg"] < 0.1 and abs(m3["rho_hat"] - rho) < 1e-6 and m3["D_rel"] < 1e-6

    S = np.diag([2.0] + [1.0] * (n - 1))             # known stretch: gap opens
    m1s, Qs = m1_state(Z @ (S @ O).T, Z)
    m2s = m2_action_axis(Qs, S @ O @ B, B)
    assert m1s["R2_lin"] > 0.999 and m1s["gap"] > 0.01 and 1.4 < m2s["cond_m"] < 2.6

    E, _ = np.linalg.qr(rng.standard_normal((16, n)))  # K > n: semi-orthogonal Procrustes
    m1e, Qe = m1_state(Z @ E.T, Z)
    m3e = m3_dynamics(Qe, E @ (rho * np.eye(n)) @ E.T, rho)
    assert m1e["R2_orth"] > 0.999 and m3e["leak"] < 1e-6 and abs(m3e["rho_hat"] - rho) < 1e-6
    print("  metrics self-test passed (rotation / stretch / K>n)")


def test_pipeline():
    cfg = dataclasses.replace(make_config(quick=True), steps=40, warmup=5, train_seeds=(0,))
    rows = run_condition("smoke", cfg, pick_device())
    row = rows[0]
    for key in ("R2_lin", "R2_orth", "gap", "cond_m", "theta_max_deg",
                "rho_hat", "D_rel", "emb_cov_err", "converged"):
        assert key in row, f"missing metric column {key}"
    assert math.isfinite(row["R2_lin"]) and -1.0 <= row["R2_lin"] <= 1.0
    assert math.isfinite(row["rho_hat"])
    print("  end-to-end pipeline row OK (n=4, 40 steps -- correctness only, not science)")


if __name__ == "__main__":
    print("[1/3] SIGReg sanity")
    test_sigreg()
    print("[2/3] metrics self-test")
    test_metrics()
    print("[3/3] tiny end-to-end pipeline")
    test_pipeline()
    print("smoke test passed.")
