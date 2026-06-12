"""Experiment configuration, seeding, device selection, output paths."""
import os
from dataclasses import dataclass

import numpy as np
import torch
from pathlib import Path

# Shared imports and utilities (Phase 2: common header of every src/ module).


RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"


def ensure_dirs():
    """Create results/ and results/figures/ (idempotent)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def pick_device():
    """cuda > mps > cpu; LEJEPA_DEVICE overrides (e.g. to benchmark cpu vs mps)."""
    forced = os.environ.get("LEJEPA_DEVICE", "")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    """Seed every RNG that training touches. The synthetic world never uses the
    global RNGs (it draws from its own explicit torch.Generator), so calling
    this with a *training* seed cannot perturb the world -- the seed separation
    described in Section 2 is enforced structurally, not by discipline."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Exact determinism on CPU; warn-only on CUDA/MPS where some kernels
    # have no deterministic variant. Conclusions rest on multi-seed stats.
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True)
class Config:
    """Every knob of the experiment, in one frozen object.

    Defaults are the FULL-mode settings from the experiment spec. QUICK mode
    (make_config) shrinks dimensions/steps/seeds so the whole notebook runs in
    seconds as a smoke test. Ablations are dataclasses.replace()-modified
    copies of the base config, so every run's full configuration is logged.
    """

    quick: bool = False
    # ---- world (ground truth) ----
    n: int = 8                  # true latent dimension
    m: int = 2                  # action dimension
    rho: float = 0.9            # OU mean-reversion / state-transition scalar
    D: int = 64                 # observation dimension
    nonlinearity_strength: float = 1.0  # 0 = linear g, 1 = pure frozen MLP
    kappa: float = 0.5          # action signal fraction: lam_max(B Sigma_a B^T) / (1 - rho^2)
    action_scale: float = 1.0   # std of action components (excitation knob)
    action_rank: int = 0        # 0 = full-rank actions; r>0 = actions confined to r-dim subspace
    noise_mode: str = "balanced"  # "balanced" (z ~ N(0,I) exactly) | "literal" (spec's sqrt(1-rho^2) I noise)
    action_conditioned: bool = True  # False = symmetric control (B = 0, predictor ignores actions)
    world_seed: int = 1234
    n_train: int = 50_000
    n_eval: int = 10_000
    traj_len: int = 100         # transitions per trajectory (many short, independent chains)
    n_calib: int = 8192         # world-seed sample used to standardize the observation map
    # ---- model ----
    K: int = 8                  # embedding dimension (= n by default, comparable by orthogonal map)
    hidden: int = 256
    # ---- training ----
    steps: int = 4000
    warmup: int = 100
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 0.01  # encoder weight matrices only (see Section 6 markdown)
    lam: float = 0.5            # SIGReg weight: L = (1-lam) L_pred + lam L_sigreg.
                                # 0.5, NOT the paper's 0.05: at toy scale the soft penalty
                                # at 0.05 sits at a partial-collapse equilibrium (dead
                                # embedding dims) -- measured by the lam=0.05 ablation;
                                # see the Section 6 markdown for the calibration argument.
    num_slices: int = 1024
    train_seeds: tuple = (0, 1, 2, 3, 4)


def make_config(quick):
    """Single derivation point for QUICK vs FULL. No other cell branches on QUICK."""
    if not quick:
        return Config(quick=False)
    return Config(
        quick=True,
        n=4, m=2, D=16, K=4, hidden=128,
        n_train=2_000, n_eval=500, traj_len=50, n_calib=2048,
        steps=150, warmup=15, batch_size=256, num_slices=128,
        train_seeds=(0, 1),
    )
