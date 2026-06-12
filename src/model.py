"""Encoder and action-conditioned linear predictor."""
import math

import torch
import torch.nn as nn

class Encoder(nn.Module):
    """f_theta: R^D -> R^K. Small MLP per the spec; nothing exotic."""

    def __init__(self, D, K, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, K),
        )

    def forward(self, x):
        return self.net(x)


class LinearPredictor(nn.Module):
    """P(zhat, a) = Rhat zhat + Bhat a + c  (affine in state, affine in action).

    Kept linear ON PURPOSE: the learned dynamics (Rhat, Bhat) are then read off
    directly as matrices -- they ARE the quantities metrics M2/M3 evaluate.
    action_conditioned=False gives the symmetric control P(zhat) = Rhat zhat + c.

    Init: Rhat ~ N(0, 1/K) entries (neutral, varied by the training seed);
    Bhat starts at 0 so the action effect is purely learned from data -- with no
    weight decay on the predictor (see training cell), any direction of Bhat
    that receives no gradient (e.g. unexcited action directions in the rank-1
    ablation) stays exactly at 0, which is the honest signature of
    unidentifiability rather than a decayed-toward-zero artifact.
    """

    def __init__(self, K, m, action_conditioned=True):
        super().__init__()
        self.action_conditioned = action_conditioned
        self.Rhat = nn.Parameter(torch.randn(K, K) / math.sqrt(K))
        self.Bhat = nn.Parameter(torch.zeros(K, m)) if action_conditioned else None
        self.c = nn.Parameter(torch.zeros(K))

    def forward(self, zhat, a):
        out = zhat @ self.Rhat.T + self.c
        if self.action_conditioned:
            out = out + a @ self.Bhat.T
        return out
