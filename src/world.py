"""Synthetic world: ground-truth latent dynamics + frozen observation map."""
import math

import torch
import torch.nn.functional as F

class World:
    """Frozen synthetic world: ground-truth latent dynamics + observation map.

        z_{t+1} = rho * z_t + B a_t + eta_t        (latent, linear-Gaussian)
        x_t     = g(z_t)                           (frozen nonlinear observation)

    Everything random here is drawn from ONE explicit torch.Generator seeded with
    cfg.world_seed -- never from the global RNG -- so the world is bit-identical
    across training seeds and immune to set_seed() calls during training.

    noise_mode:
      "balanced": eta ~ N(0, Lambda), Lambda = (1-rho^2) I - B Sigma_a B^T.
          The unique Gaussian noise making N(0, I_n) *exactly* stationary
          (plug into Sigma = rho^2 Sigma + B Sigma_a B^T + Lambda).
          Reduces to the spec's OU noise when B = 0.
      "literal":  eta ~ N(0, (1-rho^2) I) exactly as written in the spec.
          Stationary marginal becomes N(0, Sigma_lit),
          Sigma_lit = I + B Sigma_a B^T / (1-rho^2)  (anisotropic).
          Kept as an ablation so the whitening artifact is measured, not assumed.

    In both modes z_0 is drawn from the exact stationary law, so rollouts need
    no burn-in and every tuple is an unbiased stationary sample.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        n, m, rho = cfg.n, cfg.m, cfg.rho
        g = torch.Generator().manual_seed(cfg.world_seed)

        # ---- true action effect B = b * U (unit direction columns, common scale b) ----
        # b is calibrated so kappa = lam_max(B Sigma_a B^T)/(1-rho^2) at BASE settings
        # (action_scale=1, full rank): a dimensionless, rho-portable "moderate norm".
        Graw = torch.randn(n, m, generator=g)
        Udir = Graw / Graw.norm(dim=0, keepdim=True)
        lam_max_dir = float(torch.linalg.eigvalsh(Udir.T @ Udir).max())  # = lam_max(U U^T)
        self.b_scale = math.sqrt(cfg.kappa * (1.0 - rho**2) / lam_max_dir)
        if cfg.action_conditioned:
            self.B = self.b_scale * Udir
        else:
            self.B = torch.zeros(n, m)  # symmetric control: pure OU world
            # (Graw was still drawn above, so the generator state -- and hence the
            #  observation map g below -- is identical to the action-conditioned world.)

        # ---- action distribution: a = Vr c, c ~ N(0, action_scale^2 I_r) ----
        r = cfg.action_rank if cfg.action_rank > 0 else m
        Vr_raw = torch.randn(m, r, generator=g)  # drawn in all cases (RNG alignment)
        if r < m:
            self.Vr, _ = torch.linalg.qr(Vr_raw)  # m x r orthonormal: rank-deficient excitation
        else:
            self.Vr = torch.eye(m)
        self.action_rank = r
        Sigma_a = cfg.action_scale**2 * (self.Vr @ self.Vr.T)

        # ---- transition noise (see class docstring) ----
        BSB = self.B @ Sigma_a @ self.B.T
        evals, evecs = torch.linalg.eigh(BSB)
        if cfg.noise_mode == "balanced":
            assert float(evals.max()) <= 0.95 * (1.0 - rho**2) + 1e-9, (
                "infeasible world: action variance exceeds the innovation budget; "
                "lower kappa or action_scale")
            lam_sqrt = torch.sqrt(torch.clamp((1.0 - rho**2) - evals, min=0.0))
            self.noise_chol = evecs @ torch.diag(lam_sqrt) @ evecs.T  # Lambda^{1/2} (symmetric)
            self.Sigma = torch.eye(n)                                  # exact stationary cov
        elif cfg.noise_mode == "literal":
            self.noise_chol = math.sqrt(1.0 - rho**2) * torch.eye(n)
            self.Sigma = torch.eye(n) + BSB / (1.0 - rho**2)
        else:
            raise ValueError(f"unknown noise_mode {cfg.noise_mode!r}")
        sevals, sevecs = torch.linalg.eigh(self.Sigma)
        self.Sigma_sqrt = sevecs @ torch.diag(torch.sqrt(torch.clamp(sevals, min=0.0))) @ sevecs.T
        # Analytic whitening artifact IF the literal-noise world were used with this B:
        # cond(Sigma_lit^{1/2}). For the literal ablation this predicts the M2 inflation;
        # for balanced worlds it is the counterfactual being avoided.
        lit_evals = torch.linalg.eigvalsh(torch.eye(n) + BSB / (1.0 - rho**2))
        self.analytic_whiten_cond = float(torch.sqrt(lit_evals.max() / lit_evals.min()))

        # ---- frozen observation map g_s = std[(1-s) g~_lin + s g~_mlp] ----
        Wbig = torch.randn(cfg.D, n, generator=g)
        self.W_lin, _ = torch.linalg.qr(Wbig)  # D x n semi-orthogonal: clean linear endpoint
        widths = [n, 64, 64, cfg.D]            # spec's MLP, weights frozen at creation
        self.mlp_weights = []
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            Wl = torch.randn(fan_out, fan_in, generator=g) / math.sqrt(fan_in)
            bl = 0.1 * torch.randn(fan_out, generator=g)
            self.mlp_weights.append((Wl, bl))
        # Two-stage standardization on a world-seeded calibration sample: first each
        # endpoint, then the mixture (the endpoints are correlated, so the mixture's
        # variance depends on s; stage 2 makes every s give mean-0/std-1 outputs).
        Zc = torch.randn(cfg.n_calib, n, generator=g)
        lin, mlp = Zc @ self.W_lin.T, self._mlp_raw(Zc)
        self.mu_lin, self.sd_lin = lin.mean(0), lin.std(0).clamp_min(1e-6)
        self.mu_mlp, self.sd_mlp = mlp.mean(0), mlp.std(0).clamp_min(1e-6)
        s = cfg.nonlinearity_strength
        mix = (1 - s) * (lin - self.mu_lin) / self.sd_lin + s * (mlp - self.mu_mlp) / self.sd_mlp
        self.mu_mix, self.sd_mix = mix.mean(0), mix.std(0).clamp_min(1e-6)

    def _mlp_raw(self, z):
        h = z
        for i, (W, b) in enumerate(self.mlp_weights):
            h = h @ W.T + b
            if i < len(self.mlp_weights) - 1:
                h = F.gelu(h)
        return h

    @torch.no_grad()
    def observe(self, z):
        """x = g_s(z). Frozen: plain tensors, no parameters, no gradients."""
        s = self.cfg.nonlinearity_strength
        lin = (z @ self.W_lin.T - self.mu_lin) / self.sd_lin
        mlp = (self._mlp_raw(z) - self.mu_mlp) / self.sd_mlp
        return ((1 - s) * lin + s * mlp - self.mu_mix) / self.sd_mix

    def sample_actions(self, n_rows, g):
        c = self.cfg.action_scale * torch.randn(n_rows, self.action_rank, generator=g)
        return c @ self.Vr.T

    def rollout(self, n_tuples, seed):
        """Roll independent trajectories; return n_tuples of (x_t, a_t, x_{t+1})
        plus the aligned true latents (z_t, z_{t+1}) for metrics."""
        cfg = self.cfg
        g = torch.Generator().manual_seed(seed)
        L = cfg.traj_len
        n_traj = math.ceil(n_tuples / L)
        z = torch.randn(n_traj, cfg.n, generator=g) @ self.Sigma_sqrt.T  # exact stationary start
        Zt, At, Zt1 = [], [], []
        for _ in range(L):
            a = self.sample_actions(n_traj, g)
            eta = torch.randn(n_traj, cfg.n, generator=g) @ self.noise_chol.T
            z_next = cfg.rho * z + a @ self.B.T + eta
            Zt.append(z); At.append(a); Zt1.append(z_next)
            z = z_next
        # (L, n_traj, .) -> trajectory-major -> first n_tuples rows
        z_t = torch.stack(Zt).transpose(0, 1).reshape(n_traj * L, cfg.n)[:n_tuples]
        a_t = torch.stack(At).transpose(0, 1).reshape(n_traj * L, cfg.m)[:n_tuples]
        z_t1 = torch.stack(Zt1).transpose(0, 1).reshape(n_traj * L, cfg.n)[:n_tuples]
        return {"x_t": self.observe(z_t), "a_t": a_t, "x_t1": self.observe(z_t1),
                "z_t": z_t, "z_t1": z_t1}


def make_datasets(world, cfg):
    """Train/eval splits from disjoint noise substreams of the world seed.
    Identical across training seeds by construction; eval is never trained on."""
    train = world.rollout(cfg.n_train, cfg.world_seed + 7001)
    eval_ = world.rollout(cfg.n_eval, cfg.world_seed + 7002)
    return train, eval_
