# SPEC: Does the orthogonal identifiability ambiguity survive action-conditioning?

A minimal, controlled, ground-truth experiment. Reference-grade quality, toy scale.

Author / owner: Sid Ahmed Bouamama. This file is the specification to implement. Implement it faithfully, keep it small, and make every result reproducible from a seed.

## 0. One-paragraph context (so you understand what you are building)

LeJEPA (Balestriero and LeCun, 2025) shows that self-supervised joint-embedding learning works when embeddings are pushed to an isotropic Gaussian via a regulariser called SIGReg, with no teacher-student and no stop-gradient. A 2026 result (Klindt, LeCun, Balestriero) proves that, in a Gaussian world with stationary additive-noise (Ornstein-Uhlenbeck) transitions and a nonlinear observation map, LeJEPA-style training recovers the true latents up to an orthogonal transformation (a rotation or reflection), and recovers the dynamics in closed form. That guarantee covers only the state and the symmetric (no-action) case. Separately, V-JEPA 2 (Assran et al., 2025), an action-conditioned latent world model, was observed empirically to recover the action axis up to a near-rotation (reported condition number around 1.5). The open question this experiment targets: does the orthogonal ambiguity of the identifiability guarantee survive the passage to the action-conditioned regime? We test it directly, in a synthetic world where we own the ground truth.

Both outcomes are a result:
- If recovery stays orthogonal (state up to rotation, action axis up to rotation, condition number near 1), the hypothesis is confirmed and the identifiability picture extends empirically to the action-conditioned case.
- If recovery is worse than orthogonal (large condition number, distortion beyond rotation), then identifiability is more fragile under action-conditioning, which sharpens the open problem. This is also a finding, report it plainly.

## 1. The synthetic world (ground truth, fixed)

Implement a generator with a fixed random seed for the world parameters, separate from training seeds.

- Latent state `z` in R^n. Default `n = 8`. Marginal law `z ~ N(0, I_n)`.
- Action `a` in R^m. Default `m = 2`. Sample actions i.i.d. from `N(0, I_m)` (this is the persistent-excitation / sufficient-excitation condition in its simplest form; expose it as a knob, see ablations).
- Action-conditioned transition (linear-Gaussian, keeps the OU marginal when `B = 0`):
  `z_{t+1} = rho * z_t + B @ a_t + sqrt(1 - rho^2) * eta`, with `eta ~ N(0, I_n)`, `rho` in (0,1), default `rho = 0.9`.
  - `B` is a fixed `n x m` matrix (the true action effect / true action axis). Generate it once with the world seed, columns of moderate norm. This is the ground-truth quantity the action-axis metric compares against.
- Observation map `x = g(z)`, with `g` a FIXED random nonlinear map (the world simulator, not learned, frozen). Use a small random MLP: `Linear(n -> 64) -> GELU -> Linear(64 -> 64) -> GELU -> Linear(64 -> D)`, `D = 64`, weights drawn once from the world seed and never trained. Expose a `nonlinearity_strength` knob (see ablations) that interpolates `g` from linear (identity-like) to strongly nonlinear, e.g. by mixing `g_linear(z)` and `g_mlp(z)`.
- Data: roll out trajectories to produce tuples `(x_t, a_t, x_{t+1})`. Generate enough tuples for stable estimates (e.g. 50k train, 10k held-out eval). Hold out a clean eval split, never used in training, for all metrics.

Save the ground-truth `B`, `rho`, the world seed, and the held-out true latents `z` aligned to their `x`, so metrics can be computed.

## 2. The model (what is learned)

This is the action-conditioned regime, so the predictor is justified and is kept (unlike symmetric LeJEPA which drops it). Follow LeJEPA's heuristic-free recipe: no EMA target encoder, no stop-gradient, no teacher-student. SIGReg prevents collapse.

- Encoder `f_theta: R^D -> R^K`. Small MLP, e.g. `Linear(D -> 256) -> GELU -> Linear(256 -> 256) -> GELU -> Linear(256 -> K)`. Default `K = n` (so learned and true latents live in the same dimension and can be compared by an orthogonal map). Expose `K` as a knob.
- Action-conditioned predictor `P: (R^K, R^m) -> R^K`. Keep it deliberately simple so the recovered dynamics are interpretable. Default: an affine-in-state, affine-in-action map `P(zhat, a) = Rhat @ zhat + Bhat_pred @ a`, i.e. a learnable `K x K` matrix `Rhat` and a learnable `K x m` matrix `Bhat_pred`. Optionally a small MLP variant as an ablation, but the linear predictor is the main setting because it lets us read off the learned dynamics directly.
- Loss (LeJEPA form):
  `L = (1 - lambda) * L_pred + lambda * L_sigreg`
  - `L_pred = mean || P(f(x_t), a_t) - f(x_{t+1}) ||^2`, computed with NO stop-gradient on the target `f(x_{t+1})` (LeJEPA recipe).
  - `L_sigreg = mean over the batch embeddings of SIGReg(...)`, applied to the set `{ f(x_t) }` (and you may also apply it to `{ f(x_{t+1}) }` and average). Default `lambda = 0.05`.

### SIGReg (the load-bearing component, implement carefully)

Follow LeJEPA Algorithm 1 (Epps-Pulley sliced characteristic-function matching to an isotropic Gaussian). Reference implementation to reproduce (single-GPU; drop the all-reduce):

```python
def sigreg(x, step, num_slices=1024):
    # x: (N, K) batch of embeddings. Targets isotropic Gaussian N(0, I_K).
    g = torch.Generator(device=x.device); g.manual_seed(step)   # resample directions each step
    A = torch.randn((x.size(1), num_slices), generator=g, device=x.device)
    A = A / A.norm(p=2, dim=0, keepdim=True)                     # unit directions on the sphere
    t = torch.linspace(-5, 5, 17, device=x.device)              # quadrature points
    cf_target = torch.exp(-0.5 * t**2)                          # CF of N(0,1), per direction
    proj = (x @ A).unsqueeze(2) * t                             # (N, num_slices, T)
    ecf = torch.exp(1j * proj).mean(0)                          # empirical CF, (num_slices, T)
    err = (ecf - cf_target).abs().square() * cf_target         # weighted L2 in CF space
    per_slice = torch.trapz(err, t, dim=1) * x.size(0)
    return per_slice.mean()
```

Notes: any 1D projection of an isotropic Gaussian is `N(0,1)`, which is why matching each sliced empirical CF to the `N(0,1)` CF enforces the full isotropic Gaussian. Resample the directions `A` at every step (seed by global step). Defaults from the paper: 17 quadrature points on `[-5, 5]`, around 1024 slices, `lambda = 0.05`. If anything here is ambiguous, mirror the official LeJEPA repo for SIGReg, but keep this single-GPU and dependency-light.

## 3. Training

- Optimiser AdamW, cosine schedule with short warmup. Batch of tuples (e.g. 512). Train for a small number of steps, enough to converge (this is tiny, expect minutes). 
- Fixed, logged training seed, separate from the world seed. Set deterministic flags.
- Log `L_pred`, `L_sigreg`, total loss. A converged run should show `L_pred` dropping and the embeddings becoming close to isotropic Gaussian (you can sanity-check embedding mean near 0 and covariance near identity).

## 4. Metrics (the heart, define them exactly)

All metrics on the held-out eval split. Center variables before alignment. Report mean and standard deviation across at least 5 training seeds.

### M1. Linear identifiability of the state, and up-to-rotation recovery

Let `Z` be the true latents `(N x n)` and `Zhat = f(x)` the learned ones `(N x K)`, both centered. With `K = n`:
- Unrestricted linear alignment: `W = argmin ||Z - Zhat W||_F` (least squares). Report `R2_lin` = fraction of variance of `Z` explained.
- Orthogonal alignment (Procrustes): `M = Zhat^T Z`; `U, S, Vt = svd(M)`; `Q = U @ Vt` (orthogonal). Report `R2_orth` = fraction of variance of `Z` explained by `Zhat @ Q`. Also report the best scale (it should be near 1 since both are unit-variance isotropic).
- Interpretation: `R2_lin` high means the latents are linearly decodable at all (necessary condition). `R2_orth` close to `R2_lin` and both high means recovery is up to a rotation/reflection (the identifiability signature).

### M2. Action axis recovered up to rotation (the V-JEPA 2 echo, headline metric)

- True action effect: `B` `(n x m)`, known from the world.
- Learned action effect: read it from the trained predictor. With the linear predictor, `Bhat_pred` `(K x m)` is directly available. (If using an MLP predictor, estimate `Bhat_pred` as the Jacobian of `P` w.r.t. `a`, averaged over eval states.)
- Express the learned action effect in true-latent coordinates using the Procrustes map `Q` from M1, then fit the residual linear map `L` such that `Q^T Bhat_pred ~= L @ B`, i.e. `L = (Q^T Bhat_pred) @ pinv(B)` (size `n x n`).
- Headline number: `cond(L) = sigma_max(L) / sigma_min(L)`, the condition number. 
  - `cond(L)` near 1 means `L` is (a scalar times) an orthogonal matrix: the action axis is recovered up to a rotation. This is the hypothesis-confirmed regime, directly comparable to V-JEPA 2's reported value around 1.5.
  - `cond(L)` large means non-rotational distortion: the rotational ambiguity does NOT survive cleanly. Report it as such.
- Also report the principal angles or the cosine alignment between the column spaces of `Bhat_pred` (rotated back) and `B`, as a complementary readout.

### M3. Dynamics recovery

- In aligned coordinates, compare the learned state-transition `Rhat` to the true `rho * I` (up to the same orthogonal change of basis): report `|| Q^T Rhat Q - rho * I ||_F` and the estimated `rho_hat` (e.g. mean eigenvalue magnitude). Report how close `(rho_hat, Bhat_pred aligned)` are to `(rho, B)`.

## 5. Ablations and controls (this is what makes it an investigation, not a single run)

Run each across the 5+ seeds and plot mean with error bars.

1. Nonlinearity of `g`: sweep `nonlinearity_strength` from linear to strongly nonlinear. Plot `R2_orth`, `R2_lin`, and `cond(L)` vs nonlinearity. Tests whether identifiability degrades as the observation gets harder.
2. SIGReg on vs off (lambda = 0): does removing the isotropic-Gaussian constraint break recovery? Theory predicts the Gaussian constraint is essential. This is the key control that isolates SIGReg's role.
3. Symmetric vs action-conditioned: run a no-action variant (B = 0, pure OU, predictor predicts next state from current state only) and compare its state-recovery to the action-conditioned run. This is the direct test of "does the ambiguity survive the passage to action-conditioning."
4. Excitation of actions: vary the action distribution scale / rank (e.g. degenerate vs full-rank actions) to probe the persistent-excitation condition. Plot `cond(L)` vs excitation.
5. Sensitivity: vary `n`, `m`, `rho`, `K` (including `K > n`) in a small grid. Keep it minimal.

## 6. Outputs

- `results/` with a metrics table (CSV) over all conditions and seeds (mean and std).
- Figures (matplotlib, clean, labeled): (a) `R2_orth` vs `R2_lin` across nonlinearity, (b) headline `cond(L)` across conditions and seeds with error bars, (c) dynamics-recovery error, (d) symmetric vs action-conditioned comparison.
- `RESULTS.md`: an honest written summary of what was found, with the central conclusion stated plainly (hypothesis confirmed, partially, or broken), including any negative result.
- A short companion explanation (`NOTES.md` or comments) that explains each design choice in plain language, so a human can defend it: why isotropic Gaussian, what SIGReg does and why it is sliced, why identifiability is only up to rotation (because an isotropic Gaussian is rotation-invariant, so rotations are indistinguishable), what each metric measures.

## 7. Engineering requirements

- Clean, modular repo:
  ```
  src/world.py        # synthetic world generator (frozen g, ground-truth B, rho)
  src/model.py        # encoder, action-conditioned predictor
  src/sigreg.py       # SIGReg
  src/train.py        # training loop
  src/metrics.py      # M1, M2, M3 (Procrustes, condition number, dynamics)
  src/run.py          # one experiment from a config
  configs/*.yaml       # default + ablation configs
  scripts/sweep.py     # run the ablations, collect results
  RESULTS.md, NOTES.md, README.md, requirements.txt, LICENSE (MIT)
  ```
- Config-driven (yaml or argparse), all hyperparameters and all seeds in the config and logged.
- Reproducible: set and log seeds, deterministic where possible, fix the world seed separately from training seeds. A documented command must reproduce every figure.
- Lightweight: this runs in minutes on a single GPU and should also run on CPU. Do NOT add distributed training, mixed precision tricks, or large dependencies. Vectors and small MLPs only.
- README in research-investigation framing (not tutorial framing): the question, the hypothesis, the controlled setup, the headline result with the condition-number metric, what it opens, and a link placeholder to the research statement. Match the polish of a strong open-source research repo.
- A tiny smoke test (a few steps, n=4) that runs end to end in seconds, so correctness can be checked fast.

## 8. Scientific integrity (non-negotiable)

- Do NOT hardcode, mock, or fabricate any metric value. Every number in tables, figures, and `RESULTS.md` must come from an actual run, reproducible from the logged seed.
- If a run fails to converge, or the hypothesis is not confirmed, report that honestly. A clean negative result is a real outcome here and is more valuable than a fabricated positive one.
- Do NOT scale up or attempt to reproduce any published model. The entire value is the controlled toy with ground truth.
- Write the code to be read and defended by a human: comment the why, not just the what, especially in `sigreg.py` and `metrics.py`.
