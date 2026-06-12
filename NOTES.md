# Notes: every design choice, and why

Companion to the code (the explanation document required by the spec, §6). **The spec itself —
the document every "the spec says" below refers to — is committed at [docs/SPEC.md](docs/SPEC.md)**
(packaging addendum: [docs/NOTEBOOK_ADDENDUM.md](docs/NOTEBOOK_ADDENDUM.md)), so each claimed
contradiction or deviation can be checked against its text rather than taken on faith. Each entry
is written to be defended orally; the notebook markdown carries the same arguments inline.

## The target distribution

**Why an isotropic Gaussian.** Two independent reasons. Practical (LeJEPA's): $\mathcal{N}(0, I_K)$
minimizes worst-case downstream prediction risk — the safest geometry for unknown downstream
tasks. Structural (this experiment's): the isotropic Gaussian is rotation-invariant, which is
*the* source of the up-to-rotation ambiguity being probed. If the encoder $f$ attains the optimum,
so does $O \circ f$ for any orthogonal $O$ with the predictor conjugated — the objective cannot
distinguish them. The identifiability theorem's content is that nothing *more* than this rotation
is lost.

**Why SIGReg is sliced.** Matching a $K$-dimensional law directly from finite batches is hopeless
(curse of dimensionality). By Cramér–Wold, matching every 1-D projection suffices, and every unit
projection of $\mathcal{N}(0, I_K)$ is exactly $\mathcal{N}(0,1)$. Per slice, the Epps–Pulley
statistic compares empirical and target characteristic functions with weight $e^{-t^2/2}$ —
smooth, bounded, bounded gradients (unlike CDF/moment tests), linear cost in $K$. Directions are
resampled every step (seeded by the global step, as in the reference), so the expected loss covers
the sphere; bare-step seeding gives common random numbers across runs, tightening comparisons.

**Why real cos/sin arithmetic.** The reference uses `exp(1j·…)`; complex tensors are unsupported
on some accelerators (Apple MPS). Since the target CF is real,
$|\hat\varphi(t) - \varphi(t)|^2 = (\overline{\cos t p} - \varphi(t))^2 + (\overline{\sin t p})^2$
identically — Euler plus linearity of the mean, zero approximation. `torch.trapezoid` replaces the
deprecated `torch.trapz`; the per-slice integral × batch-size $N$ matches the reference (it
calibrates the null value to $O(1)$, cancelling the $1/N$ ECF variance).

## The world

**The spec's contradiction and the balanced noise.** The spec asks for marginal
$z \sim \mathcal{N}(0, I)$ *and* noise $\sqrt{1-\rho^2}\,\eta$; with $B \neq 0$ both cannot hold
(Lyapunov: $\Sigma_{\text{lit}} = I + B\Sigma_aB^\top/(1-\rho^2)$). Accepting the anisotropic
marginal would force even a perfect learner to whiten — injecting a condition-number artifact
exactly where the headline metric looks, for reasons unrelated to action-conditioning. We resolve
in favor of the marginal (the assumption that is load-bearing for the theory):
$\Lambda = (1-\rho^2)I - B\Sigma_aB^\top$, the unique noise making $\mathcal{N}(0,I)$ exactly
stationary. Consequences: exact stationarity from $t=0$ (no burn-in even at $\rho = 0.99$); exact
reduction to the spec's OU world at $B=0$ (apples-to-apples symmetric comparison); feasibility
condition $\lambda_{\max}(B\Sigma_aB^\top) \le 1-\rho^2$ asserted with margin. Because it deviates
from the spec's literal text, the literal transition is kept as a measured ablation — and the
measurement refined the prediction (see RESULTS.md: the artifact acts almost as a scalar on
col($B$) because $B$'s columns have equal norms, so it shows in the M1 gap, not in cond$_m$).

**Drawing $B$ ("columns of moderate norm").** $B = b\,U$ with unit-norm Gaussian direction columns
and one scalar $b$ calibrated so the action signal fraction
$\kappa = \lambda_{\max}(B\Sigma_aB^\top)/(1-\rho^2) = 0.5$ at base settings. Dimensionless and
$\rho$-portable: the $\rho$-sensitivity cells keep the same action-vs-noise balance instead of
silently doubling as excitation cells.

**Two-stage standardization of $g$.** $g_s = \mathrm{std}[(1-s)\tilde g_{\text{lin}} +
s\tilde g_{\text{mlp}}]$ — each endpoint standardized per output dimension on a world-seeded
calibration sample, then the mixture re-standardized (the endpoints are correlated: a random MLP
has a nonzero linear part, so the mixture variance depends on $s$). Every $s$ then yields mean-0 /
std-1 observations: the nonlinearity sweep varies only the nonlinearity, never the input scale.
$g_{\text{lin}}$ is a random semi-orthogonal embedding (condition number 1 — the cleanest
"identity-like" linear endpoint).

## The model and training

**Why the predictor exists, and why it is linear.** Symmetric LeJEPA drops the predictor; an
action makes the views asymmetric — the predictor is where the intervention enters. It is linear
*on purpose*: the true dynamics are linear in $z$, and a linear predictor makes $(\hat R, \hat B)$
directly readable — they are the objects M2/M3 evaluate. A small bias $c$ absorbs residual mean
offsets so they cannot contaminate $\hat R, \hat B$ ($\|c\|$ logged, ≈ 0.002 in practice).

**$\hat B$ initialized at zero, predictor excluded from weight decay.** Two halves of one
integrity argument. Weight decay on $\hat R$ would shrink it toward 0 and bias $\hat\rho$ low — a
metric contamination dressed as regularization (decay applies to encoder weight matrices only).
And with zero-init + no decay, any action direction that receives no gradient (the unexcited
column under rank-1 actions) stays *exactly* at zero — the honest signature of in-principle
unidentifiability, not a decayed-toward-zero artifact. The rank-1 ablation's cond$_m \sim 10^6$ is
this design working.

**No stop-gradient anywhere.** Gradients flow through prediction and target both — the LeJEPA
recipe. The SIGReg-off control shows what holds it up: $\lambda = 0$ collapses to
$R^2 = 0.006$ in 5/5 seeds.

**$\lambda = 0.5$, not the paper's 0.05 — a measured calibration.** The theory assumes the
isotropy *constraint holds*; $\lambda$ is enforcement strength, and ImageNet-scale defaults do not
transfer to toy scale. The trade is lopsided: shrinking one embedding dimension's variance by
$\delta$ saves $\approx (1-\lambda)\,\mathrm{tr}(\Lambda)/n \cdot \delta$ of prediction loss
(linear in $\delta$; the innovation noise is large relative to unit variance,
$\mathrm{tr}(\Lambda)/n \approx 0.17$), while SIGReg's penalty is locally quadratic,
$O(\lambda\delta^2)$. Equilibrium at $\lambda = 0.05$: heavy shrinkage. Measured: a 9,000-step
probe stays pinned at $R^2_{\text{lin}} = 0.49$ with a dead embedding dimension (an equilibrium,
not undertraining); the 5-seed `lam=0.05(paper)` ablation replicates it ($R^2 = 0.501 \pm 0.006$,
covariance error 0.850). At $\lambda = 0.5$ the same argument predicts ≈ 4% residual shrinkage;
measured covariance error: 0.152.

**Schedule and budget.** AdamW lr 2e-3, 100-step warmup, cosine to lr/100, batch 512, 4000 steps
(probe-calibrated: $R^2_{\text{lin}}$ crosses 0.95 near step 2100; the last quarter is plateau
margin). Convergence flag: EMA-smoothed $L_{\text{pred}}$ relative change < 2% over the final 20%
of steps. *Known limitation, learned the honest way:* the flag detects plateaus, not global
optima — training seed 2's reproducible suboptimal basin passes it (see RESULTS.md). Non-converged
runs are reported in the table and excluded from the headline verdict aggregation — with one
disclosed exception: the SIGReg-off control check is evaluated over *all* runs (see
Pre-registration below), because a collapse control that never converges would otherwise exclude
itself from the very check designed to detect collapse.

## The metrics

**Scaled Procrustes for $R^2_{\text{orth}}$.** The hypothesis is "up to rotation"; a global scalar
miscalibration (encoder variance 0.97 instead of 1.0) is theory-irrelevant and would masquerade as
non-rotational distortion. The optimal scale $c^* = \sum_i \sigma_i(M)/\|\hat Z\|_F^2$ is included
(and reported, ≈ 1.00); the unscaled value is kept in the CSV. The gap
$R^2_{\text{lin}} - R^2_{\text{orth}} \ge 0$ then isolates exactly the shear/stretch component.

**The cond$(L)$ fix — a spec bug, not a preference.** The spec's
$\mathrm{cond}(L) = \sigma_{\max}/\sigma_{\min}$ over all $n$ singular values of
$L = (Q^\top\hat B)B^+$ is $+\infty$ by construction whenever $m < n$ (rank$(L) \le m$), for every
run regardless of quality. Headline: $\mathrm{cond}_m = \sigma_1/\sigma_m$ over the top-$m$ —
identical to the spec when $m \ge n$, equals 1 for an exact scaled rotation independently of
cond$(B)$, invariant to action-space reparameterization, same scale as V-JEPA 2's number.
Secondary: the $m \times m$ map $B^+(Q^\top\hat B)$, principal angles (Björck–Golub via QR+SVD,
implemented manually to stay dependency-light), norm ratio. For $m = 1$, cond$_m \equiv 1$
trivially — flagged, excluded from verdict logic, the principal angle carries the reading.

**$\hat\rho = \mathrm{tr}(A)/n$, not mean $|\text{eig}|$.** The trace estimator is the
least-squares projection of $A = Q^\top\hat R Q$ onto $\{rI\}$, making
$\|A-\rho I\|_F^2 = \|A-\hat\rho I\|_F^2 + n(\hat\rho-\rho)^2$ exact; the eigenvalue-modulus mean
is upward-biased (modulus is convex) and kept only as a diagnostic. For $K > n$, the leakage
$\|(I-QQ^\top)\hat R Q\|_F/\|\hat R Q\|_F$ reports dynamics escaping the recovered subspace.

**Pre-registration — and exactly how far it is verifiable.** Thresholds (gate
$R^2_{\text{lin}} \ge 0.90$ & ≥ 80% converged; confirmed: gap ≤ 0.05 ∧ cond$_m$ ≤ 2 ∧ θ ≤ 15°;
broken: gap > 0.15 ∨ cond$_m$ > 5 ∨ θ > 30°) were fixed in the code before the full sweep was
executed; the conclusion cell renders `verdict(df)` — no hand-typed numbers. The 2.0 bound is
anchored to V-JEPA 2's ≈ 1.5: a finite-sample estimate of a true rotation does not sit at exactly
1. **Verifiability caveat, stated plainly:** this repository's git history begins *after* the full
run (one initial commit holding code, rules and results together), so the temporal precedence of
the rules cannot be proven from the repo — it is attested, and supported only by circumstantial
evidence (the thresholds are loose relative to the measured values; a registered check that failed
is reported as failed rather than re-thresholded). A tamper-evident record (rules committed before
data) exists only from this history forward; any future rerun inherits a verifiable chain. One implementation fix was made post hoc and is disclosed: the SIGReg-off check
is evaluated over all runs rather than converged-only, because the converged-only filter excluded
the control precisely when it collapses — the very outcome the check exists to detect
(`_mean_all_finite` in `src/metrics.py`; the notebook version that produced the run retains the
original behavior — its FULL-executed copy is a pending provenance artifact, see
`results/README.md` — and RESULTS.md reports the difference).

## Engineering

**Seed architecture.** `world_seed` (1234) fixes $B$, the observation map, its calibration, and
the data substreams (`+7001/+7002`); the world draws only from its own `torch.Generator`, so
training-time `set_seed` calls cannot perturb it structurally. Training seeds {0..4} control init
and shuffling only — seed spread measures training stochasticity, not data resampling.

**Config-driven via argparse, not YAML.** The spec allows either ("yaml or argparse"); argparse
keeps the dependency list at exactly torch/numpy/pandas/matplotlib and makes every run
self-documenting in shell history. All hyperparameters live in one frozen `Config` dataclass;
ablations are `dataclasses.replace` copies, and every field is logged into each CSV row.

**Phase-1 ↔ Phase-2 sync.** `src/` and `scripts/sweep.py` are extracted mechanically from the
notebook's module-tagged cells by `tools/extract_modules.py` (verbatim bodies, curated import
headers). To change shared logic: edit `tools/build_notebook.py`, rebuild the notebook, re-extract.
