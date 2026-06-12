# Does the orthogonal identifiability ambiguity survive action-conditioning?

A minimal, controlled, ground-truth experiment on LeJEPA-style latent recovery in an
**action-conditioned** world. Toy scale, controlled methodology: every quantity the model is
supposed to recover is known exactly; every run is seeded and logged (bitwise reproducible on CPU,
across-seed reproducible on GPU — see *Reproduce*); and the verdict is rendered by decision rules
fixed in code (`verdict()` in [src/metrics.py](src/metrics.py)), not written by hand. The
experiment was specified before implementation in [docs/SPEC.md](docs/SPEC.md); every deviation
from that spec is argued and measured, never silent ([NOTES.md](NOTES.md)).

**Headline result: yes — it survives.** Trained with SIGReg on an action-conditioned
linear-Gaussian world observed through a frozen nonlinear map, the model recovers the true action
effect up to a rotation — precisely: a scaled orthogonal map, rotation or reflection; we write
"rotation" for short — with condition number **1.018 ± 0.007** (1 = exact; V-JEPA 2's reported
value at scale is ≈ 1.5), while the gap between unrestricted-linear and orthogonal-Procrustes
state recovery is **0.0009 ± 0.0005** — essentially everything that is linearly decodable is
decodable by a rotation alone. Verdict under the pre-registered rules: **CONFIRMED** — with one
pre-registered secondary check (dynamics, on the seed mean) **failing** and reported as such; see
[RESULTS.md](RESULTS.md).

---

## The question

LeJEPA (Balestriero & LeCun, 2025) trains joint-embedding models without heuristics — no EMA
teacher, no stop-gradient — by pushing embeddings toward an isotropic Gaussian via **SIGReg**. A
2026 result by Klindt, LeCun & Balestriero proves that in a Gaussian world with stationary
additive-noise (Ornstein–Uhlenbeck) transitions and a nonlinear observation map, this training
recovers the true latents **up to an orthogonal transformation**, and the dynamics in closed form.
The ambiguity is intrinsic: $\mathcal{N}(0, I)$ is rotation-invariant, so no objective of this
form can see a rotation of the encoder.

That guarantee covers the **symmetric (action-free) case only**. Separately, V-JEPA 2
(Assran et al., 2025) — an action-conditioned world model at scale — was observed empirically to
recover its action axis up to a near-rotation (condition number ≈ 1.5). Nobody has shown whether
the orthogonal ambiguity *survives the passage to the action-conditioned regime*. Both outcomes
are a result: clean survival extends the identifiability picture; distortion beyond a rotation
sharpens the open problem.

## The controlled setup

- **World** (frozen, seeded): $z_{t+1} = \rho z_t + B a_t + \eta_t$ with $z \in \mathbb{R}^n$
  ($n = 8$), actions $a \in \mathbb{R}^m$ ($m = 2$), $a \sim \mathcal{N}(0, \Sigma_a)$ with
  $\Sigma_a = I_2$, $\rho = 0.9$, and ground-truth action effect $B$. The transition
  noise is *balanced*, $\eta \sim \mathcal{N}(0, (1-\rho^2)I - B\Sigma_a B^\top)$, the unique
  Gaussian noise making $\mathcal{N}(0, I)$ exactly stationary — so the only change from the
  proven symmetric setting is the action term itself (the spec's literal noise is kept as a
  measured ablation; see [NOTES.md](NOTES.md)). Observations $x = g(z) \in \mathbb{R}^{64}$
  through a frozen random MLP with a tunable nonlinearity knob.
- **Model**: small MLP encoder $f: \mathbb{R}^{64} \to \mathbb{R}^K$ (embedding dimension
  $K = n = 8$ by default) + deliberately **linear**
  action-conditioned predictor $P(\hat z, a) = \hat R \hat z + \hat B a + c$, so the learned
  dynamics $(\hat R, \hat B)$ are read off directly. LeJEPA recipe: no teacher–student, no
  stop-gradient; collapse prevention by SIGReg alone.
- **Metrics** (held-out split, 5 training seeds per condition, fixed world seed):
  - **M1** — state up to rotation: unrestricted-linear $R^2$ vs orthogonal-Procrustes $R^2$;
    their gap isolates exactly the non-rotational distortion.
  - **M2 (headline)** — action axis up to rotation: condition number of the residual map
    $L = (Q^\top \hat B)\, B^{+}$ over its top-$m$ singular values ($Q$ = the M1 Procrustes
    alignment, $B^{+}$ = pseudoinverse of the true effect), plus principal angles.
  - **M3** — dynamics: $\hat\rho = \mathrm{tr}(Q^\top \hat R Q)/n$ and the normalized error
    $D_{\text{rel}} = \|Q^\top \hat R Q - \rho I\|_F / (\rho\sqrt{n})$.
- **Controls**: SIGReg off; paper-default $\lambda$; symmetric ($B=0$) vs action-conditioned;
  action excitation (scale and rank); literal-spec noise; dimension/$\rho$ sensitivity.

## Results

Full sweep: 19 conditions × 5 seeds = 95 runs, single GPU (CUDA, torch 2.9.1+cu128), 64.5 min,
world seed 1234, training seeds 0–4. Complete tables and discussion — including the one
pre-registered check that **failed** and why — in **[RESULTS.md](RESULTS.md)**. *Provenance:* the
run's raw per-run table is committed at [results/metrics.csv](results/metrics.csv); the verdict
and every aggregate are machine-reproducible from it (one-liner in RESULTS.md), and the figures in
[results/figures/](results/figures) are regenerated from it. Still pending: the FULL-executed
notebook of the run — see [results/README.md](results/README.md).

| condition | $R^2_{\text{lin}}$ | $R^2_{\text{orth}}$ | gap | cond$_m$ | $\theta_{\max}$ (°) |
|---|---|---|---|---|---|
| **base (action-conditioned)** | **0.943 ± 0.054** | **0.942 ± 0.054** | **0.0009 ± 0.0005** | **1.018 ± 0.007** | **1.14 ± 0.20** |
| symmetric ($B = 0$, proven regime) | 0.940 ± 0.054 | 0.939 ± 0.055 | 0.0005 ± 0.0001 | — | — |
| SIGReg off ($\lambda = 0$) | 0.006 ± 0.001 | 0.004 ± 0.000 | — collapse — | — | — |
| paper-default $\lambda = 0.05$ | 0.501 ± 0.006 | 0.493 ± 0.005 | 0.008 | 1.042 | 2.83 |
| excitation scale 0.1 | 0.940 ± 0.054 | 0.940 ± 0.054 | 0.0005 | 1.192 ± 0.075 | 11.24 |
| rank-1 actions (unexcited dir.) | 0.941 ± 0.056 | 0.941 ± 0.057 | 0.0007 | ~10⁶ (degenerate) | 2.06 |
| literal-spec noise | 0.945 ± 0.050 | 0.939 ± 0.049 | 0.0055 ± 0.0001 | 1.028 ± 0.006 | 1.43 |

Three controls carry the story: removing SIGReg collapses everything (the Gaussian constraint is
load-bearing, as the theory predicts); the symmetric and action-conditioned worlds recover the
state equally well ($\Delta R^2_{\text{orth}} = 0.0024$, against a seed std of ≈ 0.054 — the
ambiguity survives the passage); and
weakening the action excitation degrades *only* the action-axis metric, down to a provably
unidentifiable direction under rank-deficient actions while state recovery stays intact.

## Reproduce

```bash
pip install -r requirements.txt

python tests/smoke_test.py          # seconds: SIGReg sanity + metrics self-test + tiny pipeline
python scripts/sweep.py --quick     # ~1 min:  full code path on toy dimensions (NOT science)
python scripts/sweep.py             # the real experiment: 95 runs, ~1 h on a single GPU
python -m src.run --lam 0 --label sigreg_off   # any single condition, config-driven via argparse
```

`scripts/sweep.py` writes `results/metrics.csv` (one row per condition × seed, every config field
logged), regenerates the four figures in `results/figures/`, and prints the verdict computed by
the pre-registered rules in [src/metrics.py](src/metrics.py).

Reproducibility, stated exactly: bitwise on CPU
(`torch.use_deterministic_algorithms(True, warn_only=True)`); on CUDA/MPS some kernels have no
deterministic variant, so individual runs reproduce statistically rather than bitwise — every
conclusion rests on mean ± std across 5 training seeds, never on a single run.
`requirements.lock.txt` pins the local (macOS) environment the code was verified in; the analyzed
run used torch 2.9.1+cu128.

The Phase-1 notebook `lejepa_action_identifiability.ipynb` is the same code with full narration
(every cell pair explains the *why*); `src/` was extracted from it mechanically
(`tools/extract_modules.py`).

## Repository layout

```
src/config.py      # Config dataclass, seeding, device, paths
src/world.py       # synthetic world (frozen g, ground-truth B, rho, balanced noise)
src/model.py       # encoder, action-conditioned linear predictor
src/sigreg.py      # SIGReg (sliced Epps-Pulley CF matching, real arithmetic)
src/train.py       # LeJEPA training loop (no EMA, no stop-gradient)
src/metrics.py     # M1/M2/M3, Procrustes, condition number + pre-registered verdict
src/run.py         # one experiment from a config
scripts/sweep.py   # the ablation grid, results CSV, figures, verdict
tests/smoke_test.py
lejepa_action_identifiability.ipynb   # Phase-1 narrated notebook (QUICK toggle; the committed
                                      # copy embeds QUICK smoke outputs, NOT the full run)
docs/SPEC.md       # the experiment specification this repo implements (+ NOTEBOOK_ADDENDUM.md)
RESULTS.md         # full results & interpretation            NOTES.md  # design rationale
```

## Limitations, and what this opens

Toy scale by design: linear-Gaussian dynamics, $n = 8$, a frozen smooth observation map. This is
an empirical result in a world satisfying the theory's assumptions — not a theorem, and not
evidence about web-scale models (the V-JEPA 2 number anchors the metric's scale, nothing more).
The sensitivity cells mark the honest boundary: recovery degrades at fixed training budget for
$n = 16$ and $\rho = 0.99$, and an overcomplete embedding ($K = 16 > n$) opens the only
non-trivial linear-vs-orthogonal gap (0.062) in the sweep. What this opens: the action-conditioned
identifiability *proof* (this repo is its testbed), richer worlds (nonlinear dynamics, partial
observability), and whether the condition-number diagnostic predicts planning performance at
scale. Research statement: link to be added.

## References

1. R. Balestriero, Y. LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the
   Heuristics.* arXiv:2511.08544, 2025.
2. D. Klindt, Y. LeCun, R. Balestriero (2026). Identifiability of LeJEPA-style training in
   Gaussian latent worlds, symmetric case. *(Result as described in [docs/SPEC.md](docs/SPEC.md);
   full bibliographic record pending publication.)*
3. M. Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and
   Planning.* arXiv:2506.09985, 2025. (The "condition number ≈ 1.5" anchor used throughout is the
   observation as described in [docs/SPEC.md](docs/SPEC.md) — a scale anchor for the metric, not a
   re-measured baseline.)
4. T. W. Epps, L. B. Pulley. *A test for normality based on the empirical characteristic
   function.* Biometrika 70(3), 1983.
5. P. H. Schönemann. *A generalized solution of the orthogonal Procrustes problem.* Psychometrika
   31(1), 1966.

MIT License — © 2026 Sid Ahmed Bouamama.
