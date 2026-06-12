# Results

**Verdict (pre-registered rules): CONFIRMED — the orthogonal identifiability ambiguity survives
action-conditioning in this controlled world.**

## Provenance

Full-mode sweep executed 2026-06-12 on a single CUDA GPU (torch 2.9.1+cu128):
19 conditions × 5 training seeds = **95 runs**, 64.5 min wall clock. World seed 1234 (one fixed
world per condition; datasets identical across training seeds), training seeds {0..4}.

**Provenance status, stated exactly.** The run's raw per-run table is committed at
[`results/metrics.csv`](results/metrics.csv) (95 rows, every config field logged,
`quick=False`, `steps=4000` throughout). Every aggregate in this document is recomputable from it,
and the verdict is machine-reproducible in one line from the repo root:

```bash
python -c "import pandas as pd; from src.metrics import verdict, render_verdict; \
print(render_verdict(verdict(pd.read_csv('results/metrics.csv')), quick=False))"
```

This reproduces **CONFIRMED** with the exact numbers below — including the FAILED dynamics check —
plus the SIGReg-off check that the run's notebook had silently skipped (see "What each control
showed"). The four figures in `results/figures/` are regenerated from this CSV by
`scripts/sweep.py:make_figures` (their provenance is the CSV, not the run machine's pixels). Two
caveats remain: the **FULL-executed notebook** of the run is not yet committed (the notebook at
the repo root embeds its **QUICK smoke execution** — banner "NOT science", verdict INCONCLUSIVE by
design at smoke scale — it is the *source* of the experiment, not the run artifact), and the CSV
was transferred from the run machine by the author rather than produced inside this repository's
history.

## Headline (base condition: $n=8$, $m=2$, $\rho=0.9$, nonlinear $g$, $\lambda=0.5$)

5/5 seeds passed the convergence flag; gate $R^2_{\text{lin}} \ge 0.90$ passed (0.943).

| metric | value | pre-registered bands |
|---|---|---|
| gap $= R^2_{\text{lin}} - R^2_{\text{orth}}$ | **0.0009 ± 0.0005** | confirmed ≤ 0.05 · broken > 0.15 |
| cond$_m(L)$ (1 = exact rotation) | **1.018 ± 0.007** | confirmed ≤ 2.0 · broken > 5.0 |
| max principal angle θ | **1.14° ± 0.20°** | confirmed ≤ 15° · broken > 30° |

All three sit deep inside the confirmed band. For comparison, V-JEPA 2's reported value at scale
is ≈ 1.5; this controlled world sits at 1.018. The state is recovered up to rotation
($R^2_{\text{orth}} = 0.942$, within 0.0009 of the best unrestricted linear map), the action axis
is recovered up to the *same* rotation almost exactly, and the embedding isotropy diagnostic sits
at mean covariance error 0.152 (a diagnostic the training markdown predicts, not a registered
criterion).

## Full per-condition table (mean ± std over 5 seeds)

| condition | R²_lin | R²_orth | gap | cond_m | θ_max (°) | ρ̂ | D_rel | conv |
|---|---|---|---|---|---|---|---|---|
| base | 0.943 ±0.054 | 0.942 ±0.054 | 0.0009 | 1.018 ±0.007 | 1.14 | 0.829 ±0.146 | 0.246 | 5/5 |
| nonlin_s=0.00 | 0.955 ±0.060 | 0.948 ±0.068 | 0.0072 | 1.013 | 1.08 | 0.770 ±0.180 | 0.453 | 5/5 |
| nonlin_s=0.25 | 0.971 ±0.055 | 0.971 ±0.055 | 0.0006 | 1.012 | 1.15 | 0.834 ±0.148 | 0.246 | 5/5 |
| nonlin_s=0.50 | 0.943 ±0.067 | 0.941 ±0.069 | 0.0018 | 1.010 | 1.18 | 0.773 ±0.174 | 0.462 | 5/5 |
| nonlin_s=0.75 | 0.958 ±0.055 | 0.957 ±0.056 | 0.0009 | 1.017 | 1.23 | 0.833 ±0.145 | 0.247 | 5/5 |
| sigreg_off (λ=0) | 0.006 ±0.001 | 0.004 ±0.000 | 0.0019 | 1.433 | 51.13 | −0.036 | 1.467 | 0/5 |
| lam=0.05 (paper) | 0.501 ±0.006 | 0.493 ±0.005 | 0.0082 | 1.042 | 2.83 | 0.205 | 1.369 | 5/5 |
| symmetric (B=0) | 0.940 ±0.054 | 0.939 ±0.055 | 0.0005 | — | — | 0.828 ±0.146 | 0.249 | 5/5 |
| excite_scale=0.5 | 0.941 ±0.054 | 0.941 ±0.054 | 0.0006 | 1.035 ±0.049 | 2.26 | 0.828 | 0.247 | 5/5 |
| excite_scale=0.1 | 0.940 ±0.054 | 0.940 ±0.054 | 0.0005 | 1.192 ±0.075 | 11.24 | 0.828 | 0.248 | 5/5 |
| excite_rank1 | 0.941 ±0.056 | 0.941 ±0.057 | 0.0007 | 2.7·10⁶ ±0.8·10⁶ | 2.06 | 0.824 | 0.241 | 5/5 |
| literal_noise | 0.945 ±0.050 | 0.939 ±0.049 | 0.0055 | 1.028 ±0.006 | 1.43 | 0.829 | 0.245 | 5/5 |
| n=4 | 0.934 ±0.108 | 0.920 ±0.140 | 0.0147 | 1.094 | 0.44 | 0.880 ±0.022 | 0.036 | 4/5 |
| n=16 | 0.587 ±0.025 | 0.580 ±0.026 | 0.0068 | 1.042 | 7.96 | 0.375 ±0.059 | 1.247 | 5/5 |
| m=1 | 0.940 ±0.054 | 0.939 ±0.056 | 0.0014 | 1.000 (trivial) | 1.73 | 0.823 | 0.242 | 5/5 |
| m=4 | 0.890 ±0.064 | 0.888 ±0.066 | 0.0019 | 1.129 ±0.093 | 0.48 | 0.723 | 0.579 | 4/5 |
| rho=0.5 | 0.962 ±0.001 | 0.961 ±0.001 | 0.0011 | 1.017 | 0.58 | 0.489 ±0.001 | 0.032 | 5/5 |
| rho=0.99 | 0.737 ±0.049 | 0.714 ±0.066 | 0.0235 | 1.189 ±0.108 | 3.83 | 0.728 ±0.147 | 0.804 | 5/5 |
| K=16 (> n) | 0.951 ±0.005 | 0.889 ±0.023 | 0.0616 ±0.0176 | 1.040 | 1.13 | 0.893 ±0.001 | 0.011 | 5/5 |

(Notation: gap $= R^2_{\text{lin}} - R^2_{\text{orth}}$; $D_{\text{rel}} =
\|Q^\top \hat R Q - \rho I\|_F / (\rho\sqrt{n})$; conv = seeds passing the convergence flag.
cond_m means are over finite values; the `m=1` value is trivially 1 by construction and excluded
from any verdict logic — there the principal angle is the informative readout, θ = 1.73°.)

## What each control showed

**SIGReg off (λ = 0) — the key control, and it bites.** Total collapse: $R^2_{\text{lin}} = 0.006$,
0/5 seeds converged. The Gaussian constraint is load-bearing exactly as the theory predicts, and
the experiment has discriminative power on this axis. *Evaluation note:* the notebook's verdict
cell silently skipped this check because its converged-only filter excluded a condition that never
converges — precisely because it collapses. The repo code evaluates this one check over all runs
(`_mean_all_finite` in `src/metrics.py`), which is the disclosed Phase-2 fix; the underlying data
is unchanged.

**Paper-default λ = 0.05 — the partial-collapse equilibrium, replicated at 5 seeds.**
$R^2_{\text{lin}} = 0.501 \pm 0.006$, embedding covariance error 0.850 (vs 0.152 at λ = 0.5),
$\hat\rho = 0.21$. At toy scale, the soft penalty at the paper's default sits at a stable
equilibrium with dead embedding dimensions (a 9,000-step probe confirmed it is an equilibrium, not
undertraining — see NOTES.md). This is why the experiment runs at λ = 0.5: the theory assumes the
isotropy *constraint*; λ is only its enforcement strength.

**Symmetric vs action-conditioned — the direct test of the question.**
$\Delta R^2_{\text{orth}} = 0.0024$ (0.9417 vs 0.9393): indistinguishable at the seed-noise scale
(the seed std is ≈ 0.054; 5 seeds on a shared world — no formal test is claimed). The passage from
the proven action-free regime to the action-conditioned one costs nothing measurable here.

**Excitation — degrades the action axis, and only the action axis, monotonically.**
cond$_m$: 1.018 (scale 1.0) → 1.035 (0.5) → 1.192 (0.1), with θ rising to 11.2°, while
$R^2_{\text{orth}}$ stays flat at 0.94 across the whole sweep (the balanced world guarantees the
marginal stays isotropic at every excitation level, so this separation is by construction — and it
held). Under **rank-1 actions**, the pre-registered sharp sub-prediction held exactly: the excited
direction stays recovered (cosine 0.998) while the full-map cond$_m$ blows up to ~10⁶ — with
$\hat B$ initialized at zero and no weight decay on the predictor, the unexcited column of $B$
receives no gradient and is unidentifiable *in principle*, not just in practice.

**Literal-spec noise — the measured confound, with a refined reading.** cond$_m$ rises only to
1.028 (predicted analytic whitening factor: cond$(\Sigma_{\text{lit}}^{1/2}) = 1.225$). The
resolution is instructive: $B$ is drawn with equal column norms, so the stationary inflation
$B\Sigma_a B^\top/(1-\rho^2)$ lands nearly equally on both directions of $\mathrm{col}(B)$
(eigenvalues ≈ 1.50 and 1.41) — the whitening therefore acts on the action subspace almost as a
*scalar*, which cond$_m$ is invariant to by design. The artifact shows up where the geometry says
it must: in the M1 gap, which is 6× the balanced world's (0.0055 vs 0.0009, non-overlapping stds).
The full-space prediction 1.225 conflates directions outside $\mathrm{col}(B)$; the
$\mathrm{col}(B)$-restricted prediction (≈ 1.03) matches the measurement. The pre-registered
directional check (literal ≥ balanced) passed.

**Sensitivity — where the honest boundary is.**
- `n=4`, `rho=0.5`: clean (and dynamics nearly exact: D_rel 0.036 / 0.032).
- `n=16`: recovery drops to $R^2 \approx 0.59$, $\hat\rho = 0.37$ at the fixed training budget —
  yet cond$_m$ stays at 1.04: even when state recovery is partial, what *is* recovered of the
  action axis remains rotational.
- `rho=0.99`: long-memory regime is harder ($R^2 \approx 0.74$); expected, as the innovation
  signal per step shrinks by design.
- `K=16 > n`: the only non-trivial gap in the sweep (0.0616) — an overcomplete embedding weakens
  the rotational signature; dynamics leakage stays low and $\hat\rho = 0.893$ is the cleanest in
  the table. Overcompleteness is a genuine soft spot worth theory attention.
- `m=4`: mild cond$_m$ inflation (1.13) and 4/5 convergence — more action directions compete for
  the same innovation budget.

## The check that failed (reported as required)

**"Dynamics recovered" FAILED on the pre-registered criterion:** mean $\hat\rho = 0.829$ vs true
0.9 (tolerance ±0.05) and $D_{\text{rel}} = 0.246$ (tolerance 0.2). The cause is fully localized:
**training seed 2** lands in a reproducible suboptimal basin in most base-dimension conditions
($R^2_{\text{lin}} = 0.846$, $\hat\rho = 0.567$ in the base condition: 4 seeds at
$\hat\rho = 0.894$ and seed 2 at 0.567 — the mean is exactly $0.829$). Within that basin the run *plateaus*, so the
plateau-based convergence flag does not catch it; the multi-seed protocol does (the large stds in
the table are this single seed). Honest readings, in order: (1) the criterion as registered fails
and we report it; (2) the failure is an optimization-landscape phenomenon, not a refutation of the
identifiability picture — the surviving 4/5 seeds give $\hat\rho = 0.894$, $D_{\text{rel}} \approx
0.01$, and seed 2's *action axis is still rotational* (cond$_m$ 1.03); (3) the criterion design has
a lesson: a mean over seeds conflates "biased recovery" with "bimodal optimization"; a
pre-registered per-seed or median criterion would separate them. We did not change the rule after
seeing the data — the per-seed breakdown is provided so the reader can apply either reading.

## Conclusion

In a controlled world satisfying the Gaussian assumptions, with the isotropy constraint actually
enforced, action-conditioned LeJEPA training recovers the state up to a rotation and the action
effect up to the same rotation (cond$_m$ = 1.018; on the deviation-from-perfect-rotation scale,
cond$_m - 1$ = 0.018 vs ≈ 0.5 for V-JEPA 2's reported 1.5 — a scale anchor, not a baseline). The
pre-registered dynamics check **failed** on its mean-over-seeds criterion ($\hat\rho = 0.829$ vs
$0.9 \pm 0.05$), driven by the single reproducible optimization basin discussed above; the other
4/5 seeds recover the dynamics in closed form ($\hat\rho = 0.894$, $D_{\text{rel}} \approx 0.01$).
The orthogonal ambiguity of the identifiability guarantee **survives the passage to
action-conditioning** here. The boundary of
the claim is equally clear: enforcement strength matters (λ = 0.05 fails the precondition),
excitation is a genuine identifiability condition (rank-deficient actions leave provably
unidentifiable directions), overcomplete embeddings weaken the rotational signature, and a
reproducible optimization basin can defeat mean-based criteria. These are the sharp edges the
action-conditioned theory will need to handle.
