# results/

Artifacts of the experiments analyzed in [RESULTS.md](../RESULTS.md). All runs: single CUDA GPU,
world seed(s) and training seeds logged per row, `quick=False` throughout.

- `metrics.csv` — the main sweep (95 rows = 19 conditions × 5 seeds, every config field logged;
  2026-06-12, torch 2.9.1+cu128, 64.5 min). Verified on commit: its aggregates match the run's
  printed output exactly, and `src/metrics.py:verdict()` applied to it reproduces the run's
  verdict (CONFIRMED, including the FAILED dynamics check) — the reproduction one-liner is in
  RESULTS.md. Transferred from the run machine by the author.
- `metrics_ext.csv` — the pre-registered extended sweep (115 rows = 23 conditions × 5 seeds:
  4 extra worlds, λ transition, 2× budget, K sweep, extra base seeds), run after the registration
  commit of `scripts/sweep_extended.py` (2026-06-12T03:10+02:00). Verified on commit: aggregates
  match the run's printed log; the pre-registered report is machine-reproducible from the CSVs.
- `figures/fig1..4_*.png` — regenerated from `metrics.csv` by `scripts/sweep.py:make_figures`.
- `figures/fig5..8_*.png` — regenerated from `metrics_ext.csv` + `metrics.csv` by
  `scripts/sweep_extended.py:figures`.

Anything regenerated locally with `--quick`/`--smoke` is a smoke test and must not be committed
here — quick-mode rows carry `quick=True` in the CSV precisely so they can never be mistaken for
results.
