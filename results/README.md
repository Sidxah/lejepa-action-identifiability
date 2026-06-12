# results/

Artifacts of the full-mode experiment analyzed in [RESULTS.md](../RESULTS.md) (run of 2026-06-12:
single CUDA GPU, torch 2.9.1+cu128, 95 runs, 64.5 min, world seed 1234, training seeds 0–4):

- `metrics.csv` — **committed**: the run's raw per-run table (95 rows = 19 conditions × 5 seeds,
  every config field logged, `quick=False` throughout). Verified on commit: its aggregates match
  the run's printed output exactly, and `src/metrics.py:verdict()` applied to it reproduces the
  run's verdict (CONFIRMED, including the FAILED dynamics check) — the reproduction one-liner is
  in RESULTS.md. Transferred from the run machine by the author.
- `figures/fig1..4_*.png` — **committed**: regenerated from `metrics.csv` by
  `scripts/sweep.py:make_figures`; their provenance is the CSV.
- `metrics_ext.csv` — **committed**: the pre-registered extended sweep (115 rows = 23 conditions
  × 5 seeds: 4 extra worlds, λ transition, 2× budget, K sweep, extra base seeds), run on a single
  CUDA GPU after the registration commit of `scripts/sweep_extended.py` (2026-06-12T03:10+02:00).
  Verified on commit: aggregates match the run's printed log; the pre-registered report is
  machine-reproducible from the CSVs.
- `figures/fig5..8_*.png` — **committed**: regenerated from `metrics_ext.csv` + `metrics.csv` by
  `scripts/sweep_extended.py:figures`.
- the FULL-executed notebook of the run — **still pending** transfer from the run machine.
  (The notebook committed at the repo root embeds outputs of its **QUICK smoke execution** only —
  its banner reads "NOT science" and its verdict prints INCONCLUSIVE by design at smoke scale.)

Anything regenerated locally with `python scripts/sweep.py --quick` is a smoke test and must not
be committed here — quick-mode rows carry `quick=True` in the CSV precisely so they can never be
mistaken for results.
