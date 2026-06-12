# results/

Artifacts of the **full-mode** experiment (the run analyzed in [RESULTS.md](../RESULTS.md):
2026-06-12, single CUDA GPU, 95 runs, 64.5 min, world seed 1234, training seeds 0–4):

- `metrics.csv` — one row per condition × seed, every config field logged. *(Pending commit of the
  raw file from the run machine; the aggregate table in RESULTS.md is transcribed from this run's
  output.)*
- `figures/fig1..4_*.png` — the four spec figures. *(Same provenance.)*
- The executed notebook of that run is the end-to-end provenance artifact.

Anything regenerated locally with `python scripts/sweep.py --quick` is a smoke test and must not
be committed here — quick-mode rows carry `quick=True` in the CSV precisely so they can never be
mistaken for results.
