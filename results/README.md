# results/

**Status: the full-run artifacts are NOT yet committed.** This directory is the designated home
for the artifacts of the full-mode experiment analyzed in [RESULTS.md](../RESULTS.md) (run of
2026-06-12: single CUDA GPU, torch 2.9.1+cu128, 95 runs, 64.5 min, world seed 1234, training
seeds 0–4):

- `metrics.csv` — one row per condition × seed, every config field logged. **Pending** transfer
  from the run machine; until it lands, every number in RESULTS.md/README.md is a faithful
  transcription of that run's printed output, not independently auditable from this repo.
- `figures/fig1..4_*.png` — the four spec figures of that run. **Pending**, same provenance.
- the FULL-executed notebook of that run — the end-to-end provenance artifact. **Pending.**
  (The notebook committed at the repo root embeds outputs of its **QUICK smoke execution** only —
  its banner reads "NOT science" and its verdict prints INCONCLUSIVE by design at smoke scale.)

Anything regenerated locally with `python scripts/sweep.py --quick` is a smoke test and must not
be committed here — quick-mode rows carry `quick=True` in the CSV precisely so they can never be
mistaken for results.
