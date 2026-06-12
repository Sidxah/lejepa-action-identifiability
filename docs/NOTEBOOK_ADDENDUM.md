# ADDENDUM to the SPEC: deliver the experiment as a runnable, jury-ready notebook

Read this together with `lejepa_action_identifiability_SPEC.md`. The SPEC defines the science (world, model, SIGReg, metrics, ablations, integrity). This addendum fixes the DELIVERABLE FORMAT. Where they overlap, the science is unchanged; only the packaging changes.

## Phasing

- Phase 1 (now): the primary deliverable is a single, self-contained Jupyter notebook `lejepa_action_identifiability.ipynb` that runs top to bottom on a GPU platform (the owner runs it on a V100 32GB) and also runs on CPU in quick mode. This is what the jury may read, so the markdown narration must carry it.
- Phase 2 (after results exist): extract the Python into the modular `src/` layout from the SPEC for the professional repo. To make this trivial, write all notebook code as importable functions and classes with clean module boundaries (`world`, `model`, `sigreg`, `metrics`, `train`), not as loose script cells. The refactor must then be mechanical copy-out, no rewriting.

## The notebook must read as a research investigation, not a code dump

Markdown is the point of Phase 1. Every code cell is preceded by a markdown cell that explains what it does and, crucially, WHY. The explanations are also the owner's oral preparation, so they must be correct and self-owned, not hand-waved. Use the structure below, one section per pair of markdown plus code cells.

### Cell-by-cell structure

1. Title and abstract (markdown). Title, author "Sid Ahmed Bouamama", and a short abstract: the question (does the orthogonal identifiability ambiguity survive action-conditioning), the hypothesis, and the statement that both outcomes are a result. Keep it tight and serious.

2. Background and hypothesis (markdown). The minimal science a reader needs: LeJEPA pushes embeddings to an isotropic Gaussian via SIGReg, no teacher-student, no stop-gradient. Klindt, LeCun and Balestriero (2026) prove recovery of the latents up to an orthogonal transformation in a Gaussian world with stationary additive-noise transitions, for the state and the symmetric case only. V-JEPA 2 was observed to recover the action axis up to a near-rotation (condition number around 1.5). The open question: does the orthogonal ambiguity survive the action-conditioned regime. State the falsifiable hypothesis and the two outcomes explicitly.

3. Setup (markdown plus code). Markdown: reproducibility approach, the separation between the world seed (fixed once) and training seeds (varied across runs), and a `QUICK` toggle for a fast smoke run versus the full run. Code: imports, install cell if needed, device selection, seed utilities, the `QUICK` flag.

4. The synthetic world (markdown plus code). Markdown: why this world, Gaussian latents matching the optimality and identifiability assumptions, the Ornstein-Uhlenbeck plus action transition, the fixed nonlinear observation map, and the ground-truth action effect B that the metrics will compare against. Code: the world generator as a class or set of functions, returning held-out tuples and the saved ground truth (B, rho, seed, true latents).

5. The model (markdown plus code). Markdown: the encoder and the action-conditioned predictor, why the predictor is kept here (the action makes the two views asymmetric, which is exactly when a predictor is justified), and why we follow the heuristic-free LeJEPA recipe (SIGReg prevents collapse, so no EMA target, no stop-gradient). Code: encoder MLP, linear action-conditioned predictor.

6. SIGReg (markdown plus code). This is the key explanatory cell. Markdown: why the isotropic Gaussian is the target (LeJEPA's downstream-risk optimality), what SIGReg does (it matches each one-dimensional projection of the embeddings to a standard normal characteristic function, which forces the full distribution to be isotropic Gaussian because every projection of an isotropic Gaussian is N(0,1)), and why the sliced construction is used (linear cost, bounded gradients, beats the curse of dimensionality). Code: the SIGReg implementation from the SPEC, with comments.

7. Training (markdown plus code). Markdown: the loss, (1 minus lambda) times the prediction term plus lambda times SIGReg, and the schedule. Code: the training loop, loss logging, an inline plot of the training curves, and a sanity check that the embeddings become isotropic (mean near zero, covariance near identity).

8. Metrics (markdown plus code). Markdown: define each metric precisely and say what it tests. M1, identifiability up to rotation via orthogonal Procrustes, with the explanation that recovery is only up to a rotation because an isotropic Gaussian is rotation-invariant, so rotations are indistinguishable. M2, the action axis recovered up to rotation, with the condition number of the alignment map as the headline number and the explicit comparison to V-JEPA 2's value near 1.5. M3, dynamics recovery. Code: the three metrics on the held-out split.

9. Ablations (markdown plus code). Markdown: the controls and what each isolates, the nonlinearity sweep, SIGReg on versus off, symmetric versus action-conditioned, and action excitation. Code: a small grid over at least five seeds, results collected into a table, plotted with error bars.

10. Results and interpretation (markdown, filled from the real run). A plain-language summary of what was found, stating the central conclusion honestly, hypothesis confirmed, partial, or broken, including any negative result. Write this cell as a clear framework keyed to the metrics; the actual numbers and the final verdict come from the real run and must reflect it, never a fabricated positive.

11. Limitations and next steps (markdown). Honest limitations (toy world, population-level reasoning, the action-conditioned extension is the open problem) and a pointer to the three axes of the research statement.

12. References (markdown).

## Runnability requirements

- Self-contained: the world is synthetic and generated inside the notebook, so there are no dataset downloads and it runs anywhere. No external data files.
- Runs top to bottom with no hidden state and no manual steps. An install cell at the top if any dependency is missing, otherwise standard PyTorch, NumPy, Matplotlib.
- A `QUICK` toggle near the top: when true, tiny dimensions and few steps so the whole notebook finishes in seconds for a correctness check; when false, the full settings from the SPEC for the real results.
- All randomness seeded and logged. Plots render inline. A final results table is displayed in the notebook.
- Lightweight only: vectors and small MLPs, no distributed training, no heavy dependencies. The V100 is far beyond what is needed, so do not scale up to use it.

## Integrity (unchanged from the SPEC)

No hardcoded or fabricated metric values. Every number in tables, figures, and the interpretation cell comes from an actual seeded run and is reproducible. If the hypothesis is not confirmed, report it honestly. Comment the code so a human can defend every choice.
