"""Metrics M1/M2/M3, embedding diagnostics, and the pre-registered verdict."""
import math

import numpy as np
import torch

def _center(X):
    return X - X.mean(axis=0, keepdims=True)


def procrustes_align(Zhat, Z):
    """Orthogonal Procrustes: argmin_{Q: Q^T Q = I_n} ||Zhat Q - Z||_F over K x n
    matrices with orthonormal columns (semi-orthogonal when K > n), plus the
    optimal scalar scale c* for min_c ||Z - c Zhat Q||_F.
    Inputs must be centered. Returns (Q, c)."""
    M = Zhat.T @ Z                                        # (K, n)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)      # economy SVD handles K >= n
    Q = U @ Vt                                            # (K, n), Q^T Q = I_n
    c = float(S.sum() / max((Zhat ** 2).sum(), 1e-12))    # tr(Q^T M) / ||Zhat||_F^2
    return Q, c


def _r2(Z, pred):
    """Fraction of variance of (centered) Z explained by pred."""
    return float(1.0 - ((Z - pred) ** 2).sum() / (Z ** 2).sum())


def m1_state(Zhat, Z):
    """State recovery: unrestricted-linear vs orthogonal(+scale) alignment.
    Returns (metrics dict, Q) -- Q is reused by M2/M3 so all three metrics share
    one coordinate change."""
    Zc, Zh = _center(Z), _center(Zhat)
    n = Z.shape[1]
    W, *_ = np.linalg.lstsq(Zh, Zc, rcond=None)
    R2_lin = _r2(Zc, Zh @ W)
    Q, c = procrustes_align(Zh, Zc)
    assert np.allclose(Q.T @ Q, np.eye(n), atol=1e-6), "Procrustes Q lost orthonormal columns"
    R2_orth = _r2(Zc, c * (Zh @ Q))
    R2_orth_noscale = _r2(Zc, Zh @ Q)
    assert R2_lin >= R2_orth - 1e-9, "orthogonal+scale maps are a subset of linear maps"
    return {
        "R2_lin": R2_lin,
        "R2_orth": R2_orth,
        "R2_orth_noscale": R2_orth_noscale,
        "proc_scale": c,
        "gap": R2_lin - R2_orth,
    }, Q


def principal_angles_deg(X, Y):
    """Principal angles (degrees, ascending) between col(X) and col(Y), via the
    Bjorck-Golub recipe: orthonormalize both, SVD of the cross-Gram. Identical to
    scipy.linalg.subspace_angles; implemented manually to avoid the dependency."""
    Qx, _ = np.linalg.qr(X)
    Qy, _ = np.linalg.qr(Y)
    s = np.clip(np.linalg.svd(Qx.T @ Qy, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(s))[::-1]  # ascending angles


def m2_action_axis(Q, Bhat, B, Vr=None, action_rank=0):
    """Action-axis recovery. B: true (n, m); Bhat: learned (K, m); Q from M1.

    Headline: cond_m(L) = sigma_1/sigma_m of L = (Q^T Bhat) pinv(B), the ratio
    over the top-m singular values. The spec's all-n ratio is +inf by
    construction for m < n (rank(L) <= m); restricting to the m potentially
    nonzero values is the faithful fix (== the spec's number when m >= n,
    == 1 for an exact scaled rotation regardless of cond(B)).
    """
    m = B.shape[1]
    B_al = Q.T @ Bhat                                # learned effect in true coordinates
    L = B_al @ np.linalg.pinv(B)                     # residual map: L B ~= B_al
    sv = np.linalg.svd(L, compute_uv=False)          # descending
    msv = sv[:m]
    cond_m = float("inf") if msv[-1] < 1e-12 * max(msv[0], 1e-300) else float(msv[0] / msv[-1])
    L_a = np.linalg.pinv(B) @ B_al                   # m x m action-coordinate map (secondary)
    sva = np.linalg.svd(L_a, compute_uv=False)
    cond_La = float("inf") if sva[-1] < 1e-12 * max(sva[0], 1e-300) else float(sva[0] / sva[-1])
    angles = principal_angles_deg(B_al, B)
    out = {
        "cond_m": cond_m,
        "cond_La": cond_La,
        "theta_max_deg": float(angles[-1]),
        "cos_mean": float(np.cos(np.radians(angles)).mean()),
        "norm_ratio": float(np.linalg.norm(B_al) / max(np.linalg.norm(B), 1e-12)),
        "excited_cos": float("nan"),
    }
    # Rank-deficient excitation: along the excited direction v, recovery should
    # still be good even though the unexcited direction is unidentifiable in
    # principle (pre-registered sub-prediction). cos(B v, B_al v):
    if Vr is not None and 0 < action_rank < m:
        v = np.asarray(Vr)[:, 0]
        u, w = B @ v, B_al @ v
        out["excited_cos"] = float(
            abs(u @ w) / max(np.linalg.norm(u) * np.linalg.norm(w), 1e-12))
    return out


def m3_dynamics(Q, Rhat, rho):
    """Dynamics recovery in aligned coordinates A = Q^T Rhat Q (n x n even when K > n)."""
    A = Q.T @ Rhat @ Q
    n = A.shape[0]
    rho_hat = float(np.trace(A) / n)        # LS projection of A onto {r I}; mean Re(eig)
    D = float(np.linalg.norm(A - rho * np.eye(n)))
    D_struct = float(np.linalg.norm(A - rho_hat * np.eye(n)))
    rho_mod = float(np.abs(np.linalg.eigvals(A)).mean())  # diagnostic only (upward-biased)
    K = Rhat.shape[0]
    if K > n:
        RQ = Rhat @ Q
        leak = float(np.linalg.norm(RQ - Q @ (Q.T @ RQ)) / max(np.linalg.norm(RQ), 1e-12))
    else:
        leak = 0.0
    return {
        "rho_hat": rho_hat,
        "rho_mod": rho_mod,
        "D_frob": D,
        "D_struct": D_struct,
        "D_rel": D / (rho * math.sqrt(n)),
        "leak": leak,
    }


def compute_all_metrics(encoder, predictor, eval_data, world, cfg, device):
    """All metrics for one trained run, on the held-out split, float64 on CPU."""
    encoder.eval()
    with torch.no_grad():
        Zhat = encoder(eval_data["x_t"].to(device)).cpu().double().numpy()
    Z = eval_data["z_t"].double().numpy()

    m1, Q = m1_state(Zhat, Z)
    # Embedding isotropy diagnostics (uncentered batch statistics):
    C = np.cov(Zhat.T)
    iso = {
        "emb_mean_norm": float(np.linalg.norm(Zhat.mean(0))),
        "emb_cov_err": float(np.linalg.norm(C - np.eye(cfg.K)) / math.sqrt(cfg.K)),
    }
    Rhat = predictor.Rhat.detach().cpu().double().numpy()
    m3 = m3_dynamics(Q, Rhat, cfg.rho)
    if predictor.action_conditioned:
        Bhat = predictor.Bhat.detach().cpu().double().numpy()
        m2 = m2_action_axis(Q, Bhat, world.B.double().numpy(),
                            Vr=world.Vr.double().numpy(), action_rank=cfg.action_rank)
    else:  # symmetric control: there is no action axis to recover
        m2 = {k: float("nan") for k in
              ("cond_m", "cond_La", "theta_max_deg", "cos_mean", "norm_ratio", "excited_cos")}
    row = {}
    row.update(m1); row.update(iso); row.update(m2); row.update(m3)
    row["pred_bias_norm"] = float(predictor.c.detach().norm())
    row["analytic_whiten_cond"] = world.analytic_whiten_cond
    return row

# Pre-registered decision rules. Defined BEFORE the sweep executes; the verdict
# cell at the end only applies them to the measured DataFrame. Justifications in
# the Section 7 markdown. NO metric value anywhere in this notebook is hand-typed.
R2_LIN_GATE = 0.90      # below: latents not linearly decodable -> inconclusive, not "broken"
MIN_CONV_FRAC = 0.8     # >= 80% of base seeds must pass the convergence flag
GAP_OK, GAP_BROKEN = 0.05, 0.15        # R2_lin - R2_orth bands
COND_OK, COND_BROKEN = 2.0, 5.0        # cond_m bands (V-JEPA 2 anchor ~1.5)
THETA_OK, THETA_BROKEN = 15.0, 30.0    # max principal angle bands (degrees)


def _mean_conv_finite(df, condition, col):
    """Mean of a metric over CONVERGED runs of a condition, finite values only.
    NaN when the condition is absent (e.g. QUICK grid) or has no finite values."""
    sub = df[(df.condition == condition) & (df.converged)]
    vals = sub[col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def _mean_all_finite(df, condition, col):
    """Mean over ALL runs of a condition (converged or not), finite values only.

    Used for the SIGReg-off control check, and only there. Phase-2 fix, disclosed
    in NOTES.md/RESULTS.md: in the notebook run, the converged-only filter
    silently voided this check because the lambda=0 control never converges --
    precisely BECAUSE it collapses, which is the predicted degradation the check
    exists to detect. Restricting a collapse check to converged runs is a
    selection bias against observing collapse."""
    sub = df[df.condition == condition]
    vals = sub[col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def verdict(df):
    """Apply the pre-registered rules to the sweep results."""
    out = {"verdict": "INCONCLUSIVE", "reasons": [], "secondary": []}
    base = df[df.condition == "base"]
    if len(base) == 0:
        out["reasons"].append("no base-condition runs found")
        return out
    conv_frac = float(base.converged.mean())
    R2_lin = _mean_conv_finite(df, "base", "R2_lin")
    gap = _mean_conv_finite(df, "base", "gap")
    cond_m = _mean_conv_finite(df, "base", "cond_m")
    theta = _mean_conv_finite(df, "base", "theta_max_deg")
    out.update(conv_frac=conv_frac, R2_lin=R2_lin, gap=gap, cond_m=cond_m, theta=theta)

    # Gate: a failed optimization licenses NO claim about the hypothesis.
    if conv_frac < MIN_CONV_FRAC:
        out["reasons"].append(
            f"only {conv_frac:.0%} of base seeds converged (< {MIN_CONV_FRAC:.0%}): "
            "optimization failed; no claim about the hypothesis is licensed")
        return out
    if not np.isfinite(R2_lin) or R2_lin < R2_LIN_GATE:
        out["reasons"].append(
            f"mean R2_lin = {R2_lin:.3f} < {R2_LIN_GATE}: latents are not linearly "
            "decodable, so 'recovered up to rotation?' is not well-posed for this run")
        return out

    s_ok = gap <= GAP_OK
    a_ok = (cond_m <= COND_OK) and (theta <= THETA_OK)
    broken = (gap > GAP_BROKEN) or (cond_m > COND_BROKEN) or (theta > THETA_BROKEN)
    out["verdict"] = "CONFIRMED" if (s_ok and a_ok) else ("BROKEN" if broken else "PARTIAL")
    out["reasons"].append(
        f"gap = {gap:.4f} (confirmed <= {GAP_OK}, broken > {GAP_BROKEN}); "
        f"cond_m = {cond_m:.3f} (confirmed <= {COND_OK}, broken > {COND_BROKEN}); "
        f"theta_max = {theta:.2f} deg (confirmed <= {THETA_OK}, broken > {THETA_BROKEN})")

    # ---- pre-registered secondary checks (reported, non-gating) ----
    def add(name, ok, detail):
        out["secondary"].append((name, bool(ok), detail))

    ro_b = _mean_conv_finite(df, "base", "R2_orth")
    ro_off = _mean_all_finite(df, "sigreg_off", "R2_orth")  # all runs: see _mean_all_finite
    c_off = _mean_all_finite(df, "sigreg_off", "cond_m")
    if np.isfinite(ro_off):
        degraded = (ro_b - ro_off >= 0.10) or (np.isfinite(c_off) and c_off >= 2 * cond_m)
        add("SIGReg-off control degrades", degraded,
            f"R2_orth {ro_b:.3f} -> {ro_off:.3f}; cond_m {cond_m:.3g} -> {c_off:.3g}" +
            ("" if degraded else
             "  [WARNING: control did not degrade -- the experiment lacks "
             "discriminative power on this axis and the interpretation must say so]"))
    ce_b = _mean_conv_finite(df, "base", "emb_cov_err")
    ce_p = _mean_conv_finite(df, "lam=0.05(paper)", "emb_cov_err")
    if np.isfinite(ce_p):
        r2_p = _mean_conv_finite(df, "lam=0.05(paper)", "R2_lin")
        add("paper-default lambda=0.05 shows the partial-collapse equilibrium",
            ce_p >= 2 * ce_b,
            f"emb_cov_err {ce_b:.3f} (lam=0.5) vs {ce_p:.3f} (lam=0.05); "
            f"R2_lin { _mean_conv_finite(df, 'base', 'R2_lin'):.3f} vs {r2_p:.3f}")
    ro_sym = _mean_conv_finite(df, "symmetric", "R2_orth")
    if np.isfinite(ro_sym):
        add("symmetric vs action-conditioned within 0.05 R2_orth",
            abs(ro_b - ro_sym) <= 0.05,
            f"R2_orth action = {ro_b:.3f} vs symmetric = {ro_sym:.3f}")
    exc = _mean_conv_finite(df, "excite_rank1", "excited_cos")
    if np.isfinite(exc):
        c_r1 = _mean_conv_finite(df, "excite_rank1", "cond_m")
        n_inf = int(np.isinf(df[df.condition == "excite_rank1"].cond_m).sum())
        add("rank-1 excitation: excited direction recovered, full map degenerate",
            exc >= 0.95,
            f"excited_cos = {exc:.3f} (want >= 0.95); cond_m finite-mean = {c_r1:.3g} "
            f"with {n_inf} infinite run(s) (blow-up expected)")
    rho_hat = _mean_conv_finite(df, "base", "rho_hat")
    d_rel = _mean_conv_finite(df, "base", "D_rel")
    rho_true = float(base.rho.iloc[0])
    add("dynamics recovered", abs(rho_hat - rho_true) <= 0.05 and d_rel <= 0.2,
        f"rho_hat = {rho_hat:.4f} (true {rho_true}); D_rel = {d_rel:.4f}")
    c_lit = _mean_conv_finite(df, "literal_noise", "cond_m")
    if np.isfinite(c_lit):
        pred_cond = _mean_conv_finite(df, "literal_noise", "analytic_whiten_cond")
        add("literal-spec noise shows the predicted whitening artifact",
            c_lit >= cond_m,
            f"cond_m literal = {c_lit:.3f} vs balanced = {cond_m:.3f}; "
            f"analytic prediction ~ {pred_cond:.3f}")
    return out


def render_verdict(v, quick):
    """Markdown report from verdict() output. Pure formatting -- no numbers
    originate here."""
    reading = {
        "CONFIRMED": "the orthogonal identifiability ambiguity **survives** "
                     "action-conditioning in this controlled world",
        "PARTIAL": "recovery sits **between** the pre-registered bands -- neither "
                   "cleanly orthogonal nor decisively distorted",
        "BROKEN": "recovery is **distorted beyond a rotation** -- identifiability "
                  "is more fragile under action-conditioning; this sharpens the open problem",
        "INCONCLUSIVE": "the precondition failed -- this run licenses **no claim** "
                        "about the hypothesis",
    }
    lines = []
    if quick:
        lines.append("> **QUICK MODE: smoke-test results from tiny dimensions and a few "
                     "steps. NOT science. Flip `QUICK = False` and re-run top-to-bottom "
                     "for the real verdict.**")
        lines.append("")
    lines.append(f"### Pre-registered verdict: **{v['verdict']}**")
    lines.append("")
    lines.append(f"Reading: {reading[v['verdict']]}.")
    lines.append("")
    if np.isfinite(v.get("R2_lin", float("nan"))):
        lines.append(
            f"Base condition, means over converged seeds ({v['conv_frac']:.0%} converged): "
            f"R2_lin = **{v['R2_lin']:.4f}**, gap = **{v['gap']:.4f}**, "
            f"cond_m = **{v['cond_m']:.3f}** (V-JEPA 2 anchor ~1.5), "
            f"theta_max = **{v['theta']:.2f} deg**.")
        lines.append("")
    for r in v["reasons"]:
        lines.append(f"- {r}")
    if v["secondary"]:
        lines.append("")
        lines.append("**Pre-registered secondary checks (non-gating):**")
        for name, ok, detail in v["secondary"]:
            lines.append(f"- {'PASS' if ok else 'FAIL'} -- {name}: {detail}")
    return "\n".join(lines)
