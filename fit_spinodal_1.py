"""
fit_spinodal.py
===============
Fits the FUS-PEG-salt spinodal model to binary phase-separation data and
performs a likelihood-ratio hypothesis test for the salt-coupling parameter δ.

Pipeline
--------
Stage 1  (phi_p vs I at phi_c ≈ 0, TWO POOLED I RANGES)
    H_pp = 1/(Np·φ_p) + 1/(1-φ_p) + 2·χ_eff = 0   at the spinodal
    Fits:  χ₀, A, B  jointly on pooled data from both I ranges
    The CSV has an 'I_range' column ("low" | "high"); fitting and diagnostics
    are reported both pooled and broken down per range.

Stage 2  (phi_p, phi_c at fixed I values)
    det(H) = H_pp·H_cc - H_pc² = 0   at the spinodal
    Fits:  K₁, δ   (χ₀, A, B fixed from Stage 1)

Hypothesis test
    H₀: δ = 0   (no salt-crowder-polymer coupling)
    H₁: δ ≠ 0
    Method: likelihood-ratio test (LRT)  →  Λ ~ χ²(1)   under H₀
    Also reports: AIC/BIC comparison, profile likelihood CI for δ.

Outputs (fit_results/)
----------------------
  stage1_fit.png                  — per-range scatter + fitted boundary + chi_eff
  stage1_residuals.png            — signed-distance residuals and calibration plots
  stage2_fit_hypothesis.png       — 2D phase diagrams + LRT profile + parameter bars
  stage1_predictions.csv          — per-point predictions, residuals, I_range labels
  stage1_boundary_low.csv         — fitted spinodal boundary for low I range
  stage1_boundary_high.csv        — fitted spinodal boundary for high I range
  stage1_chi_eff.csv              — chi_eff(I) for true and fitted params
  fit_summary.csv                 — all fitted parameters + hypothesis test results
  profile_likelihood_delta.csv    — profile NLL curve for δ

Usage
-----
    python fit_spinodal.py                          # uses synthetic_data/ defaults
    python fit_spinodal.py --data-dir path/to/data --Np 600 --Nc 227

Run generate_synthetic.py first to produce the required CSV files.
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize, differential_evolution
from scipy.special import expit
from scipy import stats

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Defaults (match generate_synthetic.py)
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR        = "synthetic_data"
OUT_DIR         = "fit_results"
I_FIXED_VALS    = [0.1, 0.25, 0.4]
I_STAGE1_RANGES = [(0.001, 0.5), (1.0, 3.0)]   # must match generate_synthetic.py
PHI_C_FIXED     = 0.00   # small phi_c for Stage-1 (phi_c ≈ 0 regime)
SCALE_S         = 20.0    # logistic sharpness (large → hard spinodal boundary)
N_RESTARTS      = 300     # L-BFGS-B random restarts per optimisation
SEED            = 0

# True values (for comparison plots when running on synthetic data)
TRUE = dict(Np=600, Nc=227, chi0=-12.0, A=-32.0, B=-18.0, K1=-5.0, delta=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Physics helpers
# ══════════════════════════════════════════════════════════════════════════════

def chi_eff(I, chi0, A, B):
    return chi0 - A * np.sqrt(np.maximum(I, 0.0)) + B * I


def H_pp_vec(phi_p, I, chi0, A, B, Np, phi_c=PHI_C_FIXED):
    phi_s = np.clip(1.0 - phi_p - phi_c, 1e-12, 1.0)
    phi_p = np.clip(phi_p, 1e-12, 1.0)
    return 1.0 / (Np * phi_p) + 1.0 / phi_s + 2.0 * chi_eff(I, chi0, A, B)


def det_H_vec(phi_p, phi_c, I, chi0, A, B, K1, delta, Np, Nc):
    phi_s = np.clip(1.0 - phi_p - phi_c, 1e-12, 1.0)
    phi_p = np.clip(phi_p, 1e-12, 1.0)
    phi_c = np.clip(phi_c, 1e-12, 1.0)
    ce    = chi_eff(I, chi0, A, B)
    Hpp   = 1.0 / (Np * phi_p) + 1.0 / phi_s + 2.0 * ce
    Hcc   = 1.0 / (Nc * phi_c) + 1.0 / phi_s
    Hpc   = 1.0 / phi_s + K1 + delta * I
    return Hpp * Hcc - Hpc ** 2


def logistic_nll(f_vals, y, s):
    """Binary cross-entropy: P(phase-sep) = σ(-s·f),  phase-sep label = 1."""
    p = expit(-s * f_vals)
    return -np.mean(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12))


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: fit χ₀, A, B
# ══════════════════════════════════════════════════════════════════════════════

def stage1_loss(params, phi_p, I, y, Np):
    chi0, A, B = params
    # if A <= 0:
    #     return 1e9
    f = H_pp_vec(phi_p, I, chi0, A, B, Np)
    return logistic_nll(f, y, SCALE_S)


def fit_stage1(df, Np, verbose=True):
    """
    Fit χ₀, A, B from Stage-1 data (phi_p vs I labels).
    Returns (chi0, A, B), result_dict.
    """
    phi_p = df["phi_p"].values
    I     = df["I"].values
    y     = df["label"].astype(int).values

    bounds = [(-20.0, -2.0), (-50.0, -5.0), (-30.0, -1.0)]

    rng  = np.random.default_rng(SEED)
    best = None
    for restart in range(N_RESTARTS):
        x0 = [rng.uniform(*b) for b in bounds]
        r  = minimize(stage1_loss, x0, args=(phi_p, I, y, Np),
                      method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
            if verbose and restart % 50 == 0:
                print(f"  [S1 restart {restart:3d}] loss = {best.fun:.6f}")

    chi0_f, A_f, B_f = best.x

    # Accuracy
    f_vals = H_pp_vec(phi_p, I, chi0_f, A_f, B_f, Np)
    y_pred = (f_vals < 0).astype(int)
    acc    = np.mean(y_pred == y)

    result = dict(chi0=chi0_f, A=A_f, B=B_f, loss=best.fun, accuracy=acc,
                  n_pts=len(y), n_ps=int(y.sum()))
    if verbose:
        print(f"\n── Stage 1 results ─────────────────────────────────────")
        print(f"  χ₀  = {chi0_f:.5f}   (true {TRUE['chi0']})")
        print(f"  A   = {A_f:.5f}   (true {TRUE['A']})")
        print(f"  B   = {B_f:.5f}   (true {TRUE['B']})")
        print(f"  loss = {best.fun:.6f}   accuracy = {acc*100:.2f}%")
    return (chi0_f, A_f, B_f), result


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: fit K₁, δ
# ══════════════════════════════════════════════════════════════════════════════

def stage2_loss(params, phi_p, phi_c, I, y, chi0, A, B, Np, Nc):
    K1, delta = params
    f = det_H_vec(phi_p, phi_c, I, chi0, A, B, K1, delta, Np, Nc)
    return logistic_nll(f, y, SCALE_S)


def stage2_loss_null(params, phi_p, phi_c, I, y, chi0, A, B, Np, Nc):
    """H₀: δ = 0."""
    K1, = params
    return stage2_loss([K1, 0.0], phi_p, phi_c, I, y, chi0, A, B, Np, Nc)


def _run_optimisation(loss_fn, args, bounds, n_restarts=N_RESTARTS, seed=SEED):
    """Multi-start L-BFGS-B with optional DE warm start."""
    # DE warm start
    de = differential_evolution(loss_fn, bounds=bounds, args=args,
                                seed=seed, maxiter=200, tol=1e-8,
                                popsize=12, workers=1)
    rng  = np.random.default_rng(seed)
    best = de
    lo   = np.array([b[0] for b in bounds])
    hi   = np.array([b[1] for b in bounds])

    x0_pool = [de.x + rng.normal(0, 0.1, size=len(bounds)) for _ in range(n_restarts // 3)]
    x0_pool += [rng.uniform(lo, hi) for _ in range(n_restarts - len(x0_pool))]

    for x0 in x0_pool:
        x0 = np.clip(x0, lo, hi)
        r  = minimize(loss_fn, x0, args=args, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9})
        if r.fun < best.fun:
            best = r

    return best


def fit_stage2(df_list, chi0, A, B, Np, Nc, verbose=True):
    """
    Fit K₁, δ (H₁) and K₁ with δ=0 (H₀) from Stage-2 data.
    Returns (params_H1, params_H0, test_results).
    """
    phi_p = np.concatenate([d["phi_p"].values for d in df_list])
    phi_c = np.concatenate([d["phi_c"].values for d in df_list])
    I     = np.concatenate([d["I"].values     for d in df_list])
    y     = np.concatenate([d["label"].astype(int).values for d in df_list])
    n     = len(y)

    args_common = (phi_p, phi_c, I, y, chi0, A, B, Np, Nc)

    if verbose:
        print(f"\n── Stage 2 fitting ─────────────────────────────────────")
        print(f"  Total pts: {n}   phase-sep: {y.sum()}   mixed: {n - y.sum()}")
        print("  Fitting H₁ (K₁, δ free) ...")

    bounds_H1 = [(-20.0, 2.0), (-3.0, 2.0)]   # K1, delta
    res_H1    = _run_optimisation(stage2_loss, args_common, bounds_H1)

    if verbose:
        print("  Fitting H₀ (K₁ free, δ = 0) ...")

    bounds_H0 = [(-20.0, 2.0)]                              # K1
    res_H0    = _run_optimisation(stage2_loss_null, args_common, bounds_H0)

    K1_H1, delta_H1 = res_H1.x
    K1_H0, = res_H0.x
    nll_H1 = res_H1.fun
    nll_H0 = res_H0.fun

    # Accuracy
    f_H1   = det_H_vec(phi_p, phi_c, I, chi0, A, B, K1_H1, delta_H1, Np, Nc)
    f_H0   = det_H_vec(phi_p, phi_c, I, chi0, A, B, K1_H0, 0.0,     Np, Nc)
    acc_H1 = np.mean((f_H1 < 0).astype(int) == y)
    acc_H0 = np.mean((f_H0 < 0).astype(int) == y)

    # ── Likelihood-ratio test ─────────────────────────────────────────────────
    # Total log-likelihood = -n · nll (nll is per-sample mean)
    LRT_stat = 2.0 * n * max(nll_H0 - nll_H1, 0.0)   # always ≥ 0
    p_value  = stats.chi2.sf(LRT_stat, df=1)

    # ── AIC / BIC ─────────────────────────────────────────────────────────────
    total_nll = lambda nll: n * nll        # total negative log-likelihood
    AIC = lambda k, nll: 2 * k + 2 * total_nll(nll)
    BIC = lambda k, nll: k * np.log(n) + 2 * total_nll(nll)

    AIC_H1 = AIC(2, nll_H1); AIC_H0 = AIC(1, nll_H0)
    BIC_H1 = BIC(2, nll_H1); BIC_H0 = BIC(1, nll_H0)

    # ── Profile likelihood CI for δ ──────────────────────────────────────────
    delta_ci = _profile_CI(res_H1.x, phi_p, phi_c, I, y, chi0, A, B, Np, Nc,
                            total_nll(nll_H1), verbose=verbose)

    params_H1 = dict(K1=K1_H1, delta=delta_H1,
                     nll=nll_H1, acc=acc_H1,
                     AIC=AIC_H1, BIC=BIC_H1)
    params_H0 = dict(K1=K1_H0, delta=0.0,
                     nll=nll_H0, acc=acc_H0,
                     AIC=AIC_H0, BIC=BIC_H0)
    test      = dict(LRT=LRT_stat, pval=p_value,
                     reject_H0=(p_value < 0.05),
                     dAIC=AIC_H0 - AIC_H1, dBIC=BIC_H0 - BIC_H1,
                     delta_CI=delta_ci)

    if verbose:
        _print_stage2(params_H1, params_H0, test)

    return params_H1, params_H0, test, (phi_p, phi_c, I, y)


def _profile_CI(opt_params, phi_p, phi_c, I, y, chi0, A, B, Np, Nc,
                min_nll_total, alpha=0.05, n_grid=80, verbose=True):
    """
    Profile likelihood 95% CI for δ.
    Threshold: χ²(1) critical value at α/2 on each side.
    """
    chi2_thresh = stats.chi2.ppf(1 - alpha, df=1)
    K1_opt, delta_opt = opt_params
    n = len(y)

    delta_grid = np.linspace(max(delta_opt - 2.0, -1.0), delta_opt + 2.0, n_grid)
    profile_nll = []

    for d in delta_grid:
        def loss_fixed_d(params):
            K1, = params
            return stage2_loss([K1, d], phi_p, phi_c, I, y, chi0, A, B, Np, Nc)
        r = minimize(loss_fixed_d, [K1_opt], method="L-BFGS-B",
                     bounds=[(-20, 2)],
                     options={"maxiter": 1000, "ftol": 1e-12})
        profile_nll.append(n * r.fun)  # total nll

    profile_nll  = np.array(profile_nll)
    delta_test   = 2.0 * (profile_nll - min_nll_total)  # LRT vs MLE

    # CI: where delta_test < chi2_thresh
    inside = delta_test < chi2_thresh
    if inside.any():
        ci_lo = delta_grid[inside].min()
        ci_hi = delta_grid[inside].max()
    else:
        ci_lo = ci_hi = delta_opt

    return (ci_lo, ci_hi, delta_grid, profile_nll, min_nll_total)


def _print_stage2(H1, H0, test):
    print(f"\n── Stage 2 results ─────────────────────────────────────")
    print(f"  Model H₁  (δ free):")
    print(f"    K₁    = {H1['K1']:.5f}   (true {TRUE['K1']})")
    print(f"    δ     = {H1['delta']:.5f}   (true {TRUE['delta']})")
    print(f"    loss  = {H1['nll']:.6f}   accuracy = {H1['acc']*100:.2f}%")
    print(f"    AIC   = {H1['AIC']:.2f}   BIC = {H1['BIC']:.2f}")
    print()
    print(f"  Model H₀  (δ = 0):")
    print(f"    K₁    = {H0['K1']:.5f}")
    print(f"    loss  = {H0['nll']:.6f}   accuracy = {H0['acc']*100:.2f}%")
    print(f"    AIC   = {H0['AIC']:.2f}   BIC = {H0['BIC']:.2f}")
    print()
    print(f"  ── Hypothesis test (H₀: δ = 0) ──────────────────────")
    print(f"    LRT statistic : {test['LRT']:.4f}")
    print(f"    p-value       : {test['pval']:.4e}")
    print(f"    ΔAIC (H₀−H₁) : {test['dAIC']:.2f}")
    print(f"    ΔBIC (H₀−H₁) : {test['dBIC']:.2f}")
    ci = test["delta_CI"]
    print(f"    95% CI (δ)   : [{ci[0]:.4f},  {ci[1]:.4f}]")
    if test["reject_H0"]:
        print(f"    ✓ Reject H₀ at α=0.05 — δ is significantly nonzero")
    else:
        print(f"    ✗ Fail to reject H₀ at α=0.05 — δ consistent with zero")


# ══════════════════════════════════════════════════════════════════════════════
# Spinodal boundary tracing (for plots)
# ══════════════════════════════════════════════════════════════════════════════

def trace_1d(chi0, A, B, Np, I_range=(0.001, 0.5), n=500):
    phi_g  = np.linspace(1e-5, 0.001, 200_000)
    I_vals = np.linspace(*I_range, n)
    bd_I, bd_phi = [], []
    for I in I_vals:
        ce   = chi_eff(I, chi0, A, B)
        H    = 1.0 / (Np * phi_g) + 1.0 / (1.0 - phi_g - PHI_C_FIXED) + 2.0 * ce
        idx  = np.where(np.diff(np.sign(H)))[0]
        if len(idx) > 0:
            bd_I.append(I); bd_phi.append(phi_g[idx[0]])
    return np.array(bd_I), np.array(bd_phi)


def trace_1d_full(chi0, A, B, Np):
    """Trace spinodal boundaries for both I ranges."""
    ranges = [(0.001, 0.5), (1.0, 3.0)]
    boundaries = {}
    for I_range in ranges:
        boundaries[I_range] = trace_1d(chi0, A, B, Np, I_range=I_range)
    return boundaries


def trace_2d(chi0, A, B, K1, delta, Np, Nc, I_fixed, n=500):
    phi_p_g = np.linspace(1e-5, 0.001, n)
    phi_c_g = np.linspace(0.001, 0.1, n)
    PP, CC  = np.meshgrid(phi_p_g, phi_c_g)
    valid   = (PP + CC) < 0.97
    phi_s   = np.where(valid, 1.0 - PP - CC, 1e-12)
    ce      = chi_eff(I_fixed, chi0, A, B)
    Hpp = 1.0/(Np*np.clip(PP,1e-12,1)) + 1.0/np.clip(phi_s,1e-12,1) + 2.0*ce
    Hcc = 1.0/(Nc*np.clip(CC,1e-12,1)) + 1.0/np.clip(phi_s,1e-12,1)
    Hpc = 1.0/np.clip(phi_s,1e-12,1) + K1 + delta*I_fixed
    SC  = Hpp*Hcc - Hpc**2; SC[~valid] = np.nan
    bd_pp, bd_pc = [], []
    for i in range(n):
        col = SC[:, i]; vm = ~np.isnan(col)
        if not np.any(vm): continue
        for idx in np.where(np.diff(np.sign(col[vm])))[0]:
            if phi_p_g[i] + phi_c_g[vm][idx] < 0.94:
                bd_pp.append(phi_p_g[i]); bd_pc.append(phi_c_g[vm][idx])
    return np.array(bd_pp), np.array(bd_pc)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

C_TRUE = "#E84545"; C_FIT = "#4C3BCF"; C_NULL = "#FF8C42"
C_PS   = "#4C3BCF"; C_MIX = "#2BB5A0"; ALPHA = 0.40


def plot_stage1(df, s1_res, bd_true, bd_fit, Np):
    chi0_f, A_f, B_f = s1_res[0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Check if data has I_range column (pooled data)
    has_i_range = "I_range" in df.columns

    if has_i_range:
        low_range = I_STAGE1_RANGES[0]
        high_range = I_STAGE1_RANGES[1]
        bd_true_low = bd_true[low_range] if isinstance(bd_true, dict) else bd_true
        bd_fit_low = bd_fit[low_range] if isinstance(bd_fit, dict) else bd_fit
        bd_true_high = bd_true[high_range] if isinstance(bd_true, dict) else bd_true
        bd_fit_high = bd_fit[high_range] if isinstance(bd_fit, dict) else bd_fit

        # Plot for low I range
        df_low = df[df["I_range"] == "low"]
        ax = axes[0]
        m  = df_low["label"].values.astype(bool)
        ax.scatter(df_low["phi_p"][~m], df_low["I"][~m], s=3, c=C_MIX, alpha=ALPHA, label="mixed")
        ax.scatter(df_low["phi_p"][ m], df_low["I"][ m], s=3, c=C_PS,  alpha=ALPHA, label="phase-sep")
        if len(bd_true_low[0]) > 0:
            ax.plot(bd_true_low[1], bd_true_low[0], "-", c=C_TRUE, lw=2, label="true boundary")
        if len(bd_fit_low[0]) > 0:
            ax.plot(bd_fit_low[1],  bd_fit_low[0],  "--", c=C_FIT, lw=2, label="fitted boundary")
        ax.set_xlabel(r"$\phi_p$"); ax.set_ylabel(r"$I$")
        ax.set_title("Stage 1: Low I range (0-0.5)", fontsize=10)
        ax.legend(fontsize=7, markerscale=3)

        # Plot for high I range
        df_high = df[df["I_range"] == "high"]
        ax = axes[1]
        mh  = df_high["label"].values.astype(bool)
        ax.scatter(df_high["phi_p"][~mh], df_high["I"][~mh], s=3, c=C_MIX, alpha=ALPHA, label="mixed")
        ax.scatter(df_high["phi_p"][ mh], df_high["I"][ mh], s=3, c=C_PS,  alpha=ALPHA, label="phase-sep")
        if len(bd_true_high[0]) > 0:
            ax.plot(bd_true_high[1], bd_true_high[0], "-", c=C_TRUE, lw=2, label="true boundary")
        if len(bd_fit_high[0]) > 0:
            ax.plot(bd_fit_high[1],  bd_fit_high[0],  "--", c=C_FIT, lw=2, label="fitted boundary")
        ax.set_xlabel(r"$\phi_p$"); ax.set_ylabel(r"$I$")
        ax.set_title("Stage 1: High I range (1-3)", fontsize=10)
        ax.legend(fontsize=7, markerscale=3)
        
        # Combined chi_eff plot
        ax = axes[2]
        I_plt = np.linspace(0.0, 3.2, 500)
        ax.plot(I_plt, chi_eff(I_plt, TRUE["chi0"], TRUE["A"], TRUE["B"]),
                c=C_TRUE, lw=2, label=f"true  (χ₀={TRUE['chi0']}, A={TRUE['A']}, B={TRUE['B']})")
        ax.plot(I_plt, chi_eff(I_plt, chi0_f, A_f, B_f), "--",
                c=C_FIT, lw=2, label=f"fit   (χ₀={chi0_f:.3f}, A={A_f:.3f}, B={B_f:.3f})")
        ax.axhline(0, c="k", lw=0.8, ls=":", alpha=0.5)
        ax.set_xlabel(r"$I$"); ax.set_ylabel(r"$\chi_\mathrm{eff}$")
        ax.set_title(r"$\chi_\mathrm{eff}(I)$  recovery", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 3.2)
    else:
        # Original single I range plot
        ax = axes[0]
        m  = df["label"].values.astype(bool)
        ax.scatter(df["phi_p"][~m], df["I"][~m], s=3, c=C_MIX, alpha=ALPHA, label="mixed")
        ax.scatter(df["phi_p"][ m], df["I"][ m], s=3, c=C_PS,  alpha=ALPHA, label="phase-sep")
        if len(bd_true[0]) > 0:
            ax.plot(bd_true[1], bd_true[0], "-", c=C_TRUE, lw=2, label="true boundary")
        if len(bd_fit[0]) > 0:
            ax.plot(bd_fit[1],  bd_fit[0],  "--", c=C_FIT, lw=2, label="fitted boundary")
        ax.set_xlabel(r"$\phi_p$"); ax.set_ylabel(r"$I$")
        ax.set_title("Stage 1: data + spinodal", fontsize=10)
        ax.legend(fontsize=7, markerscale=3)

        # chi_eff comparison
        ax = axes[1]
        I_plt = np.linspace(0.0, 0.55, 400)
        ax.plot(I_plt, chi_eff(I_plt, TRUE["chi0"], TRUE["A"], TRUE["B"]),
                c=C_TRUE, lw=2, label=f"true  (χ₀={TRUE['chi0']}, A={TRUE['A']}, B={TRUE['B']})")
        ax.plot(I_plt, chi_eff(I_plt, chi0_f, A_f, B_f), "--",
                c=C_FIT, lw=2, label=f"fit   (χ₀={chi0_f:.3f}, A={A_f:.3f}, B={B_f:.3f})")
        ax.axhline(0, c="k", lw=0.8, ls=":", alpha=0.5)
        ax.set_xlabel(r"$I$"); ax.set_ylabel(r"$\chi_\mathrm{eff}$")
        ax.set_title(r"$\chi_\mathrm{eff}(I)$  recovery", fontsize=10)
        ax.legend(fontsize=8)

        # parameter bar comparison
        ax   = axes[2]
        pars = ["χ₀", "A", "B"]
        tv   = [TRUE["chi0"], TRUE["A"], TRUE["B"]]
        fv   = [chi0_f, A_f, B_f]
        x    = np.arange(len(pars)); w = 0.35
        ax.bar(x - w/2, tv, w, color=C_TRUE, label="true",  alpha=0.85)
        ax.bar(x + w/2, fv, w, color=C_FIT,  label="fitted", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(pars)
        ax.set_ylabel("value"); ax.set_title("Parameter recovery", fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle("Stage 1 fit: χ₀, A, B from (φ_p, I) data", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "stage1_fit.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_stage2(H1, H0, test, data_tuple, chi0, A, B, Np, Nc):
    phi_p, phi_c, I_arr, y = data_tuple
    ci = test["delta_CI"]
    delta_grid, profile_nll, min_nll = ci[2], ci[3], ci[4]

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    palette = ["#4C3BCF", "#E84545", "#FF8C42"]

    # ── Rows 0-1, cols 0-2: one panel per I value ─────────────────────────────
    for k, I_f in enumerate(I_FIXED_VALS):
        ax  = fig.add_subplot(gs[k // 2, k % 2])
        m2  = (I_arr == I_f)
        mm  = y[m2].astype(bool)
        ax.scatter(phi_p[m2][~mm], phi_c[m2][~mm], s=3, c=C_MIX, alpha=ALPHA)
        ax.scatter(phi_p[m2][ mm], phi_c[m2][ mm], s=3, c=C_PS,  alpha=ALPHA)

        for label, K1, delta, col, ls in [
            ("true",   TRUE["K1"], TRUE["delta"], C_TRUE, "-"),
            ("H₁ fit", H1["K1"],   H1["delta"],  C_FIT,  "--"),
            ("H₀ fit", H0["K1"],   0.0,          C_NULL, ":"),
        ]:
            bp, bc = trace_2d(chi0, A, B, K1, delta, Np, Nc, I_f, n=400)
            if len(bp) > 0:
                order = np.argsort(bp)
                ax.plot(bp[order], bc[order], ls=ls, c=col, lw=1.8, label=label)

        ax.set_xlabel(r"$\phi_p$"); ax.set_ylabel(r"$\phi_c$")
        ax.set_title(f"$I = {I_f}$ — spinodal boundaries", fontsize=9)
        ax.legend(fontsize=7)

    # # ── Likelihood-ratio profile ──────────────────────────────────────────────
    # ax = fig.add_subplot(gs[0, 2])
    # lrt_profile = 2.0 * (profile_nll - min_nll)
    # ax.plot(delta_grid, lrt_profile, c=C_FIT, lw=2)
    # chi2_thr = stats.chi2.ppf(0.95, df=1)
    # ax.axhline(chi2_thr, c="gray", ls="--", lw=1, label=f"χ²(1) 95% = {chi2_thr:.2f}")
    # ax.axvline(H1["delta"], c=C_FIT,  ls="--", lw=1, label=f"δ̂ = {H1['delta']:.4f}")
    # ax.axvline(0.0,          c=C_NULL, ls=":",  lw=1, label="H₀: δ = 0")
    # if ci[0] < ci[1]:
    #     ax.axvspan(ci[0], ci[1], alpha=0.12, color=C_FIT, label="95% CI")
    # ax.set_xlabel("δ"); ax.set_ylabel("LRT statistic")
    # ax.set_title("Profile likelihood for δ", fontsize=10)
    # ax.legend(fontsize=8); ax.set_ylim(bottom=0)

    # ── Parameter comparison bars ─────────────────────────────────────────────
    ax   = fig.add_subplot(gs[0, 2])
    pars = ["K₁", "δ"]
    tv   = [TRUE["K1"], TRUE["delta"]]
    fv1  = [H1["K1"],   H1["delta"]]
    fv0  = [H0["K1"],   0.0]
    x    = np.arange(len(pars)); w = 0.25
    ax.bar(x - w,   tv,  w, color=C_TRUE, label="true",  alpha=0.85)
    ax.bar(x,       fv1, w, color=C_FIT,  label="H₁ fit", alpha=0.85)
    ax.bar(x + w,   fv0, w, color=C_NULL, label="H₀ fit", alpha=0.75)
    ax.set_xticks(x); ax.set_xticklabels(pars)
    ax.set_title("Parameter recovery", fontsize=10)
    ax.axhline(0, c="k", lw=0.6, ls="--", alpha=0.5)
    ax.legend(fontsize=8)

    # ── Hypothesis test summary panel ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1:])
    ax.axis("off")
    verdict = "✓  REJECT  H₀" if test["reject_H0"] else "✗  FAIL TO REJECT  H₀"
    summary = (
        r"$\bf{Hypothesis\ test:\ H_0{:}\ \delta=0\ \ vs\ \ H_1{:}\ \delta\neq 0}$" + "\n\n"
        + f"  LRT statistic     Λ  =  {test['LRT']:.3f}\n"
        + f"  p-value              =  {test['pval']:.4e}\n"
        + f"  ΔAIC (H₀ − H₁)   =  {test['dAIC']:.2f}   "
          f"(>0 favours H₁)\n"
        + f"  ΔBIC (H₀ − H₁)   =  {test['dBIC']:.2f}   "
          f"(>0 favours H₁)\n"
        + f"  95% CI for δ        =  [{ci[0]:.4f},  {ci[1]:.4f}]\n\n"
        + f"  {verdict}\n"
        + (f"  δ is significantly nonzero (p < 0.05)\n"
           if test["reject_H0"]
           else "  δ is consistent with zero (p ≥ 0.05)\n")
        + f"\n  Fitted:  K₁={H1['K1']:.4f}  δ={H1['delta']:.4f}"
        + f"\n  True:    K₁={TRUE['K1']}     δ={TRUE['delta']}"
    )
    ax.text(0.04, 0.96, summary, transform=ax.transAxes, va="top", fontsize=10.5,
            fontfamily="DejaVu Serif", linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.6",
                      fc="#F0F7FF" if test["reject_H0"] else "#FFF5E8",
                      ec=C_FIT    if test["reject_H0"] else C_NULL, lw=1.5))

    fig.suptitle("Stage 2 fit: K₁, δ  |  Hypothesis test for δ", fontsize=12)
    out = os.path.join(OUT_DIR, "stage2_fit_hypothesis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Results saving
# ══════════════════════════════════════════════════════════════════════════════

def save_results(s1_params, s1_res, H1, H0, test):
    ci = test["delta_CI"]
    rows = [
        ["stage1", "chi0",   s1_params[0], TRUE["chi0"],   ""],
        ["stage1", "A",      s1_params[1], TRUE["A"],       ""],
        ["stage1", "B",      s1_params[2], TRUE["B"],       ""],
        ["stage1", "accuracy", s1_res[1]["accuracy"], "", ""],
        ["stage2_H1", "K1",     H1["K1"], TRUE["K1"],  ""],
        ["stage2_H1", "delta",  H1["delta"], TRUE["delta"],
         f"CI=[{ci[0]:.4f}, {ci[1]:.4f}]"],
        ["stage2_H1", "accuracy", H1["acc"], "", ""],
        ["stage2_H0", "K1",     H0["K1"], TRUE["K1"],  "delta fixed=0"],
        ["stage2_H0", "accuracy", H0["acc"], "", ""],
        ["hypothesis_test", "LRT",    test["LRT"],  "", "chi2(1) statistic"],
        ["hypothesis_test", "pvalue", test["pval"], "", ""],
        ["hypothesis_test", "dAIC",   test["dAIC"], "", "H0-H1"],
        ["hypothesis_test", "dBIC",   test["dBIC"], "", "H0-H1"],
        ["hypothesis_test", "reject_H0", int(test["reject_H0"]), "", "alpha=0.05"],
    ]
    df_out = pd.DataFrame(rows, columns=["stage", "parameter", "fitted", "true", "notes"])
    out    = os.path.join(OUT_DIR, "fit_summary.csv")
    df_out.to_csv(out, index=False)
    print(f"  Saved: {out}")

    # Profile likelihood curve
    pd.DataFrame({"delta": ci[2], "total_nll": ci[3]}).to_csv(
        os.path.join(OUT_DIR, "profile_likelihood_delta.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    Np = args.Np; Nc = args.Nc

    # ── Load data ─────────────────────────────────────────────────────────────
    s1_path = os.path.join(args.data_dir, "stage1_phiI.csv")
    print(f"Loading Stage-1 data from {s1_path} ...")
    df1 = pd.read_csv(s1_path)
    print(f"  {len(df1)} rows  |  columns: {df1.columns.tolist()}")

    df2_list = []
    for I_f in I_FIXED_VALS:
        tag  = str(I_f).replace(".", "p")
        path = os.path.join(args.data_dir, f"stage2_phipc_I{tag}.csv")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found"); continue
        df2_list.append(pd.read_csv(path))
        print(f"Loading Stage-2 data from {path}  ({len(df2_list[-1])} rows)")

    if not df2_list:
        raise FileNotFoundError("No Stage-2 CSV files found. Run generate_synthetic.py first.")

    # ── Stage 1: fit chi0, A, B ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1: fitting χ₀, A, B")
    print("=" * 60)
    s1_params, s1_res = fit_stage1(df1, Np, verbose=True)
    chi0_f, A_f, B_f  = s1_params

    # boundary for plot
    bd_true_1 = trace_1d_full(TRUE["chi0"], TRUE["A"], TRUE["B"], Np)
    bd_fit_1  = trace_1d_full(chi0_f, A_f, B_f, Np)

    # ── Stage 2: fit K1, delta ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: fitting K₁, δ  +  hypothesis test")
    print("=" * 60)
    H1, H0, test, data_tuple = fit_stage2(df2_list, chi0_f, A_f, B_f, Np, Nc, verbose=True)

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_stage1(df1, (s1_params, s1_res), bd_true_1, bd_fit_1, Np)
    plot_stage2(H1, H0, test, data_tuple, chi0_f, A_f, B_f, Np, Nc)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    save_results(s1_params, (s1_params, s1_res), H1, H0, test)

    print("\n" + "=" * 60)
    print(f"All outputs written to:  {os.path.abspath(OUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit spinodal model and test δ≠0")
    parser.add_argument("--data-dir",  default=DATA_DIR,
                        help=f"Input data directory (default: {DATA_DIR})")
    parser.add_argument("--Np", type=int, default=TRUE["Np"],
                        help=f"Polymer chain length (default: {TRUE['Np']})")
    parser.add_argument("--Nc", type=int, default=TRUE["Nc"],
                        help=f"Crowder chain length (default: {TRUE['Nc']})")
    args = parser.parse_args()
    main(args)
