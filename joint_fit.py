"""
joint_fit.py
============
Jointly fits the FUS-PEG-salt spinodal model using only Stage-2 data.

Unlike fit_spinodal_1.py, this script does not first fit chi0, A, B from
Stage-1 data. It fits chi0, A, B, K1, and delta together from the
(phi_p, phi_c, I, label) Stage-2 slices.

Outputs are written to ./joint_fit_results/ and mirror the main diagnostics
from fit_spinodal_1.py:
  stage1_fit.png
  stage2_fit_hypothesis.png
  fit_summary.csv
  profile_likelihood_delta.csv
  stage2_predictions.csv
  stage1_boundary_low.csv
  stage1_boundary_high.csv
  stage1_chi_eff.csv
"""

import argparse
import os
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit

warnings.filterwarnings("ignore")


DATA_DIR = "synthetic_data"
OUT_DIR = "joint_fit_results"
I_FIXED_VALS = [0.1, 0.25, 0.4]
I_STAGE1_RANGES = [(0.001, 0.5), (1.0, 3.0)]
PHI_C_FIXED = 0.0
SCALE_S = 20.0
N_RESTARTS = 300
SEED = 0

# True values for synthetic-data comparison plots.
TRUE = dict(Np=600, Nc=227, chi0=-12.0, A=-32.0, B=-18.0, K1=-5.0, delta=-1.0)

C_TRUE = "#E84545"
C_FIT = "#4C3BCF"
C_NULL = "#FF8C42"
C_PS = "#4C3BCF"
C_MIX = "#2BB5A0"
ALPHA = 0.40


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
    ce = chi_eff(I, chi0, A, B)
    Hpp = 1.0 / (Np * phi_p) + 1.0 / phi_s + 2.0 * ce
    Hcc = 1.0 / (Nc * phi_c) + 1.0 / phi_s
    Hpc = 1.0 / phi_s + K1 + delta * I
    return Hpp * Hcc - Hpc**2


def logistic_nll(f_vals, y, s):
    p = expit(-s * f_vals)
    return -np.mean(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12))


def joint_loss(params, phi_p, phi_c, I, y, Np, Nc):
    chi0, A, B, K1, delta = params
    f = det_H_vec(phi_p, phi_c, I, chi0, A, B, K1, delta, Np, Nc)
    return logistic_nll(f, y, SCALE_S)


def joint_loss_null(params, phi_p, phi_c, I, y, Np, Nc):
    chi0, A, B, K1 = params
    return joint_loss([chi0, A, B, K1, 0.0], phi_p, phi_c, I, y, Np, Nc)


def _run_optimisation(loss_fn, args, bounds, n_restarts=N_RESTARTS, seed=SEED):
    de = differential_evolution(
        loss_fn,
        bounds=bounds,
        args=args,
        seed=seed,
        maxiter=200,
        tol=1e-8,
        popsize=12,
        workers=1,
    )
    rng = np.random.default_rng(seed)
    best = de
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    x0_pool = [de.x + rng.normal(0, 0.1, size=len(bounds)) for _ in range(n_restarts // 3)]
    x0_pool += [rng.uniform(lo, hi) for _ in range(n_restarts - len(x0_pool))]

    for x0 in x0_pool:
        x0 = np.clip(x0, lo, hi)
        r = minimize(
            loss_fn,
            x0,
            args=args,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9},
        )
        if r.fun < best.fun:
            best = r
    return best


def fit_joint(df_list, Np, Nc, verbose=True):
    phi_p = np.concatenate([d["phi_p"].values for d in df_list])
    phi_c = np.concatenate([d["phi_c"].values for d in df_list])
    I = np.concatenate([d["I"].values for d in df_list])
    y = np.concatenate([d["label"].astype(int).values for d in df_list])
    n = len(y)

    args_common = (phi_p, phi_c, I, y, Np, Nc)

    if verbose:
        print("\n" + "=" * 60)
        print("JOINT FIT FROM STAGE-2 DATA")
        print("=" * 60)
        print(f"  Total pts: {n}   phase-sep: {y.sum()}   mixed: {n - y.sum()}")
        print("  Fitting H1 (chi0, A, B, K1, delta free) ...")

    bounds_H1 = [(-30.0, 5.0), (-80.0, 20.0), (-60.0, 20.0), (-20.0, 5.0), (-8.0, 8.0)]
    res_H1 = _run_optimisation(joint_loss, args_common, bounds_H1)

    if verbose:
        print("  Fitting H0 (chi0, A, B, K1 free, delta = 0) ...")

    bounds_H0 = [(-30.0, 5.0), (-80.0, 20.0), (-60.0, 20.0), (-20.0, 5.0)]
    res_H0 = _run_optimisation(joint_loss_null, args_common, bounds_H0)

    chi0_H1, A_H1, B_H1, K1_H1, delta_H1 = res_H1.x
    chi0_H0, A_H0, B_H0, K1_H0 = res_H0.x
    nll_H1 = res_H1.fun
    nll_H0 = res_H0.fun

    f_H1 = det_H_vec(phi_p, phi_c, I, chi0_H1, A_H1, B_H1, K1_H1, delta_H1, Np, Nc)
    f_H0 = det_H_vec(phi_p, phi_c, I, chi0_H0, A_H0, B_H0, K1_H0, 0.0, Np, Nc)
    acc_H1 = np.mean((f_H1 < 0).astype(int) == y)
    acc_H0 = np.mean((f_H0 < 0).astype(int) == y)

    LRT_stat = 2.0 * n * max(nll_H0 - nll_H1, 0.0)
    p_value = stats.chi2.sf(LRT_stat, df=1)

    total_nll = lambda nll: n * nll
    AIC = lambda k, nll: 2 * k + 2 * total_nll(nll)
    BIC = lambda k, nll: k * np.log(n) + 2 * total_nll(nll)
    AIC_H1 = AIC(5, nll_H1)
    AIC_H0 = AIC(4, nll_H0)
    BIC_H1 = BIC(5, nll_H1)
    BIC_H0 = BIC(4, nll_H0)

    delta_ci = _profile_CI(
        res_H1.x,
        phi_p,
        phi_c,
        I,
        y,
        Np,
        Nc,
        total_nll(nll_H1),
        verbose=verbose,
    )

    H1 = dict(
        chi0=chi0_H1,
        A=A_H1,
        B=B_H1,
        K1=K1_H1,
        delta=delta_H1,
        nll=nll_H1,
        acc=acc_H1,
        AIC=AIC_H1,
        BIC=BIC_H1,
    )
    H0 = dict(
        chi0=chi0_H0,
        A=A_H0,
        B=B_H0,
        K1=K1_H0,
        delta=0.0,
        nll=nll_H0,
        acc=acc_H0,
        AIC=AIC_H0,
        BIC=BIC_H0,
    )
    test = dict(
        LRT=LRT_stat,
        pval=p_value,
        reject_H0=(p_value < 0.05),
        dAIC=AIC_H0 - AIC_H1,
        dBIC=BIC_H0 - BIC_H1,
        delta_CI=delta_ci,
    )

    if verbose:
        _print_joint(H1, H0, test)

    predictions = pd.DataFrame(
        {
            "phi_p": phi_p,
            "phi_c": phi_c,
            "I": I,
            "label": y.astype(bool),
            "det_H1": f_H1,
            "pred_H1": (f_H1 < 0).astype(bool),
            "det_H0": f_H0,
            "pred_H0": (f_H0 < 0).astype(bool),
        }
    )
    return H1, H0, test, (phi_p, phi_c, I, y), predictions


def _profile_CI(opt_params, phi_p, phi_c, I, y, Np, Nc, min_nll_total, alpha=0.05, n_grid=80, verbose=True):
    chi2_thresh = stats.chi2.ppf(1 - alpha, df=1)
    chi0_opt, A_opt, B_opt, K1_opt, delta_opt = opt_params
    n = len(y)
    delta_grid = np.linspace(max(delta_opt - 2.0, -8.0), min(delta_opt + 2.0, 8.0), n_grid)
    profile_nll = []

    for d in delta_grid:
        def loss_fixed_d(params):
            chi0, A, B, K1 = params
            return joint_loss([chi0, A, B, K1, d], phi_p, phi_c, I, y, Np, Nc)

        r = minimize(
            loss_fixed_d,
            [chi0_opt, A_opt, B_opt, K1_opt],
            method="L-BFGS-B",
            bounds=[(-30, 5), (-80, 20), (-60, 20), (-20, 5)],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        profile_nll.append(n * r.fun)

    profile_nll = np.array(profile_nll)
    delta_test = 2.0 * (profile_nll - min_nll_total)
    inside = delta_test < chi2_thresh
    if inside.any():
        ci_lo = delta_grid[inside].min()
        ci_hi = delta_grid[inside].max()
    else:
        ci_lo = ci_hi = delta_opt
    return (ci_lo, ci_hi, delta_grid, profile_nll, min_nll_total)


def _print_joint(H1, H0, test):
    print("\nJoint fit results")
    for name in ["chi0", "A", "B", "K1", "delta"]:
        print(f"  H1 {name:5s} = {H1[name]: .5f}   (true {TRUE[name]})")
    print(f"  H1 loss = {H1['nll']:.6f}   accuracy = {H1['acc'] * 100:.2f}%")
    print(f"  H1 AIC  = {H1['AIC']:.2f}   BIC = {H1['BIC']:.2f}")
    print()
    for name in ["chi0", "A", "B", "K1"]:
        print(f"  H0 {name:5s} = {H0[name]: .5f}")
    print("  H0 delta =  0.00000")
    print(f"  H0 loss = {H0['nll']:.6f}   accuracy = {H0['acc'] * 100:.2f}%")
    print(f"  H0 AIC  = {H0['AIC']:.2f}   BIC = {H0['BIC']:.2f}")
    print()
    ci = test["delta_CI"]
    print("Hypothesis test (H0: delta = 0)")
    print(f"  LRT statistic : {test['LRT']:.4f}")
    print(f"  p-value       : {test['pval']:.4e}")
    print(f"  delta 95% CI  : [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  dAIC H0-H1    : {test['dAIC']:.2f}")
    print(f"  dBIC H0-H1    : {test['dBIC']:.2f}")


def trace_1d(chi0, A, B, Np, I_range=(0.001, 0.5), n=500):
    phi_g = np.linspace(1e-5, 0.002, 200_000)
    I_vals = np.linspace(*I_range, n)
    bd_I, bd_phi = [], []
    for I in I_vals:
        H = H_pp_vec(phi_g, I, chi0, A, B, Np)
        idx = np.where(np.diff(np.sign(H)))[0]
        if len(idx) > 0:
            bd_I.append(I)
            bd_phi.append(phi_g[idx[0]])
    return np.array(bd_I), np.array(bd_phi)


def trace_1d_full(chi0, A, B, Np):
    return {r: trace_1d(chi0, A, B, Np, I_range=r) for r in I_STAGE1_RANGES}


def trace_2d(chi0, A, B, K1, delta, Np, Nc, I_fixed, n=500):
    phi_p_g = np.linspace(1e-5, 0.001, n)
    phi_c_g = np.linspace(0.005, 0.1, n)
    PP, CC = np.meshgrid(phi_p_g, phi_c_g)
    valid = (PP + CC) < 0.97
    phi_s = np.where(valid, 1.0 - PP - CC, 1e-12)
    ce = chi_eff(I_fixed, chi0, A, B)
    Hpp = 1.0 / (Np * np.clip(PP, 1e-12, 1)) + 1.0 / np.clip(phi_s, 1e-12, 1) + 2.0 * ce
    Hcc = 1.0 / (Nc * np.clip(CC, 1e-12, 1)) + 1.0 / np.clip(phi_s, 1e-12, 1)
    Hpc = 1.0 / np.clip(phi_s, 1e-12, 1) + K1 + delta * I_fixed
    SC = Hpp * Hcc - Hpc**2
    SC[~valid] = np.nan
    bd_pp, bd_pc = [], []
    for i in range(n):
        col = SC[:, i]
        vm = ~np.isnan(col)
        if not np.any(vm):
            continue
        for idx in np.where(np.diff(np.sign(col[vm])))[0]:
            if phi_p_g[i] + phi_c_g[vm][idx] < 0.94:
                bd_pp.append(phi_p_g[i])
                bd_pc.append(phi_c_g[vm][idx])
    return np.array(bd_pp), np.array(bd_pc)


def plot_stage1_implied(H1, H0, Np):
    bd_true = trace_1d_full(TRUE["chi0"], TRUE["A"], TRUE["B"], Np)
    bd_fit = trace_1d_full(H1["chi0"], H1["A"], H1["B"], Np)
    bd_null = trace_1d_full(H0["chi0"], H0["A"], H0["B"], Np)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, I_range, title in [
        (axes[0], I_STAGE1_RANGES[0], "Implied Stage 1: Low I range (0-0.5)"),
        (axes[1], I_STAGE1_RANGES[1], "Implied Stage 1: High I range (1-3)"),
    ]:
        for bd, color, ls, label in [
            (bd_true[I_range], C_TRUE, "-", "true boundary"),
            (bd_fit[I_range], C_FIT, "--", "H1 fitted boundary"),
            (bd_null[I_range], C_NULL, ":", "H0 fitted boundary"),
        ]:
            if len(bd[0]) > 0:
                ax.plot(bd[1], bd[0], ls=ls, c=color, lw=2, label=label)
        ax.set_xlabel(r"$\phi_p$")
        ax.set_ylabel(r"$I$")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)

    ax = axes[2]
    I_plt = np.linspace(0.0, 3.2, 500)
    ax.plot(I_plt, chi_eff(I_plt, TRUE["chi0"], TRUE["A"], TRUE["B"]), c=C_TRUE, lw=2, label="true")
    ax.plot(I_plt, chi_eff(I_plt, H1["chi0"], H1["A"], H1["B"]), "--", c=C_FIT, lw=2, label="H1 fit")
    ax.plot(I_plt, chi_eff(I_plt, H0["chi0"], H0["A"], H0["B"]), ":", c=C_NULL, lw=2, label="H0 fit")
    ax.axhline(0, c="k", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel(r"$I$")
    ax.set_ylabel(r"$\chi_\mathrm{eff}$")
    ax.set_title(r"$\chi_\mathrm{eff}(I)$ recovery", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 3.2)

    fig.suptitle("Stage 1 diagnostics implied by joint Stage-2 fit", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "stage1_fit.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return bd_fit


def plot_stage2(H1, H0, test, data_tuple, Np, Nc):
    phi_p, phi_c, I_arr, y = data_tuple
    ci = test["delta_CI"]
    delta_grid, profile_nll, min_nll = ci[2], ci[3], ci[4]

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    for k, I_f in enumerate(I_FIXED_VALS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        m2 = I_arr == I_f
        mm = y[m2].astype(bool)
        ax.scatter(phi_p[m2][~mm], phi_c[m2][~mm], s=3, c=C_MIX, alpha=ALPHA)
        ax.scatter(phi_p[m2][mm], phi_c[m2][mm], s=3, c=C_PS, alpha=ALPHA)
        for label, params, col, ls in [
            ("true", TRUE, C_TRUE, "-"),
            ("H1 fit", H1, C_FIT, "--"),
            ("H0 fit", H0, C_NULL, ":"),
        ]:
            bp, bc = trace_2d(
                params["chi0"], params["A"], params["B"], params["K1"], params["delta"], Np, Nc, I_f, n=400
            )
            if len(bp) > 0:
                order = np.argsort(bp)
                ax.plot(bp[order], bc[order], ls=ls, c=col, lw=1.8, label=label)
        ax.set_xlabel(r"$\phi_p$")
        ax.set_ylabel(r"$\phi_c$")
        ax.set_title(f"$I = {I_f}$ - spinodal boundaries", fontsize=9)
        ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 2])
    lrt_profile = 2.0 * (profile_nll - min_nll)
    ax.plot(delta_grid, lrt_profile, c=C_FIT, lw=2)
    chi2_thr = stats.chi2.ppf(0.95, df=1)
    ax.axhline(chi2_thr, c="gray", ls="--", lw=1, label=f"chi2(1) 95% = {chi2_thr:.2f}")
    ax.axvline(H1["delta"], c=C_FIT, ls="--", lw=1, label=f"delta_hat = {H1['delta']:.4f}")
    ax.axvline(0.0, c=C_NULL, ls=":", lw=1, label="H0: delta = 0")
    if ci[0] < ci[1]:
        ax.axvspan(ci[0], ci[1], alpha=0.12, color=C_FIT, label="95% CI")
    ax.set_xlabel("delta")
    ax.set_ylabel("LRT statistic")
    ax.set_title("Profile likelihood for delta", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    ax = fig.add_subplot(gs[0, 3])
    pars = ["chi0", "A", "B", "K1", "delta"]
    tv = [TRUE[p] for p in pars]
    fv1 = [H1[p] for p in pars]
    fv0 = [H0[p] for p in pars]
    x = np.arange(len(pars))
    w = 0.25
    ax.bar(x - w, tv, w, color=C_TRUE, label="true", alpha=0.85)
    ax.bar(x, fv1, w, color=C_FIT, label="H1 fit", alpha=0.85)
    ax.bar(x + w, fv0, w, color=C_NULL, label="H0 fit", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(pars, rotation=25)
    ax.set_title("Parameter recovery", fontsize=10)
    ax.axhline(0, c="k", lw=0.6, ls="--", alpha=0.5)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    verdict = "REJECT H0" if test["reject_H0"] else "FAIL TO REJECT H0"
    summary = (
        r"$\bf{Joint\ Stage\ 2\ fit:\ H_0{:}\ \delta=0\ vs\ H_1{:}\ \delta\neq0}$"
        + "\n\n"
        + f"  LRT statistic     =  {test['LRT']:.3f}\n"
        + f"  p-value           =  {test['pval']:.4e}\n"
        + f"  dAIC (H0 - H1)    =  {test['dAIC']:.2f}\n"
        + f"  dBIC (H0 - H1)    =  {test['dBIC']:.2f}\n"
        + f"  95% CI for delta  =  [{ci[0]:.4f},  {ci[1]:.4f}]\n\n"
        + f"  {verdict}\n\n"
        + f"  H1 fit: chi0={H1['chi0']:.4f}  A={H1['A']:.4f}  B={H1['B']:.4f}\n"
        + f"          K1={H1['K1']:.4f}  delta={H1['delta']:.4f}\n"
        + f"  True:   chi0={TRUE['chi0']}  A={TRUE['A']}  B={TRUE['B']}\n"
        + f"          K1={TRUE['K1']}  delta={TRUE['delta']}"
    )
    ax.text(
        0.04,
        0.96,
        summary,
        transform=ax.transAxes,
        va="top",
        fontsize=10.5,
        fontfamily="DejaVu Serif",
        linespacing=1.7,
        bbox=dict(boxstyle="round,pad=0.6", fc="#F0F7FF" if test["reject_H0"] else "#FFF5E8", ec=C_FIT, lw=1.5),
    )

    fig.suptitle("Joint fit from Stage-2 data | Hypothesis test for delta", fontsize=12)
    out = os.path.join(OUT_DIR, "stage2_fit_hypothesis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def save_results(H1, H0, test, predictions, bd_fit):
    ci = test["delta_CI"]
    rows = []
    for p in ["chi0", "A", "B", "K1", "delta"]:
        note = f"CI=[{ci[0]:.4f}, {ci[1]:.4f}]" if p == "delta" else ""
        rows.append(["joint_H1", p, H1[p], TRUE[p], note])
    rows.append(["joint_H1", "accuracy", H1["acc"], "", ""])
    for p in ["chi0", "A", "B", "K1", "delta"]:
        note = "delta fixed=0" if p == "delta" else ""
        rows.append(["joint_H0", p, H0[p], TRUE[p], note])
    rows.append(["joint_H0", "accuracy", H0["acc"], "", ""])
    rows += [
        ["hypothesis_test", "LRT", test["LRT"], "", "chi2(1) statistic"],
        ["hypothesis_test", "pvalue", test["pval"], "", ""],
        ["hypothesis_test", "dAIC", test["dAIC"], "", "H0-H1"],
        ["hypothesis_test", "dBIC", test["dBIC"], "", "H0-H1"],
        ["hypothesis_test", "reject_H0", int(test["reject_H0"]), "", "alpha=0.05"],
    ]
    pd.DataFrame(rows, columns=["stage", "parameter", "fitted", "true", "notes"]).to_csv(
        os.path.join(OUT_DIR, "fit_summary.csv"), index=False
    )
    predictions.to_csv(os.path.join(OUT_DIR, "stage2_predictions.csv"), index=False)
    pd.DataFrame({"delta": ci[2], "total_nll": ci[3]}).to_csv(
        os.path.join(OUT_DIR, "profile_likelihood_delta.csv"), index=False
    )

    for name, I_range in [("low", I_STAGE1_RANGES[0]), ("high", I_STAGE1_RANGES[1])]:
        bd_I, bd_phi = bd_fit[I_range]
        pd.DataFrame({"I": bd_I, "phi_p": bd_phi}).to_csv(
            os.path.join(OUT_DIR, f"stage1_boundary_{name}.csv"), index=False
        )

    I_grid = np.linspace(0.0, 3.2, 500)
    pd.DataFrame(
        {
            "I": I_grid,
            "chi_eff_true": chi_eff(I_grid, TRUE["chi0"], TRUE["A"], TRUE["B"]),
            "chi_eff_H1": chi_eff(I_grid, H1["chi0"], H1["A"], H1["B"]),
            "chi_eff_H0": chi_eff(I_grid, H0["chi0"], H0["A"], H0["B"]),
        }
    ).to_csv(os.path.join(OUT_DIR, "stage1_chi_eff.csv"), index=False)
    print(f"  Saved CSV outputs in: {OUT_DIR}")


def load_stage2(data_dir):
    df2_list = []
    for I_f in I_FIXED_VALS:
        tag = str(I_f).replace(".", "p")
        path = os.path.join(data_dir, f"stage2_phipc_I{tag}.csv")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(path)
        df2_list.append(df)
        print(f"Loading Stage-2 data from {path} ({len(df)} rows)")
    if not df2_list:
        raise FileNotFoundError("No Stage-2 CSV files found. Run generate_synthetic.py first.")
    return df2_list


def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    Np = args.Np
    Nc = args.Nc

    df2_list = load_stage2(args.data_dir)
    H1, H0, test, data_tuple, predictions = fit_joint(df2_list, Np, Nc, verbose=True)

    print("\nGenerating plots ...")
    bd_fit = plot_stage1_implied(H1, H0, Np)
    plot_stage2(H1, H0, test, data_tuple, Np, Nc)

    print("\nSaving CSV outputs ...")
    save_results(H1, H0, test, predictions, bd_fit)

    print("\n" + "=" * 60)
    print(f"All outputs written to: {os.path.abspath(OUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jointly fit spinodal model from Stage-2 data only")
    parser.add_argument("--data-dir", default=DATA_DIR, help=f"Input data directory (default: {DATA_DIR})")
    parser.add_argument("--Np", type=int, default=TRUE["Np"], help=f"Polymer chain length (default: {TRUE['Np']})")
    parser.add_argument("--Nc", type=int, default=TRUE["Nc"], help=f"Crowder chain length (default: {TRUE['Nc']})")
    main(parser.parse_args())
