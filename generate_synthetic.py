"""
generate_synthetic.py
=====================
Generates synthetic phase-separation datasets from known spinodal parameters.

System: FUS-PEG-salt (polymer p, crowder c, solvent s = 1 - phi_p - phi_c)
Free energy: Flory-Huggins + salt-mediated interactions
Spinodal condition: det(Hessian of f) < 0 ⟺ phase-separated

TWO datasets are generated:
  1. Stage 1 — phi_p vs I at phi_c ≈ 0
     Spinodal reduces to H_pp = 0; used to fit chi0, A, B
  2. Stage 2 — (phi_p, phi_c) at fixed I values
     Full 2D spinodal det(H) = 0; used to fit K1, delta

Outputs (in ./synthetic_data/):
  stage1_phiI.csv          — (phi_p, I, label)
  stage2_phipc_I{val}.csv  — (phi_p, phi_c, I, label)  for each I
  spinodal_boundary_1d.csv — (phi_p, I) on the 1D spinodal curve
  spinodal_boundary_2d_I{val}.csv — (phi_p, phi_c) on 2D spinodal at each I
  phase_diagram.png        — overview plots

Usage:
    python generate_synthetic.py

Edit TRUE_PARAMS below to explore different physics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import pandas as pd

# ── Output directory ────────────────────────────────────────────────────────
OUT_DIR = "synthetic_data"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRUE PARAMETERS  (edit these to explore different regimes)
# ══════════════════════════════════════════════════════════════════════════════
TRUE_PARAMS = dict(
    Np    = 600,     # polymer chain length
    Nc    = 227,      # crowder chain length
    chi0  = -12.0,    # bare polymer-solvent chi
    A     = -32.0,    # chi decreases as A·sqrt(I)  (salting-out)
    B     = -18.0,    # chi increases as B·I        (salting-in correction)
    K1    = -5,   # polymer-crowder linear coupling
    delta = -1,    # SALT-CROWDER-POLYMER coupling (the parameter under test)
                    # set to 0.0 to generate null-hypothesis data
)

# ── Dataset sizes ────────────────────────────────────────────────────────────
N_STAGE1      = 3000   # (phi_p, I) points for Stage 1 (each I range)
N_STAGE1_HIGH = 3000   # (phi_p, I) points for Stage 1 high I range (1-3)
N_STAGE2_PER_I = 3000  # (phi_p, phi_c) points per I slice for Stage 2
I_FIXED_VALS  = [0.1, 0.25, 0.4]   # fixed I values for Stage 2 slices
I_STAGE1_RANGES = [(0.001, 0.5), (1.0, 3.0)]  # Two I ranges for Stage 1

# ── Noise ────────────────────────────────────────────────────────────────────
LABEL_NOISE = 0.03     # fraction of labels randomly flipped (measurement noise)
SEED        = 42


# ══════════════════════════════════════════════════════════════════════════════
# Physics
# ══════════════════════════════════════════════════════════════════════════════

def chi_eff(I, chi0, A, B):
    """Salt-dependent effective chi parameter."""
    return chi0 - A * np.sqrt(np.maximum(I, 0.0)) + B * I


def hessian_elements(phi_p, phi_c, params):
    """
    Second derivatives of the Flory-Huggins free energy.
    Returns H_pp, H_cc, H_pc (the three independent Hessian entries).
    """
    Np, Nc, chi0, A, B, K1, delta, I = (
        params["Np"], params["Nc"], params["chi0"], params["A"], params["B"],
        params["K1"], params["delta"], params["I"],
    )
    phi_s = np.clip(1.0 - phi_p - phi_c, 1e-12, 1.0)
    phi_p = np.clip(phi_p, 1e-12, 1.0)
    phi_c = np.clip(phi_c, 1e-12, 1.0)
    ce = chi_eff(I, chi0, A, B)

    H_pp = 1.0 / (Np * phi_p) + 1.0 / phi_s + 2.0 * ce
    H_cc = 1.0 / (Nc * phi_c) + 1.0 / phi_s
    H_pc = 1.0 / phi_s + K1 + delta * I
    return H_pp, H_cc, H_pc


def spinodal_det(phi_p, phi_c, params):
    """det(H) = H_pp·H_cc - H_pc²  < 0 → phase-separated."""
    H_pp, H_cc, H_pc = hessian_elements(phi_p, phi_c, params)
    return H_pp * H_cc - H_pc ** 2


# ══════════════════════════════════════════════════════════════════════════════
# Spinodal boundary tracing
# ══════════════════════════════════════════════════════════════════════════════

def trace_spinodal_1d(params, I_range=(0.001, 0.5), n_I=500, phi_c_fixed=0.001):
    """
    Trace the Stage-1 spinodal: H_pp(phi_p*, I) = 0 at fixed small phi_c.
    Returns (I_array, phi_p_array) on the boundary.
    """
    phi_grid = np.linspace(1e-5, 0.001, 200_000)
    I_vals   = np.linspace(*I_range, n_I)
    bd_I, bd_phi = [], []

    for I in I_vals:
        p = dict(params)
        p["I"] = I
        ce   = chi_eff(I, p["chi0"], p["A"], p["B"])
        H_pp = 1.0 / (p["Np"] * phi_grid) + 1.0 / (1.0 - phi_grid - phi_c_fixed) + 2.0 * ce
        idx  = np.where(np.diff(np.sign(H_pp)))[0]
        if len(idx) > 0:
            bd_I.append(I)
            bd_phi.append(phi_grid[idx[0]])

    return np.array(bd_I), np.array(bd_phi)


def trace_spinodal_1d_full(params, phi_c_fixed=0.001):
    """
    Trace the Stage-1 spinodal across multiple I ranges.
    Returns dict with I_range tuples as keys and (I_array, phi_p_array) as values.
    """
    boundaries = {}
    for I_range in I_STAGE1_RANGES:
        bd_I, bd_phi = trace_spinodal_1d(params, I_range=I_range, n_I=500, phi_c_fixed=phi_c_fixed)
        boundaries[I_range] = (bd_I, bd_phi)
    return boundaries


def trace_spinodal_2d(params, n_phi=500):
    """
    Trace the Stage-2 spinodal: det(H) = 0 in (phi_p, phi_c) space at fixed I.
    Returns (phi_p_array, phi_c_array) on the boundary.
    """
    phi_p_g = np.linspace(1e-5, 0.001, n_phi)
    phi_c_g = np.linspace(0.001, 0.1, n_phi)
    PP, CC  = np.meshgrid(phi_p_g, phi_c_g)
    valid   = (PP + CC) < 0.97

    phi_s = np.where(valid, 1.0 - PP - CC, 1e-12)
    I     = params["I"]
    ce    = chi_eff(I, params["chi0"], params["A"], params["B"])
    Np, Nc, K1, delta = params["Np"], params["Nc"], params["K1"], params["delta"]

    H_pp  = 1.0 / (Np * np.clip(PP, 1e-12, 1)) + 1.0 / np.clip(phi_s, 1e-12, 1) + 2.0 * ce
    H_cc  = 1.0 / (Nc * np.clip(CC, 1e-12, 1)) + 1.0 / np.clip(phi_s, 1e-12, 1)
    H_pc  = 1.0 / np.clip(phi_s, 1e-12, 1) + K1 + delta * I
    SC    = H_pp * H_cc - H_pc ** 2
    SC[~valid] = np.nan

    bd_phi_p, bd_phi_c = [], []
    for i in range(n_phi):
        col   = SC[:, i]
        vmask = ~np.isnan(col)
        if not np.any(vmask):
            continue
        sc_v    = col[vmask]
        phi_c_v = phi_c_g[vmask]
        for idx in np.where(np.diff(np.sign(sc_v)))[0]:
            if phi_p_g[i] + phi_c_v[idx] < 0.94:
                bd_phi_p.append(phi_p_g[i])
                bd_phi_c.append(phi_c_v[idx])

    return np.array(bd_phi_p), np.array(bd_phi_c)


# ══════════════════════════════════════════════════════════════════════════════
# Data generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_stage1(params, n_pts, I_range=(0.001, 0.5), phi_c_fixed=0.001,
                    noise=LABEL_NOISE, seed=SEED):
    """
    Sample (phi_p, I) uniformly; label by H_pp sign; add label noise.
    Returns DataFrame with columns [phi_p, I, label].
    """
    rng   = np.random.default_rng(seed)
    phi_p = rng.uniform(1e-5, 0.001, n_pts)
    I     = rng.uniform(*I_range, n_pts)

    p = dict(params); p["I"] = I
    ce    = chi_eff(I, p["chi0"], p["A"], p["B"])
    H_pp  = 1.0 / (p["Np"] * phi_p) + 1.0 / (1.0 - phi_p - phi_c_fixed) + 2.0 * ce
    label = (H_pp < 0).astype(int)   # 1 = phase-separated

    flip         = rng.random(n_pts) < noise
    label[flip]  = 1 - label[flip]

    return pd.DataFrame({"phi_p": phi_p, "I": I, "label": label.astype(bool)})


def generate_stage1_pooled(params, n_pts_low, n_pts_high, I_range_low=(0.001, 0.5), 
                          I_range_high=(1.0, 3.0), phi_c_fixed=0.001,
                          noise=LABEL_NOISE, seed=SEED):
    """
    Generate Stage-1 data for two I ranges and pool them together.
    Returns DataFrame with columns [phi_p, I, label, I_range].
    """
    # Generate low I range data
    df_low = generate_stage1(params, n_pts_low, I_range_low, phi_c_fixed, noise, seed)
    df_low["I_range"] = "low"
    
    # Generate high I range data with different seed for variety
    df_high = generate_stage1(params, n_pts_high, I_range_high, phi_c_fixed, noise, seed + 100)
    df_high["I_range"] = "high"
    
    # Pool the data
    df_pooled = pd.concat([df_low, df_high], ignore_index=True)
    
    return df_pooled


def generate_stage2(params, n_pts, noise=LABEL_NOISE, seed=SEED):
    """
    Sample (phi_p, phi_c) uniformly in the simplex; label by det(H) sign.
    Returns DataFrame with columns [phi_p, phi_c, I, label].
    """
    I_fixed = params["I"]
    rng     = np.random.default_rng(seed + int(I_fixed * 1000))
    phi_p   = rng.uniform(1e-5, 0.001, n_pts)
    phi_c   = rng.uniform(0.001, 0.1, n_pts)
    valid   = (phi_p + phi_c) < 0.95
    phi_p   = phi_p[valid]; phi_c = phi_c[valid]

    det = spinodal_det(phi_p, phi_c, params)
    label = (det < 0).astype(int)

    flip        = rng.random(len(label)) < noise
    label[flip] = 1 - label[flip]

    return pd.DataFrame({"phi_p": phi_p, "phi_c": phi_c,
                          "I": I_fixed, "label": label.astype(bool)})


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {"ps": "#4C3BCF", "mix": "#2BB5A0", "bd": "#E84545", "bd2": "#FF8C42"}
ALPHA  = 0.45


def plot_all(df1, df2_list, bd1_dict, bd2_list, params):
    """
    Plot all Stage-1 and Stage-2 data.
    bd1_dict: dict with I_range tuples as keys and (I_array, phi_p_array) as values.
    """
    fig = plt.figure(figsize=(22, 10))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.42, wspace=0.38)

    # ── (a) Stage-1 scatter: phi_p vs I (low range) ─────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    df1_low = df1[df1["I_range"] == "low"]
    m1 = df1_low["label"].values.astype(bool)
    ax.scatter(df1_low["phi_p"][~m1], df1_low["I"][~m1], s=3, c=COLORS["mix"], alpha=ALPHA, label="mixed")
    ax.scatter(df1_low["phi_p"][ m1], df1_low["I"][ m1], s=3, c=COLORS["ps"],  alpha=ALPHA, label="phase-sep")
    
    # Plot boundary for low I range
    if (0.001, 0.5) in bd1_dict:
        bd_I, bd_phi = bd1_dict[(0.001, 0.5)]
        if len(bd_I) > 0:
            ax.plot(bd_phi, bd_I, "-", c=COLORS["bd"], lw=2, label="spinodal")
    
    ax.set_xlabel(r"$\phi_p$"); ax.set_ylabel(r"$I$ (mol/L)")
    ax.set_title("(a) Stage 1: $\\phi_p$ vs $I$  (low range: 0–0.5)", fontsize=9)
    ax.legend(fontsize=7, markerscale=3)

    # ── (a2) Stage-1 scatter: phi_p vs I (high range) ────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    df1_high = df1[df1["I_range"] == "high"]
    m1h = df1_high["label"].values.astype(bool)
    ax.scatter(df1_high["phi_p"][~m1h], df1_high["I"][~m1h], s=3, c=COLORS["mix"], alpha=ALPHA, label="mixed")
    ax.scatter(df1_high["phi_p"][ m1h], df1_high["I"][ m1h], s=3, c=COLORS["ps"],  alpha=ALPHA, label="phase-sep")
    
    # Plot boundary for high I range
    if (1.0, 3.0) in bd1_dict:
        bd_I_h, bd_phi_h = bd1_dict[(1.0, 3.0)]
        if len(bd_I_h) > 0:
            ax.plot(bd_phi_h, bd_I_h, "-", c=COLORS["bd"], lw=2, label="spinodal")
    
    ax.set_xlabel(r"$\phi_p$"); ax.set_ylabel(r"$I$ (mol/L)")
    ax.set_title("(a2) Stage 1: $\\phi_p$ vs $I$  (high range: 1–3)", fontsize=9)
    ax.legend(fontsize=7, markerscale=3)

    # ── (b) chi_eff vs I ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    I_plt = np.linspace(0.0, 3.2, 500)
    ax.plot(I_plt, chi_eff(I_plt, params["chi0"], params["A"], params["B"]),
            c=COLORS["bd"], lw=2)
    ax.axhline(0, c="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel(r"$I$ (mol/L)"); ax.set_ylabel(r"$\chi_\mathrm{eff}$")
    ax.set_title(fr"(b) $\chi_\mathrm{{eff}}(I)$  "
                 fr"[$\chi_0={params['chi0']}, A={params['A']}, B={params['B']}$]", fontsize=9)
    ax.set_xlim(0, 3.2)

    # ── (c) Stage-2 scatter panels ───────────────────────────────────────────
    for k, (df2, (bd_pp, bd_pc)) in enumerate(zip(df2_list, bd2_list)):
        ax = fig.add_subplot(gs[0, 3 + k] if k < 2 else gs[1, 2])
        m2 = df2["label"].values.astype(bool)
        ax.scatter(df2["phi_p"][~m2], df2["phi_c"][~m2],
                   s=3, c=COLORS["mix"], alpha=ALPHA, label="mixed")
        ax.scatter(df2["phi_p"][ m2], df2["phi_c"][ m2],
                   s=3, c=COLORS["ps"],  alpha=ALPHA, label="phase-sep")
        if len(bd_pp) > 0:
            order = np.argsort(bd_pp)
            ax.plot(bd_pp[order], bd_pc[order], "o", c=COLORS["bd"],
                    ms=1.5, label="spinodal")
        I_val = df2["I"].iloc[0]
        ax.set_xlabel(r"$\phi_p$"); ax.set_ylabel(r"$\phi_c$")
        ax.set_title(f"(c{k+1}) Stage 2: $I = {I_val}$", fontsize=9)
        if k == 0:
            ax.legend(fontsize=7, markerscale=3)

    # ── (d) Summary info panel ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    ax.axis("off")
    info = (
        r"$\bf{True\ parameters}$" + "\n"
        + "\n".join([
            f"  $N_p={params['Np']},\\ N_c={params['Nc']}$",
            f"  $\\chi_0={params['chi0']},\\ A={params['A']},\\ B={params['B']}$",
            f"  $K_1={params['K1']}$",
            f"  $\\delta={params['delta']}$   "
            r"$\leftarrow$ parameter under hypothesis test",
        ])
        + "\n\n"
        r"$\bf{Free\ energy:}$"
        + "\n"
        r"  $f = \frac{\phi_p}{N_p}\ln\phi_p + \frac{\phi_c}{N_c}\ln\phi_c + \phi_s\ln\phi_s$"
        + "\n"
        r"  $\ + \chi_\mathrm{eff}\phi_p^2 + (K_1\phi_c + \delta I\phi_c)\phi_p$"
        + "\n\n"
        r"$\bf{Spinodal:}$  $\det(H) = H_{pp}H_{cc} - H_{pc}^2 < 0$"
        + "\n"
        + f"  Stage-1 pts: {len(df1)}    Stage-2 pts: {sum(len(d) for d in df2_list)}\n"
        + f"  Label noise: {LABEL_NOISE*100:.0f}%"
    )
    ax.text(0.03, 0.97, info, transform=ax.transAxes, va="top", fontsize=9.5,
            fontfamily="DejaVu Serif", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", fc="#F7F3EE", ec="#CCBBA0", lw=1))

    # ── (e) Stage-2 multi-I boundary overlay ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 3])
    palette = ["#4C3BCF", "#E84545", "#FF8C42"]
    for i, (df2, (bd_pp, bd_pc)) in enumerate(zip(df2_list, bd2_list)):
        I_val = df2["I"].iloc[0]
        if len(bd_pp) > 0:
            order = np.argsort(bd_pp)
            ax.plot(bd_pp[order], bd_pc[order], "-", c=palette[i],
                    lw=1.5, label=f"$I={I_val}$")
    ax.set_xlabel(r"$\phi_p$"); ax.set_ylabel(r"$\phi_c$")
    ax.set_title("(e) Spinodal shift with $I$", fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle("Synthetic Phase Diagram  –  FUS-PEG-salt model", fontsize=13, y=1.01)
    out = os.path.join(OUT_DIR, "phase_diagram.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Synthetic phase diagram generation")
    print("=" * 60)
    print("True parameters:")
    for k, v in TRUE_PARAMS.items():
        print(f"  {k:6s} = {v}")
    print()

    # ---- Stage 1 ------------------------------------------------------------
    print("Generating Stage-1 data  (phi_p vs I, phi_c ~ 0) ...")
    print("  Generating data for two I ranges: (0.001-0.5) and (1.0-3.0) ...")
    df1 = generate_stage1_pooled(TRUE_PARAMS, N_STAGE1, N_STAGE1_HIGH, 
                                   I_STAGE1_RANGES[0], I_STAGE1_RANGES[1])
    print(f"  {len(df1)} pts  |  phase-sep: {df1['label'].sum()}  mixed: {(~df1['label']).sum()}")
    print(f"    - Low I range (0-0.5):  {len(df1[df1['I_range']=='low'])} pts")
    print(f"    - High I range (1-3):   {len(df1[df1['I_range']=='high'])} pts")

    print("  Tracing 1D spinodal boundaries for both I ranges ...")
    bd1_dict = trace_spinodal_1d_full(TRUE_PARAMS)
    for I_range, (bd_I, bd_phi) in bd1_dict.items():
        print(f"    {I_range}: {len(bd_I)} boundary pts")

    # Save Stage-1 data
    df1.to_csv(os.path.join(OUT_DIR, "stage1_phiI.csv"), index=False)
    
    # Save boundaries for both ranges
    for I_range, (bd_I, bd_phi) in bd1_dict.items():
        tag = f"I{I_range[0]}to{I_range[1]}".replace(".", "p")
        pd.DataFrame({"I": bd_I, "phi_p": bd_phi}).to_csv(
            os.path.join(OUT_DIR, f"spinodal_boundary_1d_{tag}.csv"), index=False)
    print(f"  Saved: stage1_phiI.csv  |  spinodal_boundary_1d_*.csv")

    # ---- Stage 2 ------------------------------------------------------------
    df2_list = []; bd2_list = []
    for I_f in I_FIXED_VALS:
        print(f"\nGenerating Stage-2 data  (phi_p vs phi_c, I = {I_f}) ...")
        p = dict(TRUE_PARAMS); p["I"] = I_f
        df2 = generate_stage2(p, N_STAGE2_PER_I)
        print(f"  {len(df2)} pts  |  phase-sep: {df2['label'].sum()}  mixed: {(~df2['label']).sum()}")

        print("  Tracing 2D spinodal boundary ...")
        bd_pp, bd_pc = trace_spinodal_2d(p)
        print(f"  {len(bd_pp)} boundary pts")

        tag = str(I_f).replace(".", "p")
        df2.to_csv(os.path.join(OUT_DIR, f"stage2_phipc_I{tag}.csv"), index=False)
        if len(bd_pp) > 0:
            pd.DataFrame({"phi_p": bd_pp, "phi_c": bd_pc}).to_csv(
                os.path.join(OUT_DIR, f"spinodal_boundary_2d_I{tag}.csv"), index=False)
        print(f"  Saved: stage2_phipc_I{tag}.csv")

        df2_list.append(df2); bd2_list.append((bd_pp, bd_pc))

    # ---- Plots --------------------------------------------------------------
    print("\nGenerating plots ...")
    plot_all(df1, df2_list, bd1_dict, bd2_list, TRUE_PARAMS)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"All outputs written to:  {os.path.abspath(OUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
