# Reproduces Figure 1 from:
#   "Safe sequential certification of transport Monte Carlo" (Hu, 2026)
# (outputs: fig1_oscillation_logr.pdf, fig2_price_logr.pdf)

"""
Full anytime certificate computation on CerT-MCMC-v2 weights.

Uses raw log-weights (h_tilde = log pi_tilde - log q_T) directly.
Oscillation certificate: gamma = 2/(1+exp(C_hat)), two-sided on log-weights.
No exponentiation or self-normalisation needed (C_rho is shift-invariant).

Produces:
  fig1_oscillation_logr.pdf  — 2x3 trajectory grid
  fig2_price_logr.pdf        — price of anytime
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

# ---------- config ----------
ALPHA = 0.05
RHO_LIST = [0.01, 0.05, 0.10]
N_MAX = 200_000
DIMS = [2, 5, 6, 8, 10, 20]
RHO_MAIN = 0.01

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ANYT_DATA_DIR", HERE / "data"))
FIG_DIR = Path(os.environ.get("ANYT_FIG_DIR", HERE / "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

n_eval = np.unique(np.geomspace(100, N_MAX, 300).astype(int))

# ---------- boundaries ----------

def eps_dkw(n, alpha):
    return np.sqrt(np.log(2.0 / alpha) / (2.0 * n))


def b_stitched(n, alpha):
    return np.sqrt((2.0 / n) * (np.log(np.log(max(np.e, 2.0 * n)))
                                + 0.72 * np.log(5.2 / alpha)))


# ---------- oscillation certificate on log-weights ----------

def gamma_osc(sorted_logr_asc, n, eps, rho):
    """CerT-style oscillation certificate.

    k_hi = ceil(n*(1 - rho + eps))   UCB index for upper quantile
    k_lo = floor(n*(rho - eps))      LCB index for lower quantile
    """
    k_hi = int(np.ceil(n * (1.0 - rho + eps)))
    k_lo = int(np.floor(n * (rho - eps)))
    if k_hi > n or k_lo < 1:
        return 0.0
    c_hat = sorted_logr_asc[k_hi - 1] - sorted_logr_asc[k_lo - 1]
    if c_hat > 500:
        return 0.0
    return min(1.0, 2.0 / (1.0 + np.exp(c_hat)))


# ---------- load data ----------

def load_logr(D):
    path = DATA_DIR / f"cert_weights_banana_d{D}.npy"
    log_r = np.load(path)
    n = len(log_r)
    osc = float(np.max(log_r) - np.min(log_r))

    # ESS diagnostic (computed from exp weights, for reporting only)
    lr_shifted = log_r - np.max(log_r)
    r = np.exp(lr_shifted)
    w = r / np.mean(r)
    ess = np.sum(w)**2 / np.sum(w**2)

    print(f"  D={D:2d}: n={n}, log_r: mean={np.mean(log_r):.4f}, "
          f"std={np.std(log_r):.4f}, osc={osc:.2f}, ESS={ess:.0f} ({ess/n*100:.2f}%)")
    return log_r, ess, osc


# ---------- compute certificates ----------

def compute_certs(log_r, rho):
    n_total = len(log_r)
    gf = np.full(len(n_eval), np.nan)
    ga = np.full(len(n_eval), np.nan)

    for idx, n in enumerate(n_eval):
        if n > n_total:
            break
        sorted_lr = np.sort(log_r[:n])

        gf[idx] = gamma_osc(sorted_lr, n, eps_dkw(n, ALPHA), rho)
        ga[idx] = gamma_osc(sorted_lr, n, b_stitched(n, ALPHA), rho)

    return gf, ga


# ---------- main ----------

def main():
    print("Loading log-weights (raw h_tilde, no exp/normalisation):")
    all_data = {}
    for D in DIMS:
        log_r, ess, osc = load_logr(D)
        all_data[D] = {"log_r": log_r, "ess": ess, "osc": osc, "n": len(log_r)}

    print("\nComputing oscillation certificates...")
    results = {}
    for D in DIMS:
        for rho in RHO_LIST:
            gf, ga = compute_certs(all_data[D]["log_r"], rho)
            results[(D, rho)] = {"fixed": gf, "anytime": ga}
        print(f"  D={D} done")

    # ══════════════════════════════════════════════════════════
    # Summary table
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"{'D':>3}  {'rho':>5}  {'g_fix(200k)':>12}  {'g_any(200k)':>12}  "
          f"{'ratio':>7}  {'1st_nv':>8}")
    print(f"{'-'*90}")
    for D in DIMS:
        for rho in RHO_LIST:
            r = results[(D, rho)]
            last = n_eval <= N_MAX
            gf_val = r["fixed"][last][-1]
            ga_val = r["anytime"][last][-1]
            ratio = ga_val / gf_val if gf_val > 0 else np.nan
            nv = np.where(r["anytime"] > 0)[0]
            nv_n = int(n_eval[nv[0]]) if len(nv) > 0 else None
            print(f"{D:3d}  {rho:5.3f}  {gf_val:12.6f}  {ga_val:12.6f}  "
                  f"{ratio:7.4f}  {str(nv_n):>8s}")

    # ── D=20, rho=0.01 detail ──
    print(f"\n{'='*60}")
    print(f"  D=20, rho=0.01  (oscillation on raw log-weights)")
    print(f"{'='*60}")
    r20 = results[(20, 0.01)]
    last = n_eval <= N_MAX
    gf20 = r20["fixed"][last][-1]
    ga20 = r20["anytime"][last][-1]
    ratio20 = ga20 / gf20 if gf20 > 0 else np.nan
    print(f"  fixed  = {gf20:.6f}")
    print(f"  anytime = {ga20:.6f}")
    print(f"  ratio  = {ratio20:.4f}")

    # ══════════════════════════════════════════════════════════
    # Figure 1: trajectory grid
    # ══════════════════════════════════════════════════════════
    print("\nGenerating fig1_oscillation_logr.pdf...")
    rho_colors = {0.01: "#377eb8", 0.05: "#4daf4a", 0.10: "#984ea3"}

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid')

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)

    for i, D in enumerate(DIMS):
        ax = axes.flat[i]
        ess_pct = all_data[D]["ess"] / all_data[D]["n"] * 100
        osc = all_data[D]["osc"]

        for rho in RHO_LIST:
            ga = results[(D, rho)]["anytime"]
            m = ~np.isnan(ga)
            color = rho_colors[rho]
            lw = 1.8 if rho == RHO_MAIN else 0.9
            al = 1.0 if rho == RHO_MAIN else 0.5
            ax.plot(n_eval[m], ga[m], color=color, linewidth=lw,
                    alpha=al, label=f"anytime rho={rho}")

        # fixed-n dots for rho_main
        gf_main = results[(D, RHO_MAIN)]["fixed"]
        mask_f = ~np.isnan(gf_main) & (gf_main > 0)
        if mask_f.sum() > 0:
            step = max(1, mask_f.sum() // 12)
            si = np.where(mask_f)[0][::step]
            ax.scatter(n_eval[si], gf_main[si], color=rho_colors[RHO_MAIN],
                       s=30, zorder=5, edgecolors="white", linewidths=0.5,
                       label=f"fixed rho={RHO_MAIN}")

        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"D={D}  (nESS={ess_pct:.1f}%, osc={osc:.1f})", fontsize=11)
        if i >= 3:
            ax.set_xlabel("n", fontsize=11)
        if i % 3 == 0:
            ax.set_ylabel(r"$\gamma$", fontsize=13)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Anytime Oscillation Certificate (raw log-weights)"
                 f"  [alpha={ALPHA}]", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_oscillation_logr.pdf", dpi=150, bbox_inches="tight")
    print(f"  Saved fig1_oscillation_logr.pdf")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════
    # Figure 2: price of anytime
    # ══════════════════════════════════════════════════════════
    print("Generating fig2_price_logr.pdf...")
    dim_colors = {2: "#1b9e77", 5: "#d95f02", 6: "#7570b3",
                  8: "#e7298a", 10: "#66a61e", 20: "#e6ab02"}

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for D in DIMS:
        gf = results[(D, RHO_MAIN)]["fixed"]
        ga = results[(D, RHO_MAIN)]["anytime"]
        valid = (~np.isnan(gf)) & (~np.isnan(ga)) & (gf > 0) & (ga > 0)
        if valid.sum() == 0:
            continue
        ratio = ga[valid] / gf[valid]
        ax2.plot(n_eval[valid], ratio, color=dim_colors[D], linewidth=1.5,
                 label=f"D={D}")

    ax2.set_xscale("log")
    ax2.set_xlabel("n", fontsize=12)
    ax2.set_ylabel(r"$\gamma_{\mathrm{anytime}} / \gamma_{\mathrm{fixed}}$",
                   fontsize=13)
    ax2.set_title(f"Price of Anytime — Oscillation Certificate"
                  f"  (rho={RHO_MAIN}, alpha={ALPHA})", fontsize=13)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0.4, 1.05)

    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "fig2_price_logr.pdf", dpi=150, bbox_inches="tight")
    print(f"  Saved fig2_price_logr.pdf")
    plt.close(fig2)

    print("\nDone.")


if __name__ == "__main__":
    main()
