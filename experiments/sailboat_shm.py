# Reproduces Figure 4 from:
#   "Safe sequential certification of transport Monte Carlo" (Hu, 2025)
# (output: fig5_sailboat.pdf)

"""
Anytime certificates for sailboat building log-weights.

Produces: fig5_sailboat.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "text.usetex": False,
    "mathtext.fontset": "cm",
})

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ANYT_DATA_DIR", HERE / "data"))
FIG_DIR = Path(os.environ.get("ANYT_FIG_DIR", HERE / "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
RHO_LIST = [0.05, 0.10]


def eps_dkw(n, alpha):
    return np.sqrt(np.log(2.0 / alpha) / (2.0 * n))

def b_stitched(n, alpha):
    return np.sqrt((2.0 / n) * (np.log(np.log(max(np.e, 2.0 * n)))
                                + 0.72 * np.log(5.2 / alpha)))

def gamma_osc(sorted_logr_asc, n, eps, rho):
    k_hi = int(np.ceil(n * (1.0 - rho + eps)))
    k_lo = int(np.floor(n * (rho - eps)))
    if k_hi > n or k_lo < 1:
        return 0.0
    c_hat = sorted_logr_asc[k_hi - 1] - sorted_logr_asc[k_lo - 1]
    if c_hat > 500:
        return 0.0
    return min(1.0, 2.0 / (1.0 + np.exp(c_hat)))

def compute_certs(log_r, rho, n_pts):
    n_total = len(log_r)
    gf = np.full(len(n_pts), np.nan)
    ga = np.full(len(n_pts), np.nan)
    for idx, n in enumerate(n_pts):
        if n > n_total:
            break
        sorted_lr = np.sort(log_r[:n])
        gf[idx] = gamma_osc(sorted_lr, n, eps_dkw(n, ALPHA), rho)
        ga[idx] = gamma_osc(sorted_lr, n, b_stitched(n, ALPHA), rho)
    return gf, ga


# ── Load ──
print("Loading sailboat weights...")
log_r = np.load(DATA_DIR / "cert_weights_sailboat.npy")
n_max = len(log_r)

w_shifted = np.exp(log_r - np.max(log_r))
w_norm = w_shifted / np.mean(w_shifted)
ess = np.sum(w_norm)**2 / np.sum(w_norm**2)
ess_pct = ess / n_max * 100
print(f"  Sailboat (D=6): n={n_max:,}, ESS={ess:.0f} ({ess_pct:.1f}%)")

n_eval = np.unique(np.geomspace(500, n_max, 300).astype(int))
CHECKPOINTS = np.array([5000, 10000, 20000, 50000, 100000, 200000])
CHECKPOINTS = CHECKPOINTS[CHECKPOINTS <= n_max]

results = {}
for rho in RHO_LIST:
    gf, ga = compute_certs(log_r, rho, n_eval)
    ga_run = ga.copy()
    valid = ~np.isnan(ga_run)
    ga_run[valid] = np.maximum.accumulate(ga_run[valid])
    gf_cp, _ = compute_certs(log_r, rho, CHECKPOINTS)
    results[rho] = {"fixed": gf, "anytime": ga_run,
                    "fixed_cp": gf_cp, "checkpoints": CHECKPOINTS}

# ── Summary ──
print(f"\n{'='*70}")
print(f"{'rho':>5}  {'g_fix':>10}  {'g_run':>10}  {'ratio':>7}  {'onset':>8}")
print(f"{'-'*70}")
for rho in RHO_LIST:
    r = results[rho]
    gf_val = r["fixed"][~np.isnan(r["fixed"])][-1]
    ga_val = r["anytime"][~np.isnan(r["anytime"])][-1]
    ratio = ga_val / gf_val if gf_val > 0 else np.nan
    nv = np.where(~np.isnan(r["anytime"]) & (r["anytime"] > 0))[0]
    nv_n = int(n_eval[nv[0]]) if len(nv) > 0 else None
    print(f"{rho:5.3f}  {gf_val:10.6f}  {ga_val:10.6f}  {ratio:7.4f}  {str(nv_n):>8s}")

# ── Figure ──
fig, ax = plt.subplots(1, 1, figsize=(6.0, 2.8))
colors = {0.05: "#0072B2", 0.10: "#56B4E9"}

for rho in RHO_LIST:
    r = results[rho]
    ls = "-" if rho == 0.05 else "--"
    lw = 1.2 if rho == 0.05 else 1.0
    al = 1.0 if rho == 0.05 else 0.6
    col = colors[rho]
    m = ~np.isnan(r["anytime"])
    ax.plot(n_eval[m], r["anytime"][m], color=col, linestyle=ls,
            linewidth=lw, alpha=al,
            label=rf"$\rho={rho:.2f}$ ($\gamma_{{\mathrm{{run}}}}$)")
    cps = r["checkpoints"]
    m_cp = ~np.isnan(r["fixed_cp"])
    mk = "o" if rho == 0.05 else "^"
    ax.scatter(cps[m_cp], r["fixed_cp"][m_cp],
               color=col, marker=mk, s=18, zorder=5, alpha=al,
               edgecolors="white", linewidths=0.3,
               label=rf"$\rho={rho:.2f}$ ($\gamma_{{\mathrm{{fix}}}}$)")

ax.set_xscale("log")
ax.set_xlim(500, n_max * 1.1)
ax.set_ylim(0, 1.05)
ax.text(0.06, 0.92, f"Sailboat building\n$D=6$, ESS={ess_pct:.0f}%",
        transform=ax.transAxes, fontsize=8, va="top")
ax.set_xlabel("Sample size $n$")
ax.set_ylabel(r"$\gamma$")
ax.tick_params(which="both", direction="in")
ax.legend(fontsize=6, loc="lower right", handlelength=1.5)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_sailboat.pdf", dpi=300, bbox_inches="tight")
print(f"\nSaved {FIG_DIR / 'fig5_sailboat.pdf'}")
plt.close(fig)
print("Done.")
