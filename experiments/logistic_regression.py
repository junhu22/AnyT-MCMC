# Reproduces Figure 3 from:
#   "Safe sequential certification of transport Monte Carlo" (Hu, 2026)
# (output: fig4_logreg.pdf)

"""
Anytime certificates for logistic regression log-weights.

Produces: fig4_logreg.pdf
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

DATASETS = [
    {"file": "cert_weights_logreg_synth20.npy",
     "name": "Logistic regression", "short": "synth20", "p": 20,
     "n_obs": 500, "color05": "#0072B2", "color10": "#56B4E9"},
    {"file": "cert_weights_logreg_breastcancer.npy",
     "name": "Breast Cancer", "short": "bc30", "p": 30,
     "n_obs": 569, "color05": "#D55E00", "color10": "#E69F00"},
]

# ── Boundaries ──

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


# ── Load and process ──

print("Loading logistic regression weights...")
valid_datasets = []
for ds in DATASETS:
    path = DATA_DIR / ds["file"]
    if not path.exists():
        print(f"  [SKIP] {path}")
        continue
    lr = np.load(path)
    w_shifted = np.exp(lr - np.max(lr))
    w_norm = w_shifted / np.mean(w_shifted)
    ess = np.sum(w_norm)**2 / np.sum(w_norm**2)
    ess_pct = ess / len(lr) * 100
    ds["log_r"] = lr
    ds["n_max"] = len(lr)
    ds["ess"] = ess
    ds["ess_pct"] = ess_pct
    print(f"  {ds['name']} (p={ds['p']}): n={len(lr)}, ESS={ess:.0f} ({ess_pct:.1f}%)")

    if ess_pct < 0.1:
        print(f"    WARNING: ESS too low ({ess_pct:.3f}%), certificates will be vacuous.")
        print(f"    Skipping from figure.")
        continue
    valid_datasets.append(ds)

if not valid_datasets:
    print("No usable datasets. Exiting.")
    raise SystemExit(1)

CHECKPOINTS = np.array([5000, 10000, 20000, 50000, 100000, 200000])

for ds in valid_datasets:
    n_max = ds["n_max"]
    ds["n_eval"] = np.unique(np.geomspace(500, n_max, 200).astype(int))
    ds["results"] = {}
    for rho in RHO_LIST:
        gf, ga = compute_certs(ds["log_r"], rho, ds["n_eval"])
        ga_run = ga.copy()
        valid = ~np.isnan(ga_run)
        ga_run[valid] = np.maximum.accumulate(ga_run[valid])
        gf_cp, ga_cp = compute_certs(ds["log_r"], rho, CHECKPOINTS[CHECKPOINTS <= n_max])
        ds["results"][rho] = {"fixed": gf, "anytime": ga_run,
                              "fixed_cp": gf_cp,
                              "checkpoints": CHECKPOINTS[CHECKPOINTS <= n_max]}

# ── Summary table ──
print(f"\n{'='*80}")
print(f"{'dataset':>20s}  {'p':>3}  {'rho':>5}  {'g_fix':>10}  {'g_any':>10}  "
      f"{'ratio':>7}  {'onset':>8}")
print(f"{'-'*80}")
for ds in valid_datasets:
    for rho in RHO_LIST:
        r = ds["results"][rho]
        gf_val = r["fixed"][~np.isnan(r["fixed"])][-1]
        ga_val = r["anytime"][~np.isnan(r["anytime"])][-1]
        ratio = ga_val / gf_val if gf_val > 0 else np.nan
        nv = np.where(~np.isnan(r["anytime"]) & (r["anytime"] > 0))[0]
        nv_n = int(ds["n_eval"][nv[0]]) if len(nv) > 0 else None
        print(f"{ds['name']:>20s}  {ds['p']:3d}  {rho:5.3f}  {gf_val:10.6f}  "
              f"{ga_val:10.6f}  {ratio:7.4f}  {str(nv_n):>8s}")

# ── Figure ──
n_ds = len(valid_datasets)
fig, axes = plt.subplots(1, n_ds, figsize=(6.0, 2.8), squeeze=False)

for j, ds in enumerate(valid_datasets):
    ax = axes[0, j]
    n_ev = ds["n_eval"]

    for rho in RHO_LIST:
        r = ds["results"][rho]
        ls = "-" if rho == 0.05 else "--"
        lw = 1.2 if rho == 0.05 else 1.0
        al = 1.0 if rho == 0.05 else 0.6
        col = ds["color05"] if rho == 0.05 else ds["color10"]
        m = ~np.isnan(r["anytime"])
        ax.plot(n_ev[m], r["anytime"][m], color=col, linestyle=ls,
                linewidth=lw, alpha=al,
                label=rf"$\rho={rho}$ (run)")
        # fixed-n checkpoints
        cps = r["checkpoints"]
        m_cp = ~np.isnan(r["fixed_cp"])
        mk = "o" if rho == 0.05 else "^"
        ax.scatter(cps[m_cp], r["fixed_cp"][m_cp],
                   color=col, marker=mk, s=18, zorder=5, alpha=al,
                   edgecolors="white", linewidths=0.3,
                   label=rf"$\rho={rho}$ (fix)")

    ax.set_xscale("log")
    ax.set_xlim(500, ds["n_max"] * 1.1)
    ax.set_ylim(0, 1.05)
    ax.text(0.06, 0.92, f"{ds['name']}\n$p={ds['p']}$",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.text(0.94, 0.08, f"nESS={ds['ess_pct']:.0f}%",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.4))
    ax.set_xlabel("Sample size $n$")
    if j == 0:
        ax.set_ylabel(r"$\gamma$")
    ax.tick_params(which="both", direction="in")
    ax.legend(fontsize=6, loc="upper right", handlelength=1.5)

fig.tight_layout(w_pad=0.5)
fig.savefig(FIG_DIR / "fig4_logreg.pdf", dpi=300, bbox_inches="tight")
print(f"\nSaved {FIG_DIR / 'fig4_logreg.pdf'}")
plt.close(fig)
print("Done.")
