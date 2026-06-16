# Reproduces Figure 6 from:
#   "Safe sequential certification of transport Monte Carlo" (Hu, 2025)
# (output: fig6_calibration.pdf)

"""
Sequential calibration experiment for anytime certificates.

Compares five protocols under H0 (gamma_0 >= gamma_rho):
  A) Plugin estimator at single n_max (no coverage correction)
  B) Plugin estimator at k checkpoints (naive sequential, no correction)
  C) DKW-corrected certificate at single n_max (valid for single n)
  D) DKW-corrected certificate at k checkpoints (invalid sequential)
  E) Anytime running certificate (valid at any stopping time)

Produces: fig6_calibration.pdf
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import beta as beta_dist
import time

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

HERE = Path(__file__).resolve().parent
CERT_DIR = Path(os.environ.get("CERT_MCMC_DIR", HERE / "checkpoints"))
FIG_DIR = Path(os.environ.get("ANYT_FIG_DIR", HERE / "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Flow architecture (replicated from CerT-MCMC-v2)
# ══════════════════════════════════════════════════════════════

def sn_linear(in_f, out_f):
    return nn.utils.spectral_norm(nn.Linear(in_f, out_f))

class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, mask, scale_clip=0.7):
        super().__init__()
        self.register_buffer('mask', mask)
        self.scale_clip = scale_clip
        self.s_net = nn.Sequential(
            sn_linear(dim, hidden_dim), nn.Tanh(),
            sn_linear(hidden_dim, hidden_dim), nn.Tanh(),
            sn_linear(hidden_dim, dim))
        self.t_net = nn.Sequential(
            sn_linear(dim, hidden_dim), nn.Tanh(),
            sn_linear(hidden_dim, hidden_dim), nn.Tanh(),
            sn_linear(hidden_dim, dim))

    def forward(self, z):
        z_m = z * self.mask
        s = self.s_net(z_m) * (1 - self.mask)
        s = self.scale_clip * torch.tanh(s)
        t = self.t_net(z_m) * (1 - self.mask)
        return z * torch.exp(s) + t, s.sum(dim=-1)

class SNRealNVP(nn.Module):
    def __init__(self, dim=2, n_layers=8, hidden_dim=64, scale_clip=0.7):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % dim] = 1.0
            self.layers.append(CouplingLayer(dim, hidden_dim, mask, scale_clip))

    def forward(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in self.layers:
            x, ld = layer(x)
            log_det += ld
        return x, log_det


class BananaTarget:
    def __init__(self, D=2, sigma1=2.0, sigma2=1.0, b=0.1):
        self.D = D
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.b = b

    def U(self, theta):
        t1, t2 = theta[:, 0], theta[:, 1]
        u = t1**2 / (2 * self.sigma1**2) + (t2 - self.b * t1**2)**2 / (2 * self.sigma2**2)
        if self.D > 2:
            u = u + 0.5 * (theta[:, 2:]**2).sum(dim=-1)
        return u


# ══════════════════════════════════════════════════════════════
# Certificate functions
# ══════════════════════════════════════════════════════════════

def eps_dkw(n, alpha):
    return np.sqrt(np.log(2.0 / alpha) / (2.0 * n))

def b_stitched(n, alpha):
    return np.sqrt((2.0 / n) * (np.log(np.log(max(np.e, 2.0 * n)))
                                + 0.72 * np.log(5.2 / alpha)))

def gamma_osc(sorted_logr, n, eps, rho):
    k_hi = int(np.ceil(n * (1.0 - rho + eps)))
    k_lo = int(np.floor(n * (rho - eps)))
    if k_hi > n or k_lo < 1:
        return 0.0
    c_hat = sorted_logr[k_hi - 1] - sorted_logr[k_lo - 1]
    if c_hat > 500:
        return 0.0
    return min(1.0, 2.0 / (1.0 + np.exp(c_hat)))

def gamma_plugin(sorted_logr, n, rho):
    k_hi = int(np.ceil(n * (1.0 - rho))) - 1
    k_lo = int(np.floor(n * rho))
    if k_hi >= n or k_lo < 0:
        return 0.0
    c_hat = sorted_logr[k_hi] - sorted_logr[k_lo]
    if c_hat > 500:
        return 0.0
    return min(1.0, 2.0 / (1.0 + np.exp(c_hat)))


def sample_logweights(flow, target, n_samples, batch_size=50_000):
    r_all = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            n_batch = min(batch_size, n_samples - i)
            z = torch.randn(n_batch, flow.dim, device=DEVICE)
            theta, log_det = flow(z)
            r = -target.U(theta) + log_det + 0.5 * (z**2).sum(dim=-1)
            r_all.append(r.cpu().numpy())
    return np.concatenate(r_all)


def clopper_pearson(k, n, alpha=0.05):
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


# ══════════════════════════════════════════════════════════════
# Core experiment loop
# ══════════════════════════════════════════════════════════════

def run_experiment(flow, target, gamma_0, R, N_MAX, ALPHA, RHO, k_list,
                   seed_offset=10_000):
    n_eval_any = np.unique(np.geomspace(4500, N_MAX, 300).astype(int))

    false_A = np.zeros(R, dtype=bool)
    false_B = {k: np.zeros(R, dtype=bool) for k in k_list}
    false_C = np.zeros(R, dtype=bool)
    false_D = {k: np.zeros(R, dtype=bool) for k in k_list}
    false_E = np.zeros(R, dtype=bool)

    t0 = time.time()
    for r in range(R):
        if (r + 1) % 100 == 0 or r == 0:
            elapsed = time.time() - t0
            print(f"  Rep {r+1}/{R}  ({elapsed:.0f}s elapsed)")

        torch.manual_seed(seed_offset + r)
        log_r = sample_logweights(flow, target, N_MAX)
        sorted_full = np.sort(log_r)

        # Protocol A: plugin at n_max
        false_A[r] = (gamma_plugin(sorted_full, N_MAX, RHO) >= gamma_0)

        # Protocol B: plugin sequential
        for k in k_list:
            step = N_MAX // k
            checkpoints = np.arange(step, N_MAX + 1, step)
            for cp in checkpoints:
                s = np.sort(log_r[:cp])
                if gamma_plugin(s, cp, RHO) >= gamma_0:
                    false_B[k][r] = True
                    break

        # Protocol C: DKW at n_max
        false_C[r] = (gamma_osc(sorted_full, N_MAX,
                                eps_dkw(N_MAX, ALPHA), RHO) >= gamma_0)

        # Protocol D: DKW sequential
        for k in k_list:
            step = N_MAX // k
            checkpoints = np.arange(step, N_MAX + 1, step)
            for cp in checkpoints:
                s = np.sort(log_r[:cp])
                g_dkw = gamma_osc(s, cp, eps_dkw(cp, ALPHA), RHO)
                if g_dkw >= gamma_0:
                    false_D[k][r] = True
                    break

        # Protocol E: anytime running certificate
        gamma_run = 0.0
        for nn in n_eval_any:
            if nn > N_MAX:
                break
            s = np.sort(log_r[:nn])
            g_any = gamma_osc(s, nn, b_stitched(nn, ALPHA), RHO)
            gamma_run = max(gamma_run, g_any)
            if gamma_run >= gamma_0:
                break
        false_E[r] = (gamma_run >= gamma_0)

    elapsed_total = time.time() - t0
    print(f"  Total time: {elapsed_total:.1f}s")

    return false_A, false_B, false_C, false_D, false_E


def format_rate(arr):
    k = int(np.sum(arr))
    n = len(arr)
    p = k / n
    lo, hi = clopper_pearson(k, n)
    return p, lo, hi


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    D = 20
    ALPHA = 0.05
    RHO = 0.05
    N_MAX = 200_000
    R = 500
    N_ORACLE = 2_000_000

    print("=" * 70)
    print("  Sequential Calibration Experiment")
    print("=" * 70)

    pt_path = CERT_DIR / f"flow_D{D}.pt"
    print(f"\nLoading banana D={D} flow from {pt_path} ...")
    flow = SNRealNVP(dim=D, n_layers=16, hidden_dim=256).to(DEVICE)
    state = torch.load(pt_path, map_location=DEVICE, weights_only=True)
    flow.load_state_dict(state)
    flow.eval()
    target = BananaTarget(D=D)
    print("  Flow loaded.")

    # ── Oracle ──
    print(f"\nComputing oracle gamma (N_oracle = {N_ORACLE:,}) ...")
    torch.manual_seed(999)
    oracle_logr = sample_logweights(flow, target, N_ORACLE)
    sorted_oracle = np.sort(oracle_logr)
    gamma_oracle = gamma_plugin(sorted_oracle, N_ORACLE, RHO)
    n_o = N_ORACLE
    c_oracle = sorted_oracle[int(np.ceil(n_o * (1 - RHO))) - 1] \
             - sorted_oracle[int(np.floor(n_o * RHO))]
    print(f"  C_oracle = {c_oracle:.6f}")
    print(f"  gamma_oracle = {gamma_oracle:.6f}")

    # ════════════════════════════════════════════════
    # Main experiment: gamma_0 = gamma_oracle (boundary)
    # ════════════════════════════════════════════════
    gamma_0 = gamma_oracle
    k_list = [5, 10, 20, 40]
    print(f"\n  gamma_0 = {gamma_0:.6f}  (boundary)")

    print(f"\n--- Main run: R={R}, gamma_0 = gamma_oracle ---")
    false_A, false_B, false_C, false_D, false_E = \
        run_experiment(flow, target, gamma_0, R, N_MAX, ALPHA, RHO, k_list)

    # ── Print results ──
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: gamma_oracle = {gamma_oracle:.4f}, "
          f"gamma_0 = {gamma_0:.4f}, alpha = {ALPHA}")
    print(f"{'=' * 70}")

    rate_A, lo_A, hi_A = format_rate(false_A)
    print(f"\n  A (plugin single-n):   {rate_A:.3f}  "
          f"[{lo_A:.3f}, {hi_A:.3f}]")

    print(f"\n  B (plugin sequential):")
    for k in k_list:
        r_, lo_, hi_ = format_rate(false_B[k])
        print(f"    k={k:2d}: {r_:.3f}  [{lo_:.3f}, {hi_:.3f}]")

    rate_C, lo_C, hi_C = format_rate(false_C)
    print(f"\n  C (DKW single-n):      {rate_C:.3f}  "
          f"[{lo_C:.3f}, {hi_C:.3f}]")

    print(f"\n  D (DKW sequential):")
    for k in k_list:
        r_, lo_, hi_ = format_rate(false_D[k])
        print(f"    k={k:2d}: {r_:.3f}  [{lo_:.3f}, {hi_:.3f}]")

    rate_E, lo_E, hi_E = format_rate(false_E)
    print(f"\n  E (anytime running):   {rate_E:.3f}  "
          f"[{lo_E:.3f}, {hi_E:.3f}]")

    # ── Figure 6 ──
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.0))

    rates_B = [format_rate(false_B[k])[0] for k in k_list]
    rates_D = [format_rate(false_D[k])[0] for k in k_list]

    ax.plot(k_list, rates_B, 's-', color='#D55E00', markersize=5,
            linewidth=1.2, label='Plugin sequential', zorder=5)
    ax.plot(k_list, rates_D, 'D--', color='#0072B2', markersize=4.5,
            linewidth=1.0, label='DKW sequential', zorder=5)

    ax.axhline(rate_A, color='#CC79A7', linestyle='-.',
               linewidth=0.9,
               label=f'Plugin single-$n$ ({rate_A:.0%})', zorder=3)
    ax.axhline(rate_C, color='#56B4E9', linestyle='--',
               linewidth=0.9,
               label=f'DKW single-$n$', zorder=3)
    ax.axhline(rate_E, color='#009E73', linestyle='-',
               linewidth=1.2,
               label='Anytime running cert.', zorder=3)
    ax.axhline(ALPHA, color='gray', linestyle=':', linewidth=0.8,
               label=rf'Nominal $\alpha = {ALPHA}$', zorder=2)

    ax.set_xlabel('Number of checkpoints $k$')
    ax.set_ylabel('Boundary-exceedance rate')
    ax.set_xticks(k_list)
    ymax = max(rates_B) * 1.15
    ax.set_ylim(-0.02, ymax)
    ax.legend(fontsize=6.0, loc='upper right', bbox_to_anchor=(1.0, 0.52))
    ax.tick_params(which='both', direction='in')

    ax.text(0.03, 0.97,
            f'Banana $D={D}$, $\\rho={RHO}$\n'
            f'$\\alpha={ALPHA}$, '
            f'$\\gamma_0 = {gamma_0:.3f}$\n'
            f'$R = {R}$, $n_{{\\max}} = {N_MAX//1000}$k',
            transform=ax.transAxes, fontsize=7, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    fig.tight_layout()
    fig_path = FIG_DIR / "fig6_calibration.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved {fig_path}")
    plt.close(fig)

    # ── LaTeX table ──
    print(f"\n{'=' * 70}")
    print("  LaTeX table (Clopper-Pearson 95% CI):")
    print(f"{'=' * 70}")

    def tex_row(name, checks, arr):
        p, lo, hi = format_rate(arr)
        return (f"  {name} & {checks} & {p:.3f} & "
                f"$[{lo:.3f},\\,{hi:.3f}]$ \\\\")

    print(tex_row("Plugin single-$n$", "1", false_A))
    for k in k_list:
        print(tex_row(f"Plugin sequential", str(k), false_B[k]))
    print(r"  \midrule")
    print(tex_row("DKW single-$n$", "1", false_C))
    for k in k_list:
        print(tex_row(f"DKW sequential", str(k), false_D[k]))
    print(r"  \midrule")
    print(tex_row("Anytime running", "any", false_E))

    # ════════════════════════════════════════════════
    # Robustness: gamma_0 = gamma_oracle + 0.01
    # ════════════════════════════════════════════════
    gamma_0_rob = gamma_oracle + 0.01
    k_rob = [10, 40]
    print(f"\n\n{'=' * 70}")
    print(f"  ROBUSTNESS: gamma_0 = {gamma_0_rob:.4f} "
          f"(oracle + 0.01)")
    print(f"{'=' * 70}")

    false_A_r, false_B_r, false_C_r, false_D_r, false_E_r = \
        run_experiment(flow, target, gamma_0_rob, R, N_MAX, ALPHA, RHO,
                       k_rob, seed_offset=20_000)

    print(f"\n  Results (gamma_0 = {gamma_0_rob:.4f}):")
    r_, lo_, hi_ = format_rate(false_A_r)
    print(f"  Plugin single-n:  {r_:.3f}  [{lo_:.3f}, {hi_:.3f}]")
    for k in k_rob:
        r_, lo_, hi_ = format_rate(false_B_r[k])
        print(f"  Plugin seq k={k}: {r_:.3f}  [{lo_:.3f}, {hi_:.3f}]")
    r_, lo_, hi_ = format_rate(false_C_r)
    print(f"  DKW single-n:     {r_:.3f}  [{lo_:.3f}, {hi_:.3f}]")
    for k in k_rob:
        r_, lo_, hi_ = format_rate(false_D_r[k])
        print(f"  DKW seq k={k}:   {r_:.3f}  [{lo_:.3f}, {hi_:.3f}]")
    r_, lo_, hi_ = format_rate(false_E_r)
    print(f"  Anytime running:  {r_:.3f}  [{lo_:.3f}, {hi_:.3f}]")

    print(f"\n  Robustness LaTeX table:")
    print(tex_row("Plugin single-$n$", "1", false_A_r))
    for k in k_rob:
        print(tex_row("Plugin sequential", str(k), false_B_r[k]))
    print(r"  \midrule")
    print(tex_row("DKW single-$n$", "1", false_C_r))
    for k in k_rob:
        print(tex_row("DKW sequential", str(k), false_D_r[k]))
    print(r"  \midrule")
    print(tex_row("Anytime running", "any", false_E_r))

    print("\nDone.")


if __name__ == "__main__":
    main()
