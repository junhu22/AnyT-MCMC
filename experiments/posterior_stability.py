# Reproduces Table (posterior summary stability) in Section 5.5 of:
#   "Safe sequential certification of transport Monte Carlo" (Hu, 2026)
# Method B: reload NF checkpoint, sample (theta, log_r) pairs,
# compute IS-weighted posterior summaries at certified stopping time tau
# vs full budget.

"""
Posterior summary stability at the certified stopping time (AnyT / S&C).

Goal: show that "certificate stopping time tau  <=>  posterior summaries
already stabilised".  We compare importance-weighted (IS) posterior
summaries computed from the first tau samples against those from the full
budget n_max = 200,000.

Scheme B: the saved cert_weights_*.npy files contain only log-weights, so
we reload the trained normalising-flow checkpoints (CerT-MCMC-v2) and
regenerate matched (theta_i, log_r_i) pairs using the *same* seed / batch
order as the original extraction.  The regenerated log-weights are verified
against the saved arrays.

Does NOT modify any files in C:\\CerT-MCMC-v2\\.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths follow the repo convention (see other experiment scripts):
#   ANYT_DATA_DIR  -> processed log-weight arrays (.npy), default ./data
#   CERT_MCMC_DIR  -> trained NF checkpoints (.pt); NOT shipped in this repo,
#                     see README / paper "Code availability". Default ./checkpoints.
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ANYT_DATA_DIR", HERE / "data"))
CERT_DIR = Path(os.environ.get("CERT_MCMC_DIR", HERE / "checkpoints"))

ALPHA = 0.05
N_MAX = 200_000
BATCH = 50_000
SEED = 42

# ══════════════════════════════════════════════════════════════
# Flow architecture (replicated from CerT-MCMC-v2 extraction scripts)
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


# ══════════════════════════════════════════════════════════════
# Targets
# ══════════════════════════════════════════════════════════════

class LogisticRegressionTarget:
    """Synthetic Bayesian logistic regression (from extract_logreg_weights)."""
    def __init__(self, D=20, n_obs=500, tau=2.0, seed=42):
        self.D = D
        self.tau = tau
        rng = np.random.RandomState(seed)
        beta_true = np.zeros(D)
        n_active = min(5, D)
        active_idx = rng.choice(D, n_active, replace=False)
        beta_true[active_idx] = rng.randn(n_active) * 1.5
        X = rng.randn(n_obs, D).astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        self.X = torch.tensor(X, dtype=torch.float32)
        logits = X @ beta_true.astype(np.float32)
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = rng.binomial(1, probs).astype(np.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def U(self, beta):
        X_dev = self.X.to(beta.device)
        y_dev = self.y.to(beta.device)
        logits = beta @ X_dev.T
        log_lik = (y_dev * logits
                   - torch.nn.functional.softplus(logits)).sum(dim=-1)
        log_prior = -0.5 * (beta ** 2).sum(dim=-1) / (self.tau ** 2)
        return -(log_lik + log_prior)

    def to_physical(self, theta):
        # logistic: flow output IS beta
        return theta


W_CALIBRATED = np.array([
    [0.03089318, 0.25500751, 0.31534939, 0.14729265, 0.15159267, 0.09986460],
    [0.05547624, 0.24682654, 0.29048217, 0.14312768, 0.16960660, 0.09448077],
    [0.07054341, 0.17353892, 0.28472797, 0.20401001, 0.18305932, 0.08412037],
    [0.30070137, 0.24373602, 0.19286535, 0.10665094, 0.10360360, 0.05244271],
    [0.38535323, 0.24641020, 0.14570898, 0.08935119, 0.07023028, 0.06294613],
    [0.48516824, 0.24917388, 0.09948227, 0.08017624, 0.05144794, 0.03455144],
], dtype=np.float64)
FREQ_OBS_SAIL = np.array([0.91, 1.08, 1.47, 2.55, 3.16, 3.81], dtype=np.float64)
NOISE_STD_SAIL = np.array([0.025, 0.030, 0.040, 0.060, 0.080, 0.100],
                          dtype=np.float64)


class SailboatTarget:
    """Sailboat building D=6 (from extract_sailboat_weights), vectorised U.

    The vectorised potential is mathematically identical to the original
    per-sample loop, so the regenerated log-weights match the saved array.
    """
    def __init__(self):
        self.D = 6
        self.W = torch.tensor(W_CALIBRATED, dtype=torch.float32)
        self.freq_obs = torch.tensor(FREQ_OBS_SAIL, dtype=torch.float32)
        self.noise_std = torch.tensor(NOISE_STD_SAIL, dtype=torch.float32)
        self.freq_nominal = self.freq_obs.clone()
        self.prior_mean = 1.0
        self.prior_std = 0.35
        self.theta_min = 0.5
        self.theta_max = 2.0
        self.theta_range = self.theta_max - self.theta_min

    def _eta_to_theta(self, eta):
        return self.theta_min + self.theta_range * torch.sigmoid(eta)

    def U(self, eta):
        dev = eta.device
        W = self.W.to(dev)
        freq_obs = self.freq_obs.to(dev)
        noise_std = self.noise_std.to(dev)
        freq_nom = self.freq_nominal.to(dev)

        theta = self._eta_to_theta(eta)              # (B,6) physical
        s = theta @ W.T                              # (B,6)
        s = torch.clamp(s, min=1e-6)
        freq_pred = freq_nom * torch.sqrt(s)         # (B,6)
        residual = (freq_pred - freq_obs) / noise_std
        log_lik = -0.5 * (residual ** 2).sum(dim=-1)
        log_prior = -0.5 * (((theta - self.prior_mean)
                             / self.prior_std) ** 2).sum(dim=-1)
        sig = torch.sigmoid(eta)
        log_jac = (torch.log(sig.clamp(min=1e-10))
                   + torch.log((1 - sig).clamp(min=1e-10))
                   + math.log(self.theta_range)).sum(dim=-1)
        return -(log_lik + log_prior + log_jac)

    def to_physical(self, eta):
        return self._eta_to_theta(eta)


# ══════════════════════════════════════════════════════════════
# Sampling: regenerate matched (theta_physical, log_r) pairs
# ══════════════════════════════════════════════════════════════

def regenerate(flow, target, D, n=N_MAX, batch=BATCH, seed=SEED):
    torch.manual_seed(seed)
    thetas, logrs = [], []
    with torch.no_grad():
        for i in range(0, n, batch):
            nb = min(batch, n - i)
            z = torch.randn(nb, D, device=DEVICE)
            theta_raw, log_det = flow(z)
            log_r = -target.U(theta_raw) + log_det + 0.5 * (z ** 2).sum(dim=-1)
            theta_phys = target.to_physical(theta_raw)
            thetas.append(theta_phys.cpu().numpy())
            logrs.append(log_r.cpu().numpy())
    return np.concatenate(thetas).astype(np.float64), np.concatenate(logrs).astype(np.float64)


# ══════════════════════════════════════════════════════════════
# Certificate machinery (matches anytime_certificate_full.py)
# ══════════════════════════════════════════════════════════════

def b_stitched(n, alpha=ALPHA):
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


def find_tau(log_r, rho, gamma0, alpha=ALPHA):
    """First n at which the anytime running certificate reaches gamma0.

    Running certificate gamma_run(n) = max_{m<=n} gamma_any(m); since it is
    monotone, the first crossing of gamma_any equals the first crossing of
    gamma_run.  Evaluated on a log-spaced grid then refined by a local
    linear scan for an integer tau.
    """
    n_total = len(log_r)
    grid = np.unique(np.geomspace(100, n_total, 600).astype(int))
    run = 0.0
    cross_lo, cross_hi = None, None
    for n in grid:
        g = gamma_osc(np.sort(log_r[:n]), n, b_stitched(n, alpha), rho)
        run = max(run, g)
        if run >= gamma0:
            cross_hi = int(n)
            break
        cross_lo = int(n)
    if cross_hi is None:
        return None, run  # never reached gamma0 within budget
    # local integer refinement between cross_lo and cross_hi
    lo = cross_lo if cross_lo is not None else 100
    for n in range(lo, cross_hi + 1):
        g = gamma_osc(np.sort(log_r[:n]), n, b_stitched(n, alpha), rho)
        if g >= gamma0:
            return int(n), g
    return cross_hi, run


# ══════════════════════════════════════════════════════════════
# IS-weighted posterior summaries
# ══════════════════════════════════════════════════════════════

def is_weights(log_r_slice):
    lw = log_r_slice - np.max(log_r_slice)
    w = np.exp(lw)
    return w / np.sum(w)


def w_mean(theta_col, w):
    return float(np.sum(w * theta_col))


def w_std(theta_col, w):
    m = np.sum(w * theta_col)
    var = np.sum(w * (theta_col - m) ** 2)
    return float(np.sqrt(max(var, 0.0)))


def w_quantile(theta_col, w, q):
    order = np.argsort(theta_col)
    x = theta_col[order]
    cw = np.cumsum(w[order])
    cw = cw / cw[-1]
    return float(np.interp(q, cw, x))


def w_ci_width(theta_col, w, lo=0.025, hi=0.975):
    return w_quantile(theta_col, w, hi) - w_quantile(theta_col, w, lo)


# ══════════════════════════════════════════════════════════════
# Per-target driver
# ══════════════════════════════════════════════════════════════

def run_target(name, flow, target, D, rho, gamma0, summaries, saved_npy):
    print("\n" + "=" * 70)
    print(f"  {name}: D={D}, rho={rho}, gamma0={gamma0}")
    print("=" * 70)

    theta, log_r = regenerate(flow, target, D)

    # verify against saved log-weights
    if saved_npy.exists():
        saved = np.load(saved_npy).astype(np.float64)
        if len(saved) == len(log_r):
            max_abs = float(np.max(np.abs(saved - log_r)))
            corr = float(np.corrcoef(saved, log_r)[0, 1])
            print(f"  verify vs {saved_npy.name}: max|diff|={max_abs:.3e}, "
                  f"corr={corr:.6f}")
            if max_abs > 1e-2:
                print("  [note] regenerated log-weights differ from saved "
                      "(device/RNG); using self-consistent regenerated pairs.")
        else:
            print(f"  [note] length mismatch with saved file.")
    else:
        print(f"  [note] {saved_npy.name} not found; skipping verification.")

    osc = float(np.ptp(log_r))
    w_full = np.exp(log_r - log_r.max())
    w_full = w_full / w_full.mean()
    ess = float(np.sum(w_full) ** 2 / np.sum(w_full ** 2))
    print(f"  n={len(log_r)}, osc={osc:.3f}, "
          f"nESS={ess:.0f} ({ess/len(log_r)*100:.1f}%)")

    tau, g_at_tau = find_tau(log_r, rho, gamma0)
    if tau is None:
        print(f"  [WARN] running certificate never reached gamma0={gamma0} "
              f"(max={g_at_tau:.4f}); skipping.")
        return []
    print(f"  certified stopping time tau = {tau:,}  "
          f"(gamma_run(tau) = {g_at_tau:.4f} >= {gamma0})")

    w_tau = is_weights(log_r[:tau])
    w_max = is_weights(log_r[:N_MAX])

    rows = []
    for label, kind, col in summaries:
        x_tau = theta[:tau, col]
        x_max = theta[:N_MAX, col]
        if kind == "mean":
            e_tau, e_max = w_mean(x_tau, w_tau), w_mean(x_max, w_max)
        elif kind == "std":
            e_tau, e_max = w_std(x_tau, w_tau), w_std(x_max, w_max)
        elif kind == "ciw":
            e_tau, e_max = w_ci_width(x_tau, w_tau), w_ci_width(x_max, w_max)
        else:
            raise ValueError(kind)
        rel = abs(e_tau - e_max) / max(abs(e_max), 1e-12) * 100.0
        rows.append((name, label, e_tau, e_max, rel))
        print(f"    {label:<10s}  tau={e_tau:+.4f}  nmax={e_max:+.4f}  "
              f"rel.diff={rel:6.2f}%")

    return [(tau, gamma0)] + rows if False else (rows, tau, gamma0)


def main():
    print(f"Device: {DEVICE}")
    results = []
    meta = {}

    # ── Logistic regression D=20 ──
    pt = CERT_DIR / "flow_logreg_D20.pt"
    flow = SNRealNVP(dim=20, n_layers=12, hidden_dim=128).to(DEVICE)
    flow.load_state_dict(torch.load(pt, map_location=DEVICE, weights_only=True))
    flow.eval()
    tgt = LogisticRegressionTarget(D=20, n_obs=500, tau=2.0, seed=42)
    summ = [
        (r"$E[\beta_1]$",  "mean", 0),
        (r"$E[\beta_2]$",  "mean", 1),
        (r"$E[\beta_3]$",  "mean", 2),
        (r"$SD[\beta_1]$", "std",  0),
    ]
    rows, tau, g0 = run_target("Logistic", flow, tgt, 20, 0.10, 0.25, summ,
                               DATA_DIR / "cert_weights_logreg_synth20.npy")
    results += rows
    meta["Logistic"] = (tau, g0)

    # ── Sailboat D=6 ──
    pt = CERT_DIR / "flow_sailboat.pt"
    flow = SNRealNVP(dim=6, n_layers=12, hidden_dim=192).to(DEVICE)
    flow.load_state_dict(torch.load(pt, map_location=DEVICE, weights_only=True))
    flow.eval()
    tgt = SailboatTarget()
    summ = [
        (r"$E[\theta_1]$",   "mean", 0),
        (r"$E[\theta_2]$",   "mean", 1),
        (r"$E[\theta_3]$",   "mean", 2),
        (r"CI$_{95}[\theta_1]$", "ciw", 0),
    ]
    rows, tau, g0 = run_target("Sailboat", flow, tgt, 6, 0.10, 0.50, summ,
                               DATA_DIR / "cert_weights_sailboat.npy")
    results += rows
    meta["Sailboat"] = (tau, g0)

    # ══════════════════════════════════════════════════════════
    # Summary + LaTeX table body
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for k, (tau, g0) in meta.items():
        print(f"  {k:<10s} gamma0={g0}  tau={tau:,}")
    max_rel = max(r[4] for r in results)
    print(f"  max relative difference across all summaries: {max_rel:.2f}%")

    print("\n--- LaTeX table body (paste into tab:posterior-stability) ---\n")
    last = None
    lines = []
    for name, label, e_tau, e_max, rel in results:
        tgt_cell = name if name != last else ""
        last = name
        lines.append(f"{tgt_cell} & {label} & {e_tau:.3f} & {e_max:.3f} "
                     f"& {rel:.2f} \\\\")
        if name == "Logistic" and label == r"$SD[\beta_1]$":
            lines.append("\\midrule")
    print("\n".join(lines))
    print(f"\n  [tau_logistic={meta['Logistic'][0]}, "
          f"tau_sailboat={meta['Sailboat'][0]}, max_rel={max_rel:.2f}%]")

    # persist for reproducibility
    np.save(DATA_DIR / "posterior_stability_results.npy",
            np.array(results, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
