"""Core anytime certificate computation."""
import numpy as np
from .boundaries import stitched_boundary, dkw_bandwidth

def _oscillation_certificate(sorted_h, n, rho, band_width):
    """Compute oscillation certificate from sorted log-weights.

    Parameters
    ----------
    sorted_h : ndarray, shape (n,)
        Sorted log-weights h_1 <= ... <= h_n.
    n : int
        Number of samples.
    rho : float
        Trimming level.
    band_width : float
        Confidence band width (b_n or eps_n).

    Returns
    -------
    gamma : float
        Certificate value in [0, 1], or 0 if vacuous.
    """
    if band_width >= rho:
        return 0.0
    k_hi = int(np.ceil(n * (1 - rho + band_width)))
    k_lo = int(np.floor(n * (rho - band_width)))
    if k_hi > n or k_lo < 1:
        return 0.0
    C_hat = sorted_h[k_hi - 1] - sorted_h[k_lo - 1]
    gamma = min(1.0, 2.0 / (1.0 + np.exp(C_hat)))
    return gamma

def anytime_certificate(log_weights, rho=0.05, alpha=0.05, n_eval=None):
    """Compute running anytime certificate trajectory.

    Parameters
    ----------
    log_weights : ndarray, shape (N,)
        Log-weight residuals h_i = log pi_tilde(x_i) - log q_T(x_i).
    rho : float
        Trimming level (default 0.05).
    alpha : float
        Confidence level (default 0.05).
    n_eval : ndarray or None
        Sample sizes at which to evaluate. If None, uses 200 geometrically
        spaced points from 100 to N.

    Returns
    -------
    result : dict with keys:
        'n' : ndarray — evaluation points
        'gamma_any' : ndarray — raw anytime certificate at each n
        'gamma_run' : ndarray — running certificate (non-decreasing)
        'gamma_fix' : ndarray — fixed-sample DKW certificate at each n
    """
    N = len(log_weights)
    if n_eval is None:
        n_eval = np.unique(np.geomspace(100, N, 200).astype(int))
    n_eval = n_eval[n_eval <= N]

    gamma_any = np.zeros(len(n_eval))
    gamma_fix = np.zeros(len(n_eval))

    for i, n in enumerate(n_eval):
        sorted_h = np.sort(log_weights[:n])
        b_n = stitched_boundary(n, alpha)
        eps_n = dkw_bandwidth(n, alpha)
        gamma_any[i] = _oscillation_certificate(sorted_h, n, rho, b_n)
        gamma_fix[i] = _oscillation_certificate(sorted_h, n, rho, eps_n)

    gamma_run = np.maximum.accumulate(gamma_any)

    return {
        'n': n_eval,
        'gamma_any': gamma_any,
        'gamma_run': gamma_run,
        'gamma_fix': gamma_fix,
    }

def certified_stop(log_weights, gamma0, rho=0.05, alpha=0.05, check_every=100):
    """Run certified stopping protocol (Algorithm 1).

    Parameters
    ----------
    log_weights : ndarray, shape (N,)
        Log-weight residuals.
    gamma0 : float
        Target certificate level.
    rho : float
        Trimming level (default 0.05).
    alpha : float
        Confidence level (default 0.05).
    check_every : int
        Evaluate certificate every this many samples (default 100).

    Returns
    -------
    tau : int
        Stopping time (or N if budget exhausted).
    gamma_tau : float
        Running certificate at stopping time.
    """
    N = len(log_weights)
    gamma_run = 0.0
    checkpoints = range(check_every, N + 1, check_every)

    for n in checkpoints:
        sorted_h = np.sort(log_weights[:n])
        b_n = stitched_boundary(n, alpha)
        g = _oscillation_certificate(sorted_h, n, rho, b_n)
        gamma_run = max(gamma_run, g)
        if gamma_run >= gamma0:
            return n, gamma_run

    return N, gamma_run
