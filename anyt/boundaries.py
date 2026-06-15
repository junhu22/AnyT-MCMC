"""Time-uniform and fixed-sample confidence band widths."""
import numpy as np

def stitched_boundary(n, alpha=0.05):
    """Howard et al. (2021) stitched boundary for time-uniform CDF coverage.

    Parameters
    ----------
    n : int or array-like
        Sample size(s).
    alpha : float
        Confidence level (default 0.05).

    Returns
    -------
    b : float or ndarray
        Boundary width b_n(alpha).
    """
    n = np.asarray(n, dtype=float)
    c_alpha = 0.72 * np.log(5.2 / alpha)
    return np.sqrt(2.0 / n * (np.log(np.log(np.maximum(np.e, 2.0 * n))) + c_alpha))

def dkw_bandwidth(n, alpha=0.05):
    """Fixed-sample DKW bandwidth (Massart 1990).

    Parameters
    ----------
    n : int or array-like
        Sample size(s).
    alpha : float
        Confidence level (default 0.05).

    Returns
    -------
    eps : float or ndarray
        DKW bandwidth epsilon_n(alpha).
    """
    n = np.asarray(n, dtype=float)
    return np.sqrt(np.log(2.0 / alpha) / (2.0 * n))
