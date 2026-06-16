# AnyT-MCMC: Safe Sequential Certification of Transport Monte Carlo

Anytime-valid minorization certificates for independence Metropolis–Hastings 
with normalizing-flow proposals.

## Overview

This package implements the anytime oscillation certificate from:

> **Safe sequential certification of transport Monte Carlo**  
> Jun Hu (2026)  
> Submitted to *Bayesian Analysis*

The running certificate $\underline{\gamma}^{\mathrm{run}}(n)$ is valid at every 
sample size simultaneously and at any data-dependent stopping time.

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```python
import numpy as np
from anyt.certificate import anytime_certificate, certified_stop

# Load or compute log-weights: h_i = log pi_tilde(x_i) - log q_T(x_i)
log_weights = np.load("your_log_weights.npy")

# Compute running certificate trajectory
result = anytime_certificate(log_weights, rho=0.05, alpha=0.05)
print(f"gamma_run at n={len(log_weights)}: {result['gamma_run'][-1]:.4f}")

# Or: certified stopping
tau, gamma_tau = certified_stop(log_weights, gamma0=0.4, rho=0.05, alpha=0.05)
print(f"Stopped at n={tau} with gamma_run={gamma_tau:.4f}")
```

## Experiments

The `experiments/` directory contains scripts reproducing all figures and tables 
in the paper:

| Script | Description |
|---|---|
| `banana_trajectories.py` | Figure 1: certificate trajectories, D=2–20 |
| `logistic_regression.py` | Figure 3: Bayesian logistic regression |
| `sailboat_shm.py` | Figure 4: real-data sailboat building |
| `calibration_experiment.py` | Figure 6: sequential calibration (R=500) |

## Citation

```bibtex
@article{hu2025anyt,
  author = {Hu, Jun},
  title = {Safe sequential certification of transport {M}onte {C}arlo},
  year = {2026},
  note = {Submitted to Bayesian Analysis}
}
```

## License

MIT License. See [LICENSE](LICENSE).
