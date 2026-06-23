"""Sensor degradation operators applied to clean LiDAR scans.

Reproduces the observation model of Section 2 of the paper:

    s_{t,i} = NaN                with probability p_drop = 0.5
            = s_{t,i}^clean + n  otherwise,
    n ~ N(0, sigma^2)            with probability p_noise = 0.5
    n = 0                        otherwise, with sigma = 2.0 m.

Dropout and noise are applied independently per ray and per time step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DegradationConfig:
    """Parameters of the dropout + Gaussian-noise degradation pipeline."""

    p_drop: float = 0.5       # probability a ray is dropped (NaN)
    p_noise: float = 0.5      # probability an additive noise sample is applied
    sigma: float = 2.0        # standard deviation of additive noise (meters)


def apply_degradation(
    clean_scan: np.ndarray,
    cfg: DegradationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a degraded copy of a clean LiDAR scan.

    Parameters
    ----------
    clean_scan : np.ndarray, shape (N,)
        Clean LiDAR ranges in meters. Any non-finite entries in the input
        are preserved as NaN in the output.
    cfg : DegradationConfig, optional
        Degradation parameters. Defaults to the paper values (0.5, 0.5, 2.0).
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray, shape (N,)
        Degraded scan with NaN for dropped rays.
    """
    cfg = cfg or DegradationConfig()
    rng = rng or np.random.default_rng()

    n = clean_scan.shape[0]
    out = clean_scan.astype(np.float32, copy=True)

    # Step 1: dropout channel.
    drop_mask = rng.random(n) < cfg.p_drop
    out[drop_mask] = np.nan

    # Step 2: additive Gaussian noise on the surviving rays.
    surviving = ~drop_mask
    noisy_mask = surviving & (rng.random(n) < cfg.p_noise)
    if noisy_mask.any():
        noise = rng.normal(0.0, cfg.sigma, size=int(noisy_mask.sum()))
        out[noisy_mask] = out[noisy_mask] + noise.astype(np.float32)

    # Negative ranges are unphysical; clip to zero.
    nonan = ~np.isnan(out)
    out[nonan] = np.maximum(out[nonan], 0.0)
    return out
