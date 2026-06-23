"""LiDAR and odometry aggregation for PQC encoding.

Implements the three-stage pipeline described in Section 2.1 of the paper:

    Stage 1: LiDAR sector aggregation (720 rays -> 12 sectors -> 4 directions
             via trimmed means with 10%/10% trimming).
    Stage 2: Odometry projection to polar (rho, beta) goal-relative
             coordinates in the body frame.
    Stage 3: Normalization and angle encoding into [0, 1]^6 for the PQC.

It also implements the 36-D sector summary used by the LLM module:
    [mean_1..12, var_1..12, null_pct_1..12].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# -----------------------------------------------------------------------------
# Constants from the paper (Section 2.1 and Table on hyperparameters)
# -----------------------------------------------------------------------------
NUM_RAYS: int = 720          # full Velodyne VLP-16 horizontal slice resolution
NUM_SECTORS: int = 12        # 30 degrees per sector
RAYS_PER_SECTOR: int = NUM_RAYS // NUM_SECTORS  # 60 rays per sector
TRIM_FRACTION: float = 0.10  # 10% trimming on each tail per sector

# Normalization scales (Section 2.1, Stage 3)
R_MAX: float = 30.0          # LiDAR maximum range in meters
RHO_MAX: float = 50.0        # largest training goal distance in meters

# Variance trigger threshold for the LLM (Section 2.4)
TAU_VARIANCE: float = 2.0
LLM_QUERY_INTERVAL: int = 10  # one call per 10 control steps


# -----------------------------------------------------------------------------
# Sector summary used by the LLM (Section 2.4, Eq. for Sigma_t)
# -----------------------------------------------------------------------------
@dataclass
class SectorSummary:
    """36-D sector summary: per-sector mean, variance, and null percentage."""

    means: np.ndarray       # shape (12,)
    variances: np.ndarray   # shape (12,)
    null_pcts: np.ndarray   # shape (12,)

    def as_vector(self) -> np.ndarray:
        """Concatenate the three 12-D vectors into a single 36-D vector."""
        return np.concatenate([self.means, self.variances, self.null_pcts])


def compute_sector_summary(scan: np.ndarray) -> SectorSummary:
    """Compute the 36-D sector summary used by the LLM module.

    Parameters
    ----------
    scan : np.ndarray, shape (720,)
        Raw degraded LiDAR scan. NaN entries denote dropped (null) rays.

    Returns
    -------
    SectorSummary
        Per-sector mean, variance, and null percentage.
    """
    assert scan.shape == (NUM_RAYS,), f"expected 720 rays, got {scan.shape}"
    means = np.zeros(NUM_SECTORS, dtype=np.float32)
    variances = np.zeros(NUM_SECTORS, dtype=np.float32)
    null_pcts = np.zeros(NUM_SECTORS, dtype=np.float32)

    for k in range(NUM_SECTORS):
        sector = scan[k * RAYS_PER_SECTOR : (k + 1) * RAYS_PER_SECTOR]
        null_mask = np.isnan(sector)
        n_null = int(null_mask.sum())
        valid = sector[~null_mask]
        null_pcts[k] = 100.0 * n_null / RAYS_PER_SECTOR
        if valid.size == 0:
            # all rays dropped; report zeros to keep dimensions consistent
            means[k] = 0.0
            variances[k] = 0.0
        else:
            means[k] = float(valid.mean())
            variances[k] = float(valid.var())
    return SectorSummary(means=means, variances=variances, null_pcts=null_pcts)


def _trimmed_mean(values: np.ndarray, trim: float = TRIM_FRACTION) -> float:
    """Compute the (1 - 2 * trim) trimmed mean of a 1-D array.

    Falls back to a plain mean when fewer than three values are available.
    """
    if values.size == 0:
        return 0.0
    if values.size < 3:
        return float(values.mean())
    k = int(np.floor(trim * values.size))
    sorted_vals = np.sort(values)
    trimmed = sorted_vals[k : values.size - k] if k > 0 else sorted_vals
    return float(trimmed.mean())


def aggregate_lidar_to_four_directions(scan: np.ndarray) -> np.ndarray:
    """Reduce the 720-ray scan to four directional features.

    Implements Stage 1 of Section 2.1. The 720 rays are split into 12 sectors
    of 30 degrees each. A 10%/10% trimmed mean is computed per sector over
    the surviving (non-NaN) rays. The 12 sector estimates are then collapsed
    into four directional features by averaging adjacent triplets, in the
    order [front, right, back, left].

    Parameters
    ----------
    scan : np.ndarray, shape (720,)
        Raw degraded LiDAR scan with NaN for dropped rays.

    Returns
    -------
    np.ndarray, shape (4,)
        Trimmed mean ranges for [front, right, back, left].
    """
    assert scan.shape == (NUM_RAYS,), f"expected 720 rays, got {scan.shape}"
    sector_trimmed = np.zeros(NUM_SECTORS, dtype=np.float32)
    for k in range(NUM_SECTORS):
        sector = scan[k * RAYS_PER_SECTOR : (k + 1) * RAYS_PER_SECTOR]
        valid = sector[~np.isnan(sector)]
        sector_trimmed[k] = _trimmed_mean(valid)
    # Collapse to four directions by averaging adjacent triplets.
    # Sector layout: 1..3 = front, 4..6 = right, 7..9 = back, 10..12 = left.
    front = sector_trimmed[0:3].mean()
    right = sector_trimmed[3:6].mean()
    back = sector_trimmed[6:9].mean()
    left = sector_trimmed[9:12].mean()
    return np.array([front, right, back, left], dtype=np.float32)


def body_frame_polar_goal(
    pos: np.ndarray, yaw: float, goal: np.ndarray
) -> tuple[float, float]:
    """Project the goal-relative displacement into body-frame polar coordinates.

    Implements Stage 2 of Section 2.1. The world-frame displacement
    (goal - pos) is rotated into the robot body frame using the heading yaw,
    then converted to polar (rho, beta).

    Parameters
    ----------
    pos : np.ndarray, shape (2,)
        Robot position in world coordinates (x, y).
    yaw : float
        Robot heading angle in radians.
    goal : np.ndarray, shape (2,)
        Goal position in world coordinates (x, y).

    Returns
    -------
    rho : float
        Euclidean distance between the robot and the goal.
    beta : float
        Bearing to the goal in the body frame, in [-pi, pi].
    """
    delta = goal - pos
    # rotation by -yaw to enter the body frame
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    dx_body = cos_y * delta[0] - sin_y * delta[1]
    dy_body = sin_y * delta[0] + cos_y * delta[1]
    rho = float(np.sqrt(dx_body ** 2 + dy_body ** 2))
    beta = float(np.arctan2(dy_body, dx_body))
    return rho, beta


def encode_for_pqc(
    scan: np.ndarray, pos: np.ndarray, yaw: float, goal: np.ndarray
) -> np.ndarray:
    """Build the 6-D normalized PQC input from raw observations.

    Implements Stage 3 of Section 2.1: four LiDAR directional features
    normalized by R_MAX, goal distance normalized by RHO_MAX, and bearing
    mapped from [-pi, pi] to [0, 1].

    Parameters
    ----------
    scan : np.ndarray, shape (720,)
        Raw degraded LiDAR scan.
    pos : np.ndarray, shape (2,)
        Robot world-frame position.
    yaw : float
        Robot heading.
    goal : np.ndarray, shape (2,)
        Goal world-frame position.

    Returns
    -------
    np.ndarray, shape (6,)
        Normalized PQC input vector u_tilde in [0, 1]^6.
    """
    four_dir = aggregate_lidar_to_four_directions(scan)
    rho, beta = body_frame_polar_goal(pos, yaw, goal)

    u = np.empty(6, dtype=np.float32)
    u[0:4] = np.clip(four_dir / R_MAX, 0.0, 1.0)
    u[4] = float(np.clip(rho / RHO_MAX, 0.0, 1.0))
    u[5] = float((beta + np.pi) / (2.0 * np.pi))
    return u


def llm_should_query(step: int, sector_summary: SectorSummary) -> bool:
    """Apply the LLM trigger rule of Section 2.4.

    The LLM is queried every LLM_QUERY_INTERVAL steps OR when any per-sector
    variance exceeds TAU_VARIANCE.
    """
    if (step % LLM_QUERY_INTERVAL) == 0:
        return True
    return float(sector_summary.variances.max()) > TAU_VARIANCE
