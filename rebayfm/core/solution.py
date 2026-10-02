"""Posterior post-processing: marginals, Gaussian fits, plane separation.

Grids are in radians; fitted parameters (mu, sigma) are in degrees.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from rebayfm.core.radiation import momtens, sanitize_dip, sanitize_rake

# ---------------------------------------------------------------------------
# Wrapped/plain Gaussians per axis (3-term approximations, all in degrees)
# ---------------------------------------------------------------------------


def wrapped_gaussian(phi, mu, sigma):
    """Wrapped Gaussian on [0,360) using 3-term approximation (strike)."""
    return (
        np.exp(-0.5 * ((phi - mu) / sigma) ** 2)
        + np.exp(-0.5 * ((phi - mu + 360) / sigma) ** 2)
        + np.exp(-0.5 * ((phi - mu - 360) / sigma) ** 2)
    )


def circ_diff_rake(r, mu):
    """Circular difference for rake in [-180,180] degrees."""
    return (r - mu + 180) % 360 - 180


def wrapped_gaussian_rake(r, mu, sigma):
    """Wrapped Gaussian for rake in [-180,180] using 3-term approximation."""
    return (
        np.exp(-0.5 * (circ_diff_rake(r, mu) / sigma) ** 2)
        + np.exp(-0.5 * (circ_diff_rake(r + 360, mu) / sigma) ** 2)
        + np.exp(-0.5 * (circ_diff_rake(r - 360, mu) / sigma) ** 2)
    )


def gaussian_dip(x, mu, sigma):
    """Plain (unwrapped) Gaussian used for the dip axis."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


_AXIS_GAUSSIAN = {
    "strike": wrapped_gaussian,
    "dip": gaussian_dip,
    "rake": wrapped_gaussian_rake,
}

# per-axis curve_fit setup: (p0_sigma, sigma_max, mu_lo, mu_hi)
_AXIS_FIT = {
    "strike": (20.0, 180.0, 0.0, 360.0),
    "dip": (10.0, 90.0, 0.0, 90.0),
    "rake": (20.0, 180.0, -180.0, 180.0),
}


# ---------------------------------------------------------------------------
# 1D marginals from the 3D posterior
# ---------------------------------------------------------------------------


def marginal_1d(post_3d: np.ndarray, axis: str) -> np.ndarray:
    """1D posterior marginal for 'strike', 'dip' or 'rake'."""
    if axis == "strike":
        return np.sum(np.sum(post_3d, axis=2), axis=1)
    if axis == "dip":
        return np.sum(np.sum(post_3d, axis=0), axis=1)
    if axis == "rake":
        return np.sum(np.sum(post_3d, axis=1), axis=0)
    raise ValueError(f"Unknown axis {axis!r}")


# ---------------------------------------------------------------------------
# Gaussian mixture fits (as in plot_single_marginal_mix2 / _one, showfig=False)
# ---------------------------------------------------------------------------


def fit_marginal_mix2(post_1d, grid_rad, axis, mu1_fixed, mu2_fixed):
    """Two-component Gaussian fit of a 1D marginal, fixed and free means.

    Returns (a1_free, sigma1_free, a2_free, sigma2_free, mu1_free, mu2_free,
    sigma1_fix, sigma2_fix), everything in degrees.
    """
    gauss = _AXIS_GAUSSIAN[axis]
    p0_sigma, sigma_max, mu_lo, mu_hi = _AXIS_FIT[axis]
    x_deg = np.rad2deg(grid_rad)

    def fit_func_fixmu(x, a1, sigma1, a2, sigma2):
        return a1 * gauss(x, mu1_fixed, sigma1) + a2 * gauss(x, mu2_fixed, sigma2)

    params_fix, _ = curve_fit(
        fit_func_fixmu,
        x_deg,
        post_1d,
        p0=[1.0, p0_sigma, 1.0, p0_sigma],
        bounds=([0, 0, 0, 0], [1, sigma_max, 1, sigma_max]),
        maxfev=20000,
    )
    _, sigma1_fix, _, sigma2_fix = params_fix

    def fit_func_freemu(x, a1, sigma1, a2, sigma2, mu1, mu2):
        return a1 * gauss(x, mu1, sigma1) + a2 * gauss(x, mu2, sigma2)

    params_free, _ = curve_fit(
        fit_func_freemu,
        x_deg,
        post_1d,
        p0=[1.0, p0_sigma, 1.0, p0_sigma, mu1_fixed, mu2_fixed],
        bounds=(
            [0, 0, 0, 0, mu_lo, mu_lo],
            [1, sigma_max, 1, sigma_max, mu_hi, mu_hi],
        ),
        maxfev=20000,
    )
    a1_free, sigma1_free, a2_free, sigma2_free, mu1_free, mu2_free = params_free

    return (
        a1_free,
        sigma1_free,
        a2_free,
        sigma2_free,
        mu1_free,
        mu2_free,
        sigma1_fix,
        sigma2_fix,
    )


def fit_marginal_single(post_1d, grid_rad, axis, mu_fixed):
    """Single Gaussian fit of a 1D marginal, fixed and free mean.

    Returns (a_free, mu_free, sigma_free, sigma_fix) in degrees.
    """
    gauss = _AXIS_GAUSSIAN[axis]
    p0_sigma, sigma_max, mu_lo, mu_hi = _AXIS_FIT[axis]
    x_deg = np.rad2deg(grid_rad)

    def fit_func_fixmu(x, a1, sigma1):
        return a1 * gauss(x, mu_fixed, sigma1)

    params_fix, _ = curve_fit(
        fit_func_fixmu,
        x_deg,
        post_1d,
        p0=[1.0, p0_sigma],
        bounds=([0, 0], [1, sigma_max]),
        maxfev=20000,
    )
    _, sigma_fix = params_fix

    def fit_func_freemu(x, a1, sigma1, mu1):
        return a1 * gauss(x, mu1, sigma1)

    params_free, _ = curve_fit(
        fit_func_freemu,
        x_deg,
        post_1d,
        p0=[1.0, p0_sigma, mu_fixed],
        bounds=([0, 0, mu_lo], [1, sigma_max, mu_hi]),
        maxfev=20000,
    )
    a_free, sigma_free, mu_free = params_free

    return a_free, mu_free, sigma_free, sigma_fix


# ---------------------------------------------------------------------------
# Plane separation and mean mechanism
# ---------------------------------------------------------------------------


def split_posterior_by_strike(post_3d, strike_grid, a1, mu1, sigma1, a2, mu2, sigma2):
    """Split the 3D posterior into per-plane posteriors using the fitted
    strike wrapped-Gaussian mixture (all fit parameters in degrees)."""
    strike_deg = np.rad2deg(strike_grid)

    g1 = a1 * wrapped_gaussian(strike_deg, mu1, sigma1)
    g2 = a2 * wrapped_gaussian(strike_deg, mu2, sigma2)

    g1 /= np.sum(g1)
    g2 /= np.sum(g2)
    g1_3d = g1[:, np.newaxis, np.newaxis]  # shape (Ns, 1, 1)
    g2_3d = g2[:, np.newaxis, np.newaxis]

    post_3d_plane1 = post_3d * g1_3d
    post_3d_plane2 = post_3d * g2_3d
    post_3d_plane1 /= np.sum(post_3d_plane1)
    post_3d_plane2 /= np.sum(post_3d_plane2)

    return post_3d_plane1, post_3d_plane2


def mean_moment_tensor(theta_grid, post):
    """Probability-weighted mean 3x3 moment tensor (NED convention)."""
    mt_mean = np.zeros((3, 3), dtype=float)

    for (stk, dp, rk), p in zip(theta_grid, post):
        stk_deg = np.rad2deg(stk) % 360
        dp_deg = sanitize_dip(np.rad2deg(dp))
        rk_deg = sanitize_rake(np.rad2deg(rk))

        mt = momtens(1.0, stk_deg, dp_deg, rk_deg)
        mt_mean += p * mt

    return mt_mean
