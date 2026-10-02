"""Tests for posterior post-processing (rebayfm.core.solution)."""

import numpy as np
import pytest

from rebayfm.core.inversion import build_parameter_grid
from rebayfm.core.radiation import momtens, rmech_fast, sanitize_dip, sanitize_rake
from rebayfm.core.solution import (
    fit_marginal_mix2,
    fit_marginal_single,
    marginal_1d,
    mean_moment_tensor,
    split_posterior_by_strike,
    wrapped_gaussian,
)


def _grids():
    strike_grid = np.deg2rad(np.arange(0, 360, 5.0))
    dip_grid = np.deg2rad(np.vectorize(sanitize_dip)(np.arange(5, 95, 5.0)))
    rake_grid = np.deg2rad(np.vectorize(sanitize_rake)(np.arange(-180, 180, 5.0)))
    return strike_grid, dip_grid, rake_grid


def _synthetic_posterior(
    strike_grid, dip_grid, rake_grid, mu1=(40, 55, -80), mu2=(200, 40, -100), sigma=(15, 8, 12)
):
    """Two-plane Gaussian bump posterior on the 3D grid."""
    s = np.rad2deg(strike_grid)[:, None, None]
    d = np.rad2deg(dip_grid)[None, :, None]
    r = np.rad2deg(rake_grid)[None, None, :]

    def bump(mu):
        return (
            wrapped_gaussian(s, mu[0], sigma[0])
            * np.exp(-0.5 * ((d - mu[1]) / sigma[1]) ** 2)
            * np.exp(-0.5 * (((r - mu[2] + 180) % 360 - 180) / sigma[2]) ** 2)
        )

    post_3d = bump(mu1) + bump(mu2)
    return post_3d / post_3d.sum()


def test_marginals_sum_to_one():
    strike_grid, dip_grid, rake_grid = _grids()
    post_3d = _synthetic_posterior(strike_grid, dip_grid, rake_grid)
    for axis in ("strike", "dip", "rake"):
        assert marginal_1d(post_3d, axis).sum() == pytest.approx(1.0)


def test_marginal_unknown_axis():
    with pytest.raises(ValueError):
        marginal_1d(np.zeros((2, 2, 2)), "bogus")


def test_fit_marginal_mix2_recovers_means():
    strike_grid, dip_grid, rake_grid = _grids()
    post_3d = _synthetic_posterior(strike_grid, dip_grid, rake_grid)
    post_1d = marginal_1d(post_3d, "strike")
    (a1, s1, a2, s2, mu1, mu2, s1_fix, s2_fix) = fit_marginal_mix2(
        post_1d, strike_grid, "strike", 40.0, 200.0
    )
    assert abs(mu1 - 40.0) < 5.0
    assert abs(mu2 - 200.0) < 5.0
    assert 5.0 < s1 < 40.0
    assert s1_fix > 0 and s2_fix > 0


def test_fit_marginal_single_dip():
    strike_grid, dip_grid, rake_grid = _grids()
    post_3d = _synthetic_posterior(
        strike_grid,
        dip_grid,
        rake_grid,
        mu2=(40, 55, -80),  # single bump
    )
    post_1d = marginal_1d(post_3d, "dip")
    a, mu, sigma, sigma_fix = fit_marginal_single(post_1d, dip_grid, "dip", 55.0)
    assert abs(mu - 55.0) < 3.0
    assert 3.0 < sigma < 20.0


def test_split_posterior_by_strike():
    strike_grid, dip_grid, rake_grid = _grids()
    post_3d = _synthetic_posterior(strike_grid, dip_grid, rake_grid)
    p1, p2 = split_posterior_by_strike(post_3d, strike_grid, 1.0, 40.0, 15.0, 1.0, 200.0, 15.0)
    assert p1.sum() == pytest.approx(1.0)
    assert p2.sum() == pytest.approx(1.0)
    # plane posteriors peak near their strikes
    s1 = np.rad2deg(strike_grid[np.argmax(marginal_1d(p1, "strike"))])
    s2 = np.rad2deg(strike_grid[np.argmax(marginal_1d(p2, "strike"))])
    assert abs(s1 - 40.0) < 15.0
    assert abs(s2 - 200.0) < 15.0


def test_mean_moment_tensor_delta_posterior():
    strike_grid, dip_grid, rake_grid = _grids()
    theta_grid = build_parameter_grid(strike_grid, dip_grid, rake_grid)
    post = np.zeros(theta_grid.shape[0])
    # all mass on one mechanism
    target = np.deg2rad([40.0, 55.0, -80.0])
    idx = np.argmin(np.sum((theta_grid - target) ** 2, axis=1))
    post[idx] = 1.0
    mt_mean = mean_moment_tensor(theta_grid, post)
    s, d, r = np.rad2deg(theta_grid[idx])
    expected = momtens(1.0, s % 360, sanitize_dip(d), sanitize_rake(r))
    assert np.allclose(mt_mean, expected, atol=1e-9)
    # and it decomposes back to the same plane
    fps, *_ = rmech_fast(mt_mean)
    diffs = [abs((fps[i][0] - s) % 360) % 360 + abs(fps[i][1] - d) for i in range(2)]
    assert min(d_ % 360 if d_ < 720 else d_ for d_ in diffs) < 2.0
