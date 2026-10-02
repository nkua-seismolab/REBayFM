"""Synthetic-recovery tests for the Bayesian grid search (rebayfm.core.inversion)."""

import numpy as np
import pytest

from rebayfm.core.inversion import (
    PPolarityData,
    SPolarizationData,
    build_parameter_grid,
    dist_calc,
    grid_search_map,
    hpd_region,
    normalize_posterior,
    wrap_angle_diff,
)
from rebayfm.core.kagan import get_kagan_angle
from rebayfm.core.radiation import rpgen_mgs_p, sanitize_dip, sanitize_rake
from rebayfm.core.spol import predict_s_polarization_angle

MODEL = np.array(
    [[4.8, 0.0], [5.2, 4.0], [5.8, 7.2], [6.1, 8.2], [6.3, 10.4], [6.5, 15.0], [7.0, 30.0]]
)
FDEPTH = 8.0

TRUE_STRIKE, TRUE_DIP, TRUE_RAKE = 40.0, 55.0, -80.0

# well-distributed synthetic rays (upgoing, lower hemisphere)
N_STA = 24
RNG = np.random.default_rng(42)
AZIMUTHS = np.linspace(0, 360, N_STA, endpoint=False) + RNG.uniform(-5, 5, N_STA)
TAKEOFFS = RNG.uniform(95, 165, N_STA)


def _p_data():
    p_obs = np.array(
        [
            np.sign(rpgen_mgs_p(TRUE_STRIKE, TRUE_DIP, TRUE_RAKE, take, az))
            for take, az in zip(TAKEOFFS, AZIMUTHS)
        ]
    )
    return PPolarityData(
        station_ids=[f"S{i:02d}" for i in range(N_STA)],
        takeoff_angles=np.deg2rad(TAKEOFFS),
        azimuths=np.deg2rad(AZIMUTHS),
        observed_signs=p_obs,
        mis_pick_probs=np.full(N_STA, 0.10),
    )


def _grids(step_s=15.0, step_d=10.0, step_r=15.0):
    strike_grid = np.deg2rad(np.arange(0, 360, step_s))
    dip_grid = np.deg2rad(np.vectorize(sanitize_dip)(np.arange(5, 90 + step_d, step_d)))
    rake_grid = np.deg2rad(np.vectorize(sanitize_rake)(np.arange(-180, 180, step_r)))
    return strike_grid, dip_grid, rake_grid


def test_dist_calc_known_distance():
    km, deg = dist_calc(38.0, 22.0, 39.0, 22.0)
    assert abs(deg - 1.0) < 1e-6
    assert abs(km - 111.19) < 0.5


def test_wrap_angle_diff():
    assert abs(wrap_angle_diff(np.pi - 0.1) - (-0.1)) < 1e-12
    assert abs(wrap_angle_diff(0.2)) == pytest.approx(0.2)


def test_grid_search_recovers_mechanism_fmp_only():
    strike_grid, dip_grid, rake_grid = _grids()
    theta_best, log_post, theta_grid = grid_search_map(
        strike_grid,
        dip_grid,
        rake_grid,
        MODEL,
        FDEPTH,
        p_data=_p_data(),
        s_data=SPolarizationData([], [], [], [], []),
        fmp_amp=True,
        w_fmp=2.0,
    )
    s, d, r = np.rad2deg(theta_best)
    kagan = get_kagan_angle(s, d, r, TRUE_STRIKE, TRUE_DIP, TRUE_RAKE)
    assert kagan < 25.0


def test_spol_tightens_solution():
    strike_grid, dip_grid, rake_grid = _grids()
    p_data = _p_data()

    # synthetic S polarizations from the true mechanism
    idx = np.arange(0, N_STA, 3)
    s_takeoff = np.deg2rad(TAKEOFFS[idx])
    s_azimuth = np.deg2rad(AZIMUTHS[idx])

    s_data = SPolarizationData(
        station_ids=[f"S{i:02d}" for i in idx],
        takeoff_angles=s_takeoff,
        azimuths=s_azimuth,
        obs_polarization_angle=np.zeros(len(idx)),
        sigma_angle=np.full(len(idx), np.deg2rad(15.0)),
    )
    alpha_true = predict_s_polarization_angle(
        np.deg2rad(TRUE_STRIKE),
        np.deg2rad(TRUE_DIP),
        np.deg2rad(TRUE_RAKE),
        s_takeoff,
        s_azimuth,
        MODEL,
        FDEPTH,
        s_data,
    )
    s_data = SPolarizationData(
        station_ids=s_data.station_ids,
        takeoff_angles=s_takeoff,
        azimuths=s_azimuth,
        obs_polarization_angle=alpha_true,
        sigma_angle=np.full(len(idx), np.deg2rad(15.0)),
    )

    theta_best, log_post, theta_grid = grid_search_map(
        strike_grid,
        dip_grid,
        rake_grid,
        MODEL,
        FDEPTH,
        p_data=p_data,
        s_data=s_data,
        fmp_amp=True,
        w_spol=0.2,
        w_fmp=2.0,
        f_spol=3,
    )
    s, d, r = np.rad2deg(theta_best)
    kagan = get_kagan_angle(s, d, r, TRUE_STRIKE, TRUE_DIP, TRUE_RAKE)
    assert kagan < 20.0

    # posterior machinery
    post = normalize_posterior(log_post)
    assert post.sum() == pytest.approx(1.0)
    hpd = hpd_region(theta_grid, post, level=0.68)
    assert len(hpd) >= 1
    # the MAP solution is inside its own HPD region
    assert any(np.allclose(row, [s, d, r], atol=1e-6) for row in hpd)


def test_build_parameter_grid_shape():
    strike_grid, dip_grid, rake_grid = _grids(30.0, 15.0, 30.0)
    theta_grid = build_parameter_grid(strike_grid, dip_grid, rake_grid)
    assert theta_grid.shape == (len(strike_grid) * len(dip_grid) * len(rake_grid), 3)
