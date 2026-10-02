"""Tests for the radiation-pattern / mechanism helpers (rebayfm.core.radiation)."""

import numpy as np

from rebayfm.core.kagan import calc_theta, get_kagan_angle
from rebayfm.core.radiation import (
    mech_both_planes,
    momtens,
    rmech_fast,
    rpgen_mgs,
    rpgen_mgs_p,
    sanitize_dip,
    sanitize_rake,
    second_plane,
)


def test_sanitize_rake_moves_off_extremes():
    assert sanitize_rake(0.0) != 0.0
    assert sanitize_rake(90.0) != 90.0
    assert sanitize_rake(-90.0) != -90.0
    assert sanitize_rake(180.0) != 180.0
    assert sanitize_rake(45.0) == 45.0


def test_sanitize_dip_moves_off_extremes():
    assert 0.0 < sanitize_dip(0.0) < 1.0
    assert 89.0 < sanitize_dip(90.0) < 90.0
    assert sanitize_dip(45.0) == 45.0


def test_momtens_double_couple_traceless():
    mt = momtens(1.0, 30.0, 60.0, -85.0)
    assert np.isclose(np.trace(mt), 0.0, atol=1e-12)
    assert np.allclose(mt, mt.T)


def test_rmech_recovers_planes():
    strike, dip, rake = 40.0, 55.0, -80.0
    mt = momtens(1.0, strike, dip, rake)
    fps, t_ax, p_ax, b_ax, _, _, _ = rmech_fast(mt)
    # one of the two planes must match the input mechanism
    diffs = [
        abs((fps[i][0] - strike + 180) % 360 - 180)
        + abs(fps[i][1] - dip)
        + abs((fps[i][2] - rake + 180) % 360 - 180)
        for i in range(2)
    ]
    assert min(diffs) < 1.0


def test_second_plane_is_conjugate():
    strike, dip, rake = 40.0, 55.0, -80.0
    s2, d2, r2 = second_plane(strike, dip, rake)
    m1 = momtens(1.0, strike, dip, rake)
    m2 = momtens(1.0, s2, d2, r2)
    # conjugate plane represents the same mechanism
    assert calc_theta(m1, m2) < 1.0


def test_mech_both_planes_same_mechanism():
    fps = mech_both_planes(40.0, 55.0, -80.0)
    # plane ordering depends on the eigen decomposition; both must represent
    # the input mechanism
    for i in range(2):
        assert get_kagan_angle(fps[i][0], fps[i][1], fps[i][2], 40.0, 55.0, -80.0) < 1.0


def test_kagan_zero_for_identical():
    assert get_kagan_angle(30, 60, -85, 30, 60, -85) < 1e-3


def test_kagan_known_rotation():
    # pure strike rotation of a vertical strike-slip fault
    angle = get_kagan_angle(0, 90, 0, 30, 90, 0)
    assert abs(angle - 30.0) < 0.5


def test_rpgen_p_sign_flips_across_nodal_plane():
    # vertical strike-slip: P radiation changes sign across azimuth quadrants
    gp_q1 = rpgen_mgs_p(0.0, 89.9, 0.5, 90.0, 45.0)
    gp_q2 = rpgen_mgs_p(0.0, 89.9, 0.5, 90.0, 135.0)
    assert np.sign(gp_q1) != np.sign(gp_q2)


def test_rpgen_components_consistent():
    gp, gs, gsh, gsv = rpgen_mgs(30.0, 60.0, -85.0, 70.0, 120.0)
    assert np.isclose(gs, np.sqrt(gsh**2 + gsv**2))
