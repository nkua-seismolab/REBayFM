"""Tests for S-polarization geometry helpers (rebayfm.core.spol)."""

import numpy as np
import pytest

from rebayfm.core.spol import (
    angle_diff,
    angle_diff_deg,
    compute_ray_vectors,
    compute_rotation_matrix,
    polarization_direction_from_vector,
    rotate_ray_vectors,
    rotate_single_ray,
)

MODEL = np.array(
    [[4.8, 0.0], [5.2, 4.0], [5.8, 7.2], [6.1, 8.2], [6.3, 10.4], [6.5, 15.0], [7.0, 30.0]]
)


def test_angle_diff_wraps_pi():
    assert angle_diff(np.pi - 0.05, -np.pi + 0.05) == pytest.approx(-0.1, abs=1e-12)
    assert angle_diff(0.3, 0.1) == pytest.approx(0.2)


def test_angle_diff_deg_wraps_180():
    assert angle_diff_deg(179.0, -179.0) == pytest.approx(-2.0)
    assert angle_diff_deg(10.0, 350.0) == pytest.approx(20.0)


def test_compute_ray_vectors_s_composition():
    az, take = np.deg2rad(30.0), np.deg2rad(140.0)
    pvec, shvec, svvec, svec = compute_ray_vectors(az, take, gp=0.5, gsh=0.3, gsv=-0.4)
    assert np.allclose(svec, shvec + svvec)
    # P vector points along the ray with amplitude |gp|
    assert np.linalg.norm(pvec) == pytest.approx(0.5)
    # SH is horizontal
    assert shvec[2] == 0.0


def test_rotation_matrix_orthonormal():
    az, take = np.deg2rad(30.0), np.deg2rad(140.0)
    pvec, *_ = compute_ray_vectors(az, take, gp=0.5, gsh=0.3, gsv=-0.4)
    rot = compute_rotation_matrix(pvec, gp=0.5, rotcorr=20.0)
    assert rot is not None
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rot), 1.0)


def test_rotation_vertical_ray_is_identity():
    # vertical ray: pvec has no horizontal component
    pvec = np.array([0.0, 0.0, -0.5])
    assert compute_rotation_matrix(pvec, gp=0.5, rotcorr=20.0) is None
    p, sh, sv, s = rotate_ray_vectors(
        pvec, np.zeros(3), np.zeros(3), np.zeros(3), gp=0.5, rotcorr=20.0
    )
    assert np.allclose(p, pvec)


def test_rotation_preserves_norm():
    az, take = np.deg2rad(60.0), np.deg2rad(120.0)
    pvec, shvec, svvec, svec = compute_ray_vectors(az, take, gp=0.4, gsh=0.2, gsv=0.6)
    _, _, _, s_rot = rotate_ray_vectors(pvec, shvec, svvec, svec, gp=0.4, rotcorr=25.0)
    assert np.linalg.norm(s_rot) == pytest.approx(np.linalg.norm(svec))


def test_rotate_single_ray_returns_vectors():
    s_rot, sh_rot, sv_rot, p_rot = rotate_single_ray(
        az_deg=60.0, take_deg=140.0, str1=40.0, dip1=55.0, rak1=-80.0, model=MODEL, fdepth=8.0
    )
    for vec in (s_rot, sh_rot, sv_rot, p_rot):
        assert np.all(np.isfinite(vec))
    assert np.allclose(s_rot, sh_rot + sv_rot)


def test_polarization_direction_from_vector():
    # S vector pointing north -> polarization 0 deg
    assert polarization_direction_from_vector(np.array([0.0, 1.0, 0.0])) == 0.0
    # pointing east -> 90 deg
    assert polarization_direction_from_vector(np.array([1.0, 0.0, 0.0])) == 90.0
