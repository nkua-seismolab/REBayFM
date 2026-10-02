"""S-wave polarization vectors, ray rotation and predicted polarization angles.

Computes S-wave vector rotation and polarization at the surface.
"""

from __future__ import annotations

import numpy as np

from rebayfm.core.radiation import rpgen_mgs


def angle_diff(a, b):
    """Signed angular difference a - b in radians, wrapped to (-pi, pi]."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def angle_diff_deg(a, b):
    """Signed angular difference a - b in degrees, wrapped to (-180, 180]."""
    return np.degrees(np.arctan2(np.sin(np.radians(a - b)), np.cos(np.radians(a - b))))


def compute_ray_vectors(az, take, gp, gsh, gsv):
    """P, SH, SV and total S vectors for a single ray (angles in radians)."""
    pvec = np.array(
        [
            gp * np.sin(az) * np.sin(take),
            gp * np.cos(az) * np.sin(take),
            gp * np.cos(take),
        ]
    )

    phi = np.pi / 2 - az

    shvec = np.array(
        [
            np.cos(phi - np.pi / 2) * gsh,
            np.sin(phi - np.pi / 2) * gsh,
            0.0,
        ]
    )

    svh = np.cos(take) * gsv
    svvec = np.array(
        [
            np.cos(phi - np.pi) * svh,
            np.sin(phi - np.pi) * svh,
            np.sin(take) * gsv,
        ]
    )

    svec = shvec + svvec

    return pvec, shvec, svvec, svec


def compute_rotation_matrix(pvec, gp, rotcorr):
    """Rodrigues rotation matrix for a single ray; None for vertical rays."""
    pnorm = np.linalg.norm(pvec)
    if pnorm == 0:
        return None

    pvecn = pvec / pnorm

    # horizontal perpendicular axis
    h = np.array([-pvecn[1], pvecn[0], 0.0])
    h_norm = np.linalg.norm(h)

    if h_norm == 0 or np.isnan(h_norm):
        return None  # vertical ray -> no rotation

    h = h / h_norm

    theta = np.radians(np.sign(gp) * rotcorr)

    kx, ky, kz = h
    k = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])

    return np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * np.outer(h, h) + np.sin(theta) * k


def rotate_ray_vectors(pvec, shvec, svvec, svec, gp, rotcorr):
    """Rotate P, SH, SV, S vectors for a single ray."""
    rot = compute_rotation_matrix(pvec, gp, rotcorr)

    if rot is None:
        # vertical ray -> no rotation
        return pvec, shvec, svvec, svec

    return rot @ pvec, rot @ shvec, rot @ svvec, rot @ svec


def rotate_full_grid(azimuth, takeoff, gp, gsh, gsv, rotcorr_all):
    """Rotate the S vector on a full (takeoff, azimuth) grid.

    Returns the rotated S horizontal/vertical components and the angular
    difference between rotated and unrotated horizontal polarization.
    """
    rows, cols = azimuth.shape

    rxp = np.zeros_like(azimuth)
    ryp = np.zeros_like(azimuth)
    rzp = np.zeros_like(azimuth)
    rdf = np.zeros_like(azimuth)

    for i in range(rows):
        for j in range(cols):
            az = azimuth[i, j]
            take = takeoff[i, j]

            pvec, shvec, svvec, svec = compute_ray_vectors(az, take, gp[i, j], gsh[i, j], gsv[i, j])

            _, _, _, s_rot = rotate_ray_vectors(
                pvec, shvec, svvec, svec, gp[i, j], rotcorr_all[i, j]
            )

            rxp[i, j], ryp[i, j], rzp[i, j] = s_rot

            theta0 = 90 - np.degrees(np.arctan2(svec[1], svec[0]))
            theta2 = 90 - np.degrees(np.arctan2(s_rot[1], s_rot[0]))
            rdf[i, j] = np.degrees(np.angle(np.exp(1j * np.radians(theta2 - theta0))))

    return rxp, ryp, rzp, rdf


def rotate_single_ray(az_deg, take_deg, str1, dip1, rak1, model, fdepth):
    """Rotated S-wave vector for a single ray (angles in degrees).

    ``model`` is the (Vp, layer top depth) array; ``fdepth`` the source depth.
    """
    az = np.deg2rad(az_deg)
    take = np.deg2rad(take_deg)

    gp, _, gsh, gsv = rpgen_mgs(str1, dip1, rak1, take_deg, az_deg)

    # IMPORTANT: match the grid's SV sign convention
    gsv = -gsv

    # rotation correction (Snell refraction from source layer to surface)
    ivel = np.where(model[:, 1] < fdepth)[0][-1]
    ainc = np.degrees(np.arcsin(model[0, 0] * np.sin(take) / model[ivel, 0]))
    rotcorr = ainc - take_deg

    pvec, shvec, svvec, svec = compute_ray_vectors(az, take, gp, gsh, gsv)

    p_rot, sh_rot, sv_rot, s_rot = rotate_ray_vectors(pvec, shvec, svvec, svec, gp, rotcorr)

    return s_rot, sh_rot, sv_rot, p_rot


def polarization_direction_from_vector(svec) -> float:
    """Polarization direction (degrees) of a rotated S vector, 0.1 deg rounded."""
    sx, sy = svec[0], svec[1]
    theta_rad = np.arctan2(sy, sx)
    theta_deg = 90 - np.degrees(theta_rad)
    return round(theta_deg, 1)


def predict_s_polarization_angle(strike, dip, rake, takeoff, azimuth, model, fdepth, s_data):
    """Predicted S polarization angle at the surface per station (radians, 0-pi).

    ``strike``, ``dip``, ``rake`` and the ``takeoff``/``azimuth`` arrays are
    in radians (matching the inversion parameter space).
    """
    n = len(s_data.station_ids)
    alpha_mod = np.zeros(n)

    strike_deg = np.rad2deg(strike)
    dip_deg = np.rad2deg(dip)
    rake_deg = np.rad2deg(rake)

    for i in range(n):
        takeoff_deg = np.rad2deg(takeoff[i])
        azimuth_deg = np.rad2deg(azimuth[i])

        if takeoff_deg > 90:
            takeoff_deg = 180 - takeoff_deg
            azimuth_deg = (azimuth_deg + 180) % 360

        s_rot, _, _, _ = rotate_single_ray(
            azimuth_deg, takeoff_deg, strike_deg, dip_deg, rake_deg, model, fdepth
        )
        pol_dir = polarization_direction_from_vector(s_rot)
        alpha_mod[i] = np.deg2rad(pol_dir) % np.pi

    return alpha_mod
