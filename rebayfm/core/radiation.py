"""Moment tensor, fault plane and radiation pattern math.

Based on Modern Global Seismology equations.
All angles are in degrees unless noted otherwise.
"""

from __future__ import annotations

import numpy as np


def sanitize_rake(rake_deg: float, eps: float = 5e-1) -> float:
    """Nudge extreme rake values slightly so geometry doesn't degenerate."""
    r = float(rake_deg)

    # Bring into [-180, 180] range
    if abs(r) > 180 + eps:
        r = ((r + 180) % 360) - 180

    # Pure strike-slip 0, 180, -180
    if abs(r) < eps:
        return eps
    if abs(r) > 180 - eps:
        return np.sign(r) * (180 - eps)

    # Pure dip-slip +/-90
    if abs(abs(r) - 90) < eps:
        return np.sign(r) * (90 - eps)

    return r


def sanitize_dip(dip_deg: float, eps: float = 5e-1) -> float:
    """Nudge extreme dip values slightly so geometry doesn't degenerate."""
    r = float(dip_deg)

    if abs(r) < eps:
        return eps

    if abs(r) > 90 - eps:
        return np.sign(r) * (90 - eps)

    return r


def momtens(mo: float, phi: float, delt: float, lam: float) -> np.ndarray:
    """Moment tensor from scalar moment, strike, dip and rake (degrees)."""
    phi = np.deg2rad(phi)
    delt = np.deg2rad(sanitize_dip(delt))
    lam = np.deg2rad(sanitize_rake(lam))

    m11 = -mo * (
        (np.sin(delt) * np.cos(lam) * np.sin(2 * phi))
        + (np.sin(2 * delt) * np.sin(lam) * (np.sin(phi) ** 2))
    )
    m22 = mo * (
        (np.sin(delt) * np.cos(lam) * np.sin(2 * phi))
        - (np.sin(2 * delt) * np.sin(lam) * (np.cos(phi) ** 2))
    )
    m33 = mo * (np.sin(2 * delt) * np.sin(lam))
    m12 = mo * (
        (np.sin(delt) * np.cos(lam) * np.cos(2 * phi))
        + 0.5 * (np.sin(2 * delt) * np.sin(lam) * np.sin(2 * phi))
    )
    m13 = -mo * (
        (np.cos(delt) * np.cos(lam) * np.cos(phi)) + (np.cos(2 * delt) * np.sin(lam) * np.sin(phi))
    )
    m23 = -mo * (
        (np.cos(delt) * np.cos(lam) * np.sin(phi)) - (np.cos(2 * delt) * np.sin(lam) * np.cos(phi))
    )

    return np.array([[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]])


def f_dec_clvd(mr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Decompose a moment tensor into best double-couple, CLVD and F ratio."""
    # eigh: the moment tensor is symmetric, guarantees real eigenvalues
    vals, vecs = np.linalg.eigh(mr)
    trace = np.sum(vals)

    dv = vals - trace / 3.0

    m1_idx = int(np.argmin(np.abs(dv)))  # minor
    m3_idx = int(np.argmax(np.abs(dv)))  # major
    m2_idx = list({0, 1, 2} - {m1_idx, m3_idx})[0]

    m1_val = dv[m1_idx]
    m3_val = dv[m3_idx]

    # CLVD/DC ratio
    f = -m1_val / m3_val

    m1m = np.zeros((3, 3))
    m2m = np.zeros((3, 3))

    # Double-couple part
    m1m[m2_idx, m2_idx] = -m3_val * (1 - 2 * f)
    m1m[m3_idx, m3_idx] = m3_val * (1 - 2 * f)

    # CLVD part
    m2m[m1_idx, m1_idx] = -m3_val * f
    m2m[m2_idx, m2_idx] = -m3_val * f
    m2m[m3_idx, m3_idx] = 2 * m3_val * f

    mdc = vecs @ m1m @ vecs.T
    mclvd = vecs @ m2m @ vecs.T

    return mdc, mclvd, f


def rmech_fast(mt: np.ndarray):
    """Fault plane solutions, principal axes and slip vectors of a DC tensor.

    Returns:
        fps: 2x3 array of [strike, dip, rake] for the two nodal planes
        t_ax, p_ax, b_ax: dicts with eigenvector 'v', plunge 'pl', trend 'tr'
        d1, d2: dicts with slip vectors and SDR
        tp: [trend, plunge] of the B axis
    """
    mone = -1

    mt_norm = mt / np.linalg.norm(mt)

    # eigh: the moment tensor is symmetric, guarantees real eigenvalues
    vals, vecs = np.linalg.eigh(mt_norm)
    rounded = np.round(vals).astype(int)

    t_ax: dict = {}
    p_ax: dict = {}
    b_ax: dict = {}
    for k in range(3):
        v = vecs[:, k]
        pl = np.rad2deg(np.arcsin(v[2]))
        tr = np.rad2deg(np.arctan2(v[1], v[0]))
        if rounded[k] == -1:
            p_ax["v"] = v
            p_ax["pl"] = pl
            p_ax["tr"] = tr
        elif rounded[k] == 0:
            b_ax["v"] = v
            b_ax["pl"] = pl
            b_ax["tr"] = tr
        elif rounded[k] == 1:
            t_ax["v"] = v
            t_ax["pl"] = pl
            t_ax["tr"] = tr
        else:
            raise ValueError("non DC mechanism?")

    # Adjust plunges to be positive
    for axis in (p_ax, t_ax, b_ax):
        if axis["pl"] < 0:
            axis["pl"] = -axis["pl"]
            axis["tr"] = (axis["tr"] + 180) % 360
            axis["v"] = -axis["v"]

    tx, ty, tz = t_ax["v"]
    px, py, pz = p_ax["v"]
    bx, by, bz = b_ax["v"]

    nx = -(tx + px) / np.sqrt(2)
    ny = -(ty + py) / np.sqrt(2)
    nz = -(tz + pz) / np.sqrt(2)
    dx = mone * (tx - px) / np.sqrt(2)
    dy = mone * (ty - py) / np.sqrt(2)
    dz = mone * (tz - pz) / np.sqrt(2)

    d1: dict = {}
    d1["v"] = np.array([dx, dy, dz])
    d1["pl"] = np.rad2deg(np.arcsin(d1["v"][2]))
    d1["tr"] = np.rad2deg(np.arctan2(d1["v"][1], d1["v"][0]))

    if nz < -0.999:
        dip = 0.0
        phi = 0.0
        rak = np.arctan2(-dy, dx)
    else:
        dip = np.arccos(-nz)
        phi = np.arctan2(-nx, ny)
        rak = np.arctan2((-dz / np.sin(dip)), (dx * np.cos(phi)) + (dy * np.sin(phi)))

    tp_dip = np.arccos(bz)
    tp_phi = np.arctan2(bx, -by)
    tp = [np.mod(np.rad2deg(tp_phi), 360), np.rad2deg(tp_dip)]

    fps = np.zeros((2, 3))
    fps[0, :] = [np.mod(np.rad2deg(phi), 360), np.rad2deg(dip), np.rad2deg(rak)]
    d1["sdr"] = fps[0, :]

    # Second nodal plane: swap normal and slip vectors
    nx2, ny2, nz2 = dx, dy, dz
    dx2, dy2, dz2 = nx, ny, nz
    nx, ny, nz = nx2, ny2, nz2
    dx, dy, dz = dx2, dy2, dz2

    if nz > 0:
        nx, ny, nz = -nx, -ny, -nz
        dx, dy, dz = -dx, -dy, -dz

    d2: dict = {}
    d2["v"] = np.array([dx, dy, dz])
    d2["pl"] = np.rad2deg(np.arcsin(d2["v"][2]))
    d2["tr"] = np.rad2deg(np.arctan2(d2["v"][1], d2["v"][0]))

    if nz < -0.999:
        dip = 0.0
        phi = 0.0
        rak = np.arctan2(-dy, dx)
    else:
        dip = np.arccos(-nz)
        phi = np.arctan2(-nx, ny)
        rak = np.arctan2((-dz / np.sin(dip)), (dx * np.cos(phi)) + (dy * np.sin(phi)))

    fps[1, :] = [np.mod(np.rad2deg(phi), 360), np.rad2deg(dip), np.rad2deg(rak)]
    d2["sdr"] = fps[1, :]

    for axis in (t_ax, p_ax, b_ax, d1, d2):
        axis["tr"] = np.mod(axis["tr"], 360)

    return fps, t_ax, p_ax, b_ax, d1, d2, tp


def mech_both_planes(strike: float, dip: float, rake: float) -> np.ndarray:
    """Both fault plane solutions of a mechanism as a 2x3 [strike, dip, rake]."""
    dip = sanitize_dip(dip)
    rake = sanitize_rake(rake)
    mt = momtens(1.0, strike, dip, rake)
    fps = rmech_fast(mt)[0]
    return fps


def second_plane(strike: float, dip: float, rake: float) -> np.ndarray:
    """Conjugate nodal plane of a mechanism as [strike, dip, rake]."""
    dip = sanitize_dip(dip)
    rake = sanitize_rake(rake)
    fps = mech_both_planes(strike, dip, rake)

    # The plane with the larger strike+dip difference is the conjugate
    diffs = []
    for p in fps:
        d = abs(p[0] - strike) % 360
        ds = min(d, 360 - d)
        dd = abs(p[1] - dip)
        diffs.append(ds + dd)

    idx = int(np.argmax(diffs))
    return fps[idx]


def rpgen_mgs(strike, dip, rake, ain, azm):
    """Radiation pattern (Modern Global Seismology equations).

    Args:
        strike, dip, rake: fault geometry angles in degrees.
        ain: takeoff angle in degrees.
        azm: azimuth (clockwise from North) in degrees.

    Returns:
        (gp, gs, gsh, gsv): P amplitude, total S amplitude, SH and SV components.
    """
    strike = np.deg2rad(strike)
    dip = np.deg2rad(dip)
    rake = np.deg2rad(rake)
    ain = np.deg2rad(ain)
    azm = np.deg2rad(azm)

    phi = azm - strike

    gp = (
        np.cos(rake) * np.sin(dip) * (np.sin(ain) ** 2) * np.sin(2 * phi)
        - np.cos(rake) * np.cos(dip) * np.sin(2 * ain) * np.cos(phi)
        + np.sin(rake)
        * np.sin(2 * dip)
        * ((np.cos(ain) ** 2) - (np.sin(ain) ** 2) * (np.sin(phi) ** 2))
        + np.sin(rake) * np.cos(2 * dip) * np.sin(2 * ain) * np.sin(phi)
    )

    gsv = (
        np.sin(rake) * np.cos(2 * dip) * np.cos(2 * ain) * np.sin(phi)
        - np.cos(rake) * np.cos(dip) * np.cos(2 * ain) * np.cos(phi)
        + 0.5 * np.cos(rake) * np.sin(dip) * np.sin(2 * ain) * np.sin(2 * phi)
        - 0.5 * np.sin(rake) * np.sin(2 * dip) * np.sin(2 * ain) * (1 + np.sin(phi) ** 2)
    )

    gsh = (
        np.cos(rake) * np.cos(dip) * np.cos(ain) * np.sin(phi)
        + np.cos(rake) * np.sin(dip) * np.sin(ain) * np.cos(2 * phi)
        + np.sin(rake) * np.cos(2 * dip) * np.cos(ain) * np.cos(phi)
        - 0.5 * np.sin(rake) * np.sin(2 * dip) * np.sin(ain) * np.sin(2 * phi)
    )

    gs = np.sqrt(gsv**2 + gsh**2)

    return gp, gs, gsh, gsv


def rpgen_mgs_p(strike, dip, rake, ain, azm):
    """P-wave radiation pattern amplitude only (all angles in degrees)."""
    return rpgen_mgs(strike, dip, rake, ain, azm)[0]


def rpgen_mgs_s(strike, dip, rake, ain, azm):
    """Total S-wave radiation pattern amplitude only (all angles in degrees)."""
    return rpgen_mgs(strike, dip, rake, ain, azm)[1]
