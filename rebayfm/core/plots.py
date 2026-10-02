"""Figure generation: HPD beachball and posterior marginals.

Includes focal sphere projections, quadrant filling and marginal PDF layouts.
Matplotlib runs headless (Agg); figures are only ever saved, never shown.
HASH reference-solution overlays are not included.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.path import Path  # noqa: E402

from rebayfm.core.kagan import calc_theta  # noqa: E402
from rebayfm.core.radiation import (  # noqa: E402
    mech_both_planes,
    momtens,
    rmech_fast,
    rpgen_mgs,
    rpgen_mgs_p,
    sanitize_dip,
    sanitize_rake,
)
from rebayfm.core.spol import (  # noqa: E402
    angle_diff_deg,
    compute_ray_vectors,
    polarization_direction_from_vector,
    rotate_full_grid,
    rotate_single_ray,
)

# ---------------------------------------------------------------------------
# Focal sphere projection helpers
# ---------------------------------------------------------------------------


def m_pltsym(az, ain, cx, cy, rmax):
    """Azimuth/takeoff (degrees) to (x, y) on the focal sphere plot."""
    rad = np.pi / 180.0

    azr = az * rad
    ainr = ain * rad

    # Adjust for takeoff angles > 90 deg
    if np.any(ain > 90):
        ainr = np.where(ain > 90, np.pi - ainr, ainr)
        azr = np.where(ain > 90, np.pi + azr, azr)

    con = rmax * np.sqrt(2.0)
    r = con * np.sin(ainr * 0.5)

    x = r * np.sin(azr) + cx
    y = r * np.cos(azr) + cy

    return x, y


def m_plpl(strkdg, dpidg, cx, cy, rmax):
    """Projection of a fault plane onto the focal sphere (180-point curve)."""
    rad = np.pi / 180.0
    strkrd = np.deg2rad(strkdg)
    diprd = np.deg2rad(dpidg)
    tpd = np.tan(np.pi * 0.5 - diprd) ** 2

    # Vertical plane special case
    if dpidg == 90.0:
        x = np.array([rmax * np.sin(strkrd) + cx, rmax * np.sin(strkrd + np.pi) + cx])
        y = np.array([rmax * np.cos(strkrd) + cy, rmax * np.cos(strkrd + np.pi) + cy])
        return x, y

    saz = np.zeros(92)
    ainp = np.zeros(92)
    for i in range(1, 91):
        ang = (i - 1) * rad
        arg = np.sqrt((np.cos(diprd) ** 2) * (np.sin(ang) ** 2)) / np.cos(ang)
        saz[i] = np.arctan(arg)
        taz = np.tan(saz[i]) ** 2
        arg = np.sqrt(tpd + tpd * taz + taz)
        ainp[i] = np.arccos(np.tan(saz[i]) / arg)

    saz[91] = 90.0 * rad
    ainp[91] = np.pi * 0.5 - diprd

    con = rmax * np.sqrt(2.0)
    x = np.zeros(180)
    y = np.zeros(180)
    for i in range(1, 181):
        if i <= 91:
            mi = i
            az = saz[i] + strkrd
        else:
            mi = 181 - i
            az = np.pi - saz[mi] + strkrd

        radius = con * np.sin(ainp[mi] * 0.5)
        x[i - 1] = radius * np.sin(az) + cx
        y[i - 1] = radius * np.cos(az) + cy

    return x, y


def circle(ax, center, radius, nop, style="k-"):
    """Draw a circle on ``ax`` and return the Line2D handle."""
    theta = np.linspace(0, 2 * np.pi, nop)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    (handle,) = ax.plot(x, y, style)
    return handle


def _idx_range_matlab(start, end, step, n):
    """Convert one-based range bounds to zero-based row indices."""
    if step == 1:
        return np.arange(start - 1, end, 1, dtype=int)
    elif step == -1:
        return np.arange(start - 1, end - 1, -1, dtype=int)
    else:
        raise ValueError("Only step = +/-1 supported")


def fill_quadrants(
    x1,
    y1,
    x2,
    y2,
    xyc,
    px,
    py,
    tx,
    ty,
    bx=0.0,
    by=0.0,
    ax=None,
    comp_color=(0.7, 0.7, 0.7),
    rake=None,
):
    """Fill compressive/dilatational quadrants of the beachball."""
    if ax is None:
        ax = plt.gca()

    xyc = np.asarray(xyc)
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)
    y2 = np.asarray(y2)
    n = xyc.shape[0]

    # SPECIAL CASE: extreme rake values (pure strike-slip or dip-slip)
    if rake is not None and (
        abs(rake) < 1e-6 or abs(abs(rake) - 180) < 1e-6 or abs(abs(rake) - 90) < 1e-6
    ):
        angles = np.arctan2(xyc[:, 1] - by, xyc[:, 0] - bx)
        angles = (angles + 2 * np.pi) % (2 * np.pi)

        quads = [
            xyc[(angles >= 0) & (angles < np.pi / 2)],
            xyc[(angles >= np.pi / 2) & (angles < np.pi)],
            xyc[(angles >= np.pi) & (angles < 3 * np.pi / 2)],
            xyc[(angles >= 3 * np.pi / 2) & (angles < 2 * np.pi)],
        ]

        polys = [np.vstack([q, q[0]]) for q in quads]

        # classify: T quadrant + opposite quadrant are compressive
        comp = np.zeros(4, dtype=int)
        for i, poly in enumerate(polys):
            if Path(poly).contains_point((tx, ty)):
                comp[i] = 1
                comp[(i + 2) % 4] = 1
                break

        for i, poly in enumerate(polys):
            color = comp_color if comp[i] else (1, 1, 1)
            ax.fill(poly[:, 0], poly[:, 1], color=color, zorder=0)

        return

    # GENERAL CASE
    d1 = np.linalg.norm(xyc - np.array([x1[0], y1[0]]), axis=1)
    d2 = np.linalg.norm(xyc - np.array([x1[-1], y1[-1]]), axis=1)
    d3 = np.linalg.norm(xyc - np.array([x2[0], y2[0]]), axis=1)
    d4 = np.linalg.norm(xyc - np.array([x2[-1], y2[-1]]), axis=1)

    e = [int(np.argmin(d1)), int(np.argmin(d2)), int(np.argmin(d3)), int(np.argmin(d4))]

    # Convert to one-based indices for circle-arc bounds
    em = [idx + 1 for idx in e]
    e1m_p, e2m_p, e3m, e4m = em[0], em[1], em[2], em[3]

    # Closest points of each nodal plane to the B axis
    b = np.array([bx, by])
    c1 = np.linalg.norm(np.column_stack([x1, y1]) - b, axis=1)
    c2 = np.linalg.norm(np.column_stack([x2, y2]) - b, axis=1)
    e1_0 = int(np.argmin(c1))
    e2_0 = int(np.argmin(c2))
    e1m = e1_0 + 1
    e2m = e2_0 + 1
    del e1m, e2m

    def circle_arc(a, bb):
        if a > bb:
            idx_rev = _idx_range_matlab(a, bb, -1, n)
            if len(idx_rev) > n / 2:
                return np.concatenate(
                    [_idx_range_matlab(a, n, 1, n), _idx_range_matlab(1, bb, 1, n)]
                )
            return idx_rev
        idx_fwd = _idx_range_matlab(a, bb, 1, n)
        if len(idx_fwd) > n / 2:
            return np.concatenate([_idx_range_matlab(a, 1, -1, n), _idx_range_matlab(n, bb, -1, n)])
        return idx_fwd

    a1 = np.vstack(
        [
            np.column_stack([x1[: e1_0 + 1], y1[: e1_0 + 1]]),
            np.column_stack([x2[e2_0::-1], y2[e2_0::-1]]),
            xyc[circle_arc(e3m, e1m_p), :],
        ]
    )

    a2 = np.vstack(
        [
            np.column_stack([x2[: e2_0 + 1], y2[: e2_0 + 1]]),
            np.column_stack([x1[e1_0:], y1[e1_0:]]),
            xyc[circle_arc(e2m_p, e3m), :],
        ]
    )

    a3 = np.vstack(
        [
            np.column_stack([x1[e1_0:][::-1], y1[e1_0:][::-1]]),
            np.column_stack([x2[e2_0:], y2[e2_0:]]),
            xyc[circle_arc(e4m, e2m_p), :],
        ]
    )

    a4 = np.vstack(
        [
            np.column_stack([x2[e2_0:][::-1], y2[e2_0:][::-1]]),
            np.column_stack([x1[: e1_0 + 1][::-1], y1[: e1_0 + 1][::-1]]),
            xyc[circle_arc(e1m_p, e4m), :],
        ]
    )

    polys = [a1, a2, a3, a4]

    # Classification: find quadrant containing P or T
    dc = None
    for i, poly in enumerate(polys, start=1):
        path = Path(poly)
        if path.contains_point((px, py)):
            dc = (i, 0)  # P-axis quadrant -> dilatation
            break
        elif path.contains_point((tx, ty)):
            dc = (i, 1)  # T-axis quadrant -> compression
            break

    comp = np.zeros(4, dtype=int)

    if dc is not None:
        q_idx, comp_flag = dc
        if q_idx % 2 == 1:  # 1 or 3
            comp[[0, 2]] = comp_flag
            comp[[1, 3]] = 1 - comp_flag
        else:  # 2 or 4
            comp[[0, 2]] = 1 - comp_flag
            comp[[1, 3]] = comp_flag

    for i, poly in enumerate(polys):
        color = comp_color if comp[i] else (1.0, 1.0, 1.0)
        ax.fill(poly[:, 0], poly[:, 1], color=color, zorder=0)


# ---------------------------------------------------------------------------
# Beachball with HPD mechanisms, S polarization vectors and FMP markers
# ---------------------------------------------------------------------------


def plot_full_beachball_hpd(
    stat_df,
    strike_best,
    dip_best,
    rake_best,
    model,
    fdepth,
    evcode,
    hpd,
    plot_spol_vectors=True,
    save_path=None,
    log=print,
):
    """Beachball of the MAP mechanism with HPD planes, S-pol vectors and FMPs.

    Also computes the FMP match/misfit statistics, the average S polarization
    misfit and the RMS Kagan angle of the HPD region vs the MAP solution.

    Returns (rms_kagan, ppol_matches, ppol_misfits, ppol_matches_score,
    ppol_misfits_score, n_spol, avg_spol_misfit).
    """
    log(f"Event: {evcode}")

    circ_width = 0.5
    plane_color = "k"
    plane_width = 0.8
    plane_color2 = [0.6, 0.6, 0.6]
    plane_width2 = 0.4

    rake_best = sanitize_rake(rake_best)
    dip_best = sanitize_dip(dip_best)

    cx, cy, rmax = 0.0, 0.0, 1.0

    fps_in = np.array([strike_best, dip_best, rake_best], dtype=float)
    fps_in[2] = sanitize_rake(fps_in[2])

    # Moment tensor & focal mechanism
    mt = momtens(1.0, *fps_in)
    fps, t_ax, p_ax, b_ax, _, _, _ = rmech_fast(mt)

    str1, dip1, rak1 = fps[0]
    str2, dip2, rak2 = fps[1]

    rak1 = sanitize_rake(rak1)
    rak2 = sanitize_rake(rak2)
    dip1 = sanitize_dip(dip1)
    dip2 = sanitize_dip(dip2)

    if plot_spol_vectors:
        # Ray grid (lower hemisphere, upgoing view)
        azi = np.deg2rad(np.arange(0, 361, 10) + 180)
        take = np.deg2rad(180 - np.arange(90, 181, 10))

        azimuth, takeoff = np.meshgrid(azi, take)
        azimuth_draw, takeoff_draw = np.meshgrid(azi, take)

        # Radiation pattern on grid
        gp, _, gsh, gsv = rpgen_mgs(str1, dip1, rak1, np.rad2deg(takeoff), np.rad2deg(azimuth))

        # Focal sphere projection
        spx, spy = m_pltsym(np.rad2deg(azimuth_draw), np.rad2deg(takeoff_draw), cx, cy, rmax)

        gsv = -gsv  # SV sign convention for polarization vectors

        # Rotation correction (grid)
        ivel = np.where(model[:, 1] < fdepth)[0][-1]
        ainc_all = np.degrees(np.arcsin(model[0, 0] * np.sin(takeoff) / model[ivel, 0]))
        rotcorr_all = ainc_all - np.degrees(takeoff)

        rxp, ryp, _, _ = rotate_full_grid(azimuth, takeoff, gp, gsh, gsv, rotcorr_all)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    hc = circle(ax, [cx, cy], rmax, 720, "k-")
    hc.set_linewidth(circ_width)

    # Best nodal planes
    x1, y1 = m_plpl(str1, dip1, cx, cy, rmax)
    x2, y2 = m_plpl(str2, dip2, cx, cy, rmax)

    # Principal axes
    px, py = m_pltsym(p_ax["tr"], 90 - p_ax["pl"], cx, cy, rmax)
    tx, ty = m_pltsym(t_ax["tr"], 90 - t_ax["pl"], cx, cy, rmax)
    bx, by = m_pltsym(b_ax["tr"], 90 - b_ax["pl"], cx, cy, rmax)

    xyc = np.column_stack([hc.get_xdata(), hc.get_ydata()])
    fill_quadrants(
        x1,
        y1,
        x2,
        y2,
        xyc,
        px,
        py,
        tx,
        ty,
        bx,
        by,
        ax,
        comp_color=(0.9, 0.9, 0.9),
        rake=fps_in[2],
    )

    valid_mechanisms = [tuple(row) for row in hpd]

    if len(valid_mechanisms) > 200:
        idx = np.random.choice(len(valid_mechanisms), size=200, replace=False)
        valid_mechanisms = [valid_mechanisms[i] for i in idx]

    if len(valid_mechanisms) > 100:
        plane_width2 = 0.2

    m_best = momtens(1, str1, dip1, rak1)
    kagan_list = []
    for mech in valid_mechanisms:
        s1, d1, r1 = mech
        fps_ind = mech_both_planes(s1, d1, r1)
        strike_ind1, dip_ind1, _ = fps_ind[0]
        strike_ind2, dip_ind2, _ = fps_ind[1]
        ix1, iy1 = m_plpl(strike_ind1, dip_ind1, cx, cy, rmax)
        ix2, iy2 = m_plpl(strike_ind2, dip_ind2, cx, cy, rmax)
        ax.plot(ix1, iy1, "-", linewidth=plane_width2, color=plane_color2)
        ax.plot(ix2, iy2, "-", linewidth=plane_width2, color=plane_color2)

        # axes for individual solutions
        t_mt = momtens(1.0, *mech)
        _, t_t, t_p, _, _, _, _ = rmech_fast(t_mt)
        t_px, t_py = m_pltsym(t_p["tr"], 90 - t_p["pl"], cx, cy, rmax)
        t_tx, t_ty = m_pltsym(t_t["tr"], 90 - t_t["pl"], cx, cy, rmax)
        ax.text(t_px, t_py, "P", fontsize=8, color=[1, 0.5, 0.5])
        ax.text(t_tx, t_ty, "T", fontsize=8, color=[0.5, 0.5, 1])

        # Kagan angle of each HPD mechanism from the best (MAP) solution
        m1 = momtens(1, s1, d1, r1)
        kagan_list.append((mech, calc_theta(m_best, m1)))

    kag_angles = [a for (_, a) in kagan_list]
    rms_kagan = np.sqrt(np.mean(np.square(kag_angles)))

    # Best nodal planes
    ax.plot(x1, y1, "-", linewidth=plane_width, color=plane_color)
    ax.plot(x2, y2, "-", linewidth=plane_width, color=plane_color)

    # P/T markers
    ax.plot(px, py, "d", color=[0.6, 0, 0], markersize=8, linewidth=2, markerfacecolor="r")
    ax.plot(tx, ty, "d", color=[0, 0, 0.5], markersize=8, linewidth=2, markerfacecolor="b")
    ax.text(px + rmax / 30, py + 2 * rmax / 30, "P", fontsize=16, color="r", fontweight="bold")
    ax.text(tx + rmax / 30, ty + 2 * rmax / 30, "T", fontsize=16, color="b", fontweight="bold")

    if plot_spol_vectors:
        # Rotated S field (gray)
        ax.quiver(
            spx, spy, rxp, ryp, angles="xy", scale_units="xy", scale=5, width=0.002, color="gray"
        )

    ax.set_aspect("equal")
    ax.axis("off")

    title = (
        "Focal mechanism for event: " + evcode + "\n"
        f"Strike: {strike_best:.1f}°, Dip: {dip_best:.1f}°, "
        f"Rake: {rake_best:.1f}° (RMS={rms_kagan:.1f}°)"
    )
    ax.set_title(title)
    ax.set_xlabel(f"Strike/Dip/Rake : {str1:.1f} / {dip1:.1f} / {rak1:.1f}")

    polsize = 6
    statsize = 8
    scale_test = 0.4  # size of theoretical S-pol vectors
    spolwid = 0.003  # width of S-pol vectors
    spolwid2 = 2
    spolscale = 1
    line_scale = 0.1  # S-pol direction measurement line length

    spol_misfits = []
    ppol_misfits = 0
    ppol_matches = 0
    ppol_misfits_score = 0.0
    ppol_matches_score = 0.0

    spol_num = 0

    # LOOP OVER STATIONS
    for row in stat_df.itertuples(index=False):
        statname = row.station
        pol = row.polarity
        clarity = row.clarity
        azimuth_test = row.azimuth
        takeoff_test = row.takeoff
        spol_val = row.Spol

        if pol in ["+", "-"] or not pd.isna(spol_val):
            log(" ")
            log(f"STATION:  {statname}")
        else:
            continue

        if takeoff_test is None or pd.isna(takeoff_test):
            log(f"! [{statname}] no takeoff angle, skipping")
            continue

        if takeoff_test > 90:
            takeoff_test = 180 - takeoff_test
            azimuth_test = (180 + azimuth_test) % 360

        sx, sy = m_pltsym(azimuth_test, takeoff_test, cx, cy, rmax)

        # S-wave polarization directions
        if not pd.isna(spol_val):
            spol_num += 1

            spol_rad = np.deg2rad(90 - spol_val)

            az = np.deg2rad(azimuth_test)
            take = np.deg2rad(takeoff_test)

            # 1. Rotate test ray using SOURCE angles
            s_rot, _, _, _ = rotate_single_ray(
                azimuth_test, takeoff_test, str1, dip1, rak1, model, fdepth
            )

            gpt, _, gsht, gsvt = rpgen_mgs(str1, dip1, rak1, takeoff_test, azimuth_test)

            _, _, _, svec_t = compute_ray_vectors(az, take, gpt, gsht, -gsvt)

            # Focal sphere position
            spxt, spyt = m_pltsym(azimuth_test, takeoff_test, cx, cy, rmax)

            # Unrotated S-wave vector (black)
            ax.quiver(
                spxt,
                spyt,
                svec_t[0] * scale_test,
                svec_t[1] * scale_test,
                angles="xy",
                scale_units="xy",
                scale=spolscale,
                color="k",
                width=spolwid,
                zorder=11,
            )

            # Rotated S-wave vector (red)
            ax.quiver(
                spxt,
                spyt,
                s_rot[0] * scale_test,
                s_rot[1] * scale_test,
                angles="xy",
                scale_units="xy",
                scale=spolscale,
                color="r",
                width=spolwid,
                zorder=12,
            )

            # Measured polarization direction (green), both directions
            dx = np.cos(spol_rad)
            dy = np.sin(spol_rad)

            ax.plot(
                [spxt, spxt + dx * line_scale],
                [spyt, spyt + dy * line_scale],
                color="green",
                linewidth=spolwid2,
                zorder=13,
            )
            ax.plot(
                [spxt, spxt - dx * line_scale],
                [spyt, spyt - dy * line_scale],
                color="green",
                linewidth=spolwid2,
                zorder=13,
            )

            pol_dir = polarization_direction_from_vector(s_rot)
            log(f"Measured S-polarization direction [STATION]: {spol_val % 180:.1f} deg")
            log(f"Calculated S-polarization direction [SURFACE]: {pol_dir % 180:.1f} deg")
            angdf = angle_diff_deg(spol_val * 2, pol_dir * 2) / 2
            log(f"Difference [Measured - Theoretical]:  {angdf:.1f} deg")
            spol_misfits.append(abs(angdf))

        # Plot FMPs, count misfits in the final solution
        if pol in ["+", "-"]:
            p_pred = np.sign(
                rpgen_mgs_p(strike_best, dip_best, rake_best, takeoff_test, azimuth_test)
            )

            if pol == "+" and clarity == 1:  # emergent compression
                ax.plot(
                    sx,
                    sy,
                    "o",
                    markersize=polsize,
                    markeredgecolor="b",
                    markerfacecolor="none",
                    linewidth=2,
                    zorder=40,
                )
                if p_pred > 0:
                    log("P-polarity: emergent Compression (matching)")
                    ppol_matches += 1
                    ppol_matches_score += 0.5
                else:
                    log("! P-polarity: emergent Compression (MISMATCH)")
                    ppol_misfits += 1
                    ppol_misfits_score += 0.5

            if pol == "-" and clarity == 1:  # emergent dilatation
                ax.plot(
                    sx,
                    sy,
                    "v",
                    markersize=polsize,
                    markeredgecolor="r",
                    markerfacecolor="none",
                    linewidth=2,
                    zorder=40,
                )
                if p_pred < 0:
                    log("P-polarity: emergent Dilatation (matching)")
                    ppol_matches += 1
                    ppol_matches_score += 0.5
                else:
                    log("! P-polarity: emergent Dilatation (MISMATCH)")
                    ppol_misfits += 1
                    ppol_misfits_score += 0.5

            if pol == "+" and clarity == 0:  # impulsive compression
                ax.plot(
                    sx,
                    sy,
                    "o",
                    markersize=polsize,
                    markeredgecolor="b",
                    markerfacecolor=[0, 0, 0.5],
                    linewidth=2,
                    zorder=40,
                )
                if p_pred > 0:
                    log("P-polarity: impulsive Compression (matching)")
                    ppol_matches += 1
                    ppol_matches_score += 1
                else:
                    log("! P-polarity: impulsive Compression (MISMATCH)")
                    ppol_misfits += 1
                    ppol_misfits_score += 1

            if pol == "-" and clarity == 0:  # impulsive dilatation
                ax.plot(
                    sx,
                    sy,
                    "v",
                    markersize=polsize,
                    markeredgecolor="r",
                    markerfacecolor=[0.5, 0, 0],
                    linewidth=2,
                    zorder=40,
                )
                if p_pred < 0:
                    log("P-polarity: Dilatation (matching)")
                    ppol_matches += 1
                    ppol_matches_score += 1
                else:
                    log("! P-polarity: Dilatation (MISMATCH)")
                    ppol_misfits += 1
                    ppol_misfits_score += 1

        if pol in ["+", "-"] or not pd.isna(spol_val):
            ax.text(
                sx,
                sy + 0.02,
                statname,
                ha="center",
                va="bottom",
                fontsize=statsize,
                fontweight="light",
                color="black",
                zorder=45,
            )

    log(" ")
    n_fmp = ppol_matches + ppol_misfits
    if n_fmp > 0:
        log(
            f"- Matching P-wave polarities: {ppol_matches}/{n_fmp} "
            f"({100 * ppol_matches / n_fmp:.1f}%)"
        )
        n_fmp_w = ppol_matches_score + ppol_misfits_score
        log(
            f"- Matching weighted P-wave polarities: {ppol_matches_score}/{n_fmp_w} "
            f"({100 * ppol_matches_score / n_fmp_w:.1f}%)"
        )
    if spol_num > 0:
        log(
            f"- Average misfits of {len(spol_misfits)} S-wave polarization "
            f"directions (unweighted): {np.mean(spol_misfits):.1f}"
        )

    log(f"- RMS Kagan angle: {rms_kagan:.1f} deg")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    avg_spol_misfit = np.mean(spol_misfits) if spol_misfits else np.nan

    return (
        rms_kagan,
        ppol_matches,
        ppol_misfits,
        ppol_matches_score,
        ppol_misfits_score,
        len(spol_misfits),
        avg_spol_misfit,
    )


# ---------------------------------------------------------------------------
# Marginal PDF layout
# ---------------------------------------------------------------------------


def _gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _wrapped_gaussian(x, a, mu, sigma, low, high):
    period = high - low
    return (
        _gaussian(x, a, mu, sigma)
        + _gaussian(x, a, mu - period, sigma)
        + _gaussian(x, a, mu + period, sigma)
    )


def plot_1d_marginal(
    ax,
    grid_1d_x,
    post_1d,
    mu1_fixed,
    mu2_fixed,
    label_1d,
    gauss_p1=None,
    gauss_p2=None,
    wrap_range=None,
):
    """Horizontal 1D marginal with MAP lines and per-plane Gaussian overlays."""
    x_deg = np.rad2deg(grid_1d_x)
    ax.plot(x_deg, post_1d, color="black", lw=2)
    ax.set_ylabel("Marginal PDF")

    # MAP vertical lines
    ax.axvline(mu1_fixed, color="red", linestyle="--", linewidth=2)
    ax.axvline(mu2_fixed, color="blue", linestyle="--", linewidth=2)

    if gauss_p1 is not None or gauss_p2 is not None:
        fine_1d = np.linspace(-180, 180, 2000)

        if gauss_p1 is not None:
            a1 = np.interp(gauss_p1["mu_free"], x_deg, post_1d)
            if wrap_range is None:
                g1 = _gaussian(fine_1d, a1, gauss_p1["mu_free"], gauss_p1["sigma_free"])
            else:
                low, high = wrap_range
                g1 = _wrapped_gaussian(
                    fine_1d, a1, gauss_p1["mu_free"], gauss_p1["sigma_free"], low, high
                )
            ax.plot(fine_1d, g1, color="red", linestyle="--", lw=1.2, label="MAP plane Gaussian")

        if gauss_p2 is not None:
            a2 = np.interp(gauss_p2["mu_free"], x_deg, post_1d)
            if wrap_range is None:
                g2 = _gaussian(fine_1d, a2, gauss_p2["mu_free"], gauss_p2["sigma_free"])
            else:
                low, high = wrap_range
                g2 = _wrapped_gaussian(
                    fine_1d, a2, gauss_p2["mu_free"], gauss_p2["sigma_free"], low, high
                )
            ax.plot(fine_1d, g2, color="blue", linestyle="--", lw=1.2, label="Conj. plane Gaussian")

    ax.tick_params(labelbottom=False)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{label_1d} Marginal")


def plot_1d_marginal_vertical(
    ax,
    grid_y,
    post_1d,
    mu1_fixed,
    mu2_fixed,
    label_1d,
    gauss_p1=None,
    gauss_p2=None,
    wrap_range=None,
    showlegend=False,
):
    """Vertical 1D marginal with MAP lines and per-plane Gaussian overlays."""
    yvals = np.rad2deg(grid_y)

    idx = np.argsort(yvals)
    yvals_sorted = yvals[idx]
    post_sorted = post_1d[idx]

    ax.plot(post_sorted, yvals_sorted, color="black")

    ax.set_xlabel("Marginal PDF")
    ax.set_ylabel(f"{label_1d} Marginal")
    ax.grid(True, linestyle="--", alpha=0.3)

    # MAP lines
    ax.axhline(mu1_fixed, color="red", linestyle="--", linewidth=2, label="MAP solution")
    ax.axhline(
        mu2_fixed, color="blue", linestyle="--", linewidth=2, label="conjugate plane solution"
    )

    if gauss_p1 is not None or gauss_p2 is not None:
        fine_y = np.linspace(yvals_sorted.min(), yvals_sorted.max(), 2000)

        if gauss_p1 is not None:
            a1 = np.interp(gauss_p1["mu_free"], yvals, post_1d)
            if wrap_range is None:
                g1 = _gaussian(fine_y, a1, gauss_p1["mu_free"], gauss_p1["sigma_free"])
            else:
                low, high = wrap_range
                g1 = _wrapped_gaussian(
                    fine_y, a1, gauss_p1["mu_free"], gauss_p1["sigma_free"], low, high
                )
            ax.plot(g1, fine_y, color="red", linestyle="--", lw=1.2, label="MAP plane Gaussian")

        if gauss_p2 is not None:
            a2 = np.interp(gauss_p2["mu_free"], yvals, post_1d)
            if wrap_range is None:
                g2 = _gaussian(fine_y, a2, gauss_p2["mu_free"], gauss_p2["sigma_free"])
            else:
                low, high = wrap_range
                g2 = _wrapped_gaussian(
                    fine_y, a2, gauss_p2["mu_free"], gauss_p2["sigma_free"], low, high
                )
            ax.plot(g2, fine_y, color="blue", linestyle="--", lw=1.2, label="Conj. plane Gaussian")

    ax.invert_xaxis()

    # Force y-axis to increase bottom -> top
    ax.set_ylim(yvals_sorted.min(), yvals_sorted.max())

    if showlegend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0)


def plot_2d_marginal(
    ax,
    grid_1d_x,
    grid_1d_y,
    post_2d_plot,
    cmap,
    plottype,
    use_strike360,
    label_2d_x,
    label_2d_y,
    mu1_fixed,
    mu1_fixed_y,
    mu2_fixed,
    mu2_fixed_y,
    showlegend=False,
):
    """2D joint marginal with MAP markers."""
    pcm = ax.pcolormesh(
        np.rad2deg(grid_1d_x), np.rad2deg(grid_1d_y), post_2d_plot, cmap=cmap, shading="auto"
    )

    ax.set_xlabel(f"{label_2d_x} (deg)")
    ax.set_ylabel(f"{label_2d_y} (deg)")

    if plottype == "strike1":  # x = Strike, y = Dip
        if not use_strike360:
            ax.set_xlim(0, 180)
            ax.set_xticks(np.arange(0, 181, 30))
        else:
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 30))
        ax.set_ylim(0, 90)
        ax.set_yticks(np.arange(0, 91, 15))
    elif plottype == "strike2":  # x = Strike, y = Rake
        if not use_strike360:
            ax.set_xlim(0, 180)
            ax.set_xticks(np.arange(0, 181, 30))
        else:
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 30))
        ax.set_ylim(-180, 180)
        ax.set_yticks(np.arange(-180, 181, 45))
    elif plottype == "dip1":  # x = Dip, y = Rake
        ax.set_xlim(0, 90)
        ax.set_xticks(np.arange(0, 91, 15))
        ax.set_ylim(-180, 180)
        ax.set_yticks(np.arange(-180, 181, 45))
    elif plottype == "dip2":  # x = Dip, y = Strike
        ax.set_xlim(0, 90)
        ax.set_xticks(np.arange(0, 91, 15))
        if not use_strike360:
            ax.set_ylim(0, 180)
            ax.set_yticks(np.arange(0, 181, 30))
        else:
            ax.set_ylim(0, 360)
            ax.set_yticks(np.arange(0, 361, 30))
    elif plottype == "rake1":  # x = Rake, y = Strike
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 45))
        if not use_strike360:
            ax.set_ylim(0, 180)
            ax.set_yticks(np.arange(0, 181, 30))
        else:
            ax.set_ylim(0, 360)
            ax.set_yticks(np.arange(0, 361, 30))
    elif plottype == "rake2":  # x = Rake, y = Dip
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 45))
        ax.set_ylim(0, 90)
        ax.set_yticks(np.arange(0, 91, 15))

    # MAP markers
    ax.plot(
        mu1_fixed,
        mu1_fixed_y,
        marker="*",
        markersize=19,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=2,
        linestyle="none",
        label="MAP fps",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
    )
    ax.plot(
        mu2_fixed,
        mu2_fixed_y,
        marker="*",
        markersize=16,
        markerfacecolor="none",
        markeredgecolor="cyan",
        markeredgewidth=2,
        linestyle="none",
        label="MAP conjugate plane",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
    )

    if showlegend:
        ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.07), borderaxespad=0)

    return pcm


def plot_all_marginals(
    post_3d,
    theta_best,
    theta_mean,
    strike_grid,
    dip_grid,
    rake_grid,
    plotparams,
    gs_fits,
    save_path=None,
    evcode=None,
):
    """Composite figure with three 2D and three 1D marginals."""
    strike_best1, dip_best1, rake_best1, strike_best2, dip_best2, rake_best2 = theta_best
    strike_mean1, dip_mean1, rake_mean1, strike_mean2, dip_mean2, rake_mean2 = theta_mean
    use_strike360 = plotparams["use_strike360"]
    cmap = plotparams["cmap"]

    cases = {}

    # STRIKE1: x=Strike, y=Dip (sum over rake)
    post_2d_strike = np.sum(post_3d, axis=2)
    cases["strike1"] = {
        "post_2d_plot": post_2d_strike.T,
        "grid_x": strike_grid,
        "grid_y": dip_grid,
        "label_x": "Strike",
        "label_y": "Dip",
        "mu1_fixed_y": dip_best1,
        "mu2_fixed_y": dip_best2,
        "label_1d": "Strike",
        "post_1d": np.sum(post_2d_strike, axis=1),
        "mu1_fixed": strike_best1,
        "mu2_fixed": strike_best2,
    }

    # DIP1: x=Dip, y=Rake (sum over strike)
    post_2d_dip = np.sum(post_3d, axis=0)
    cases["dip1"] = {
        "post_2d_plot": post_2d_dip.T,
        "grid_x": dip_grid,
        "grid_y": rake_grid,
        "label_x": "Dip",
        "label_y": "Rake",
        "mu1_fixed_y": rake_best1,
        "mu2_fixed_y": rake_best2,
        "label_1d": "Dip",
        "post_1d": np.sum(post_2d_dip, axis=1),
        "mu1_fixed": dip_best1,
        "mu2_fixed": dip_best2,
    }

    # RAKE1: x=Rake, y=Strike (sum over dip)
    post_2d_rake = np.sum(post_3d, axis=1)
    cases["rake1"] = {
        "post_2d_plot": post_2d_rake,
        "grid_x": rake_grid,
        "grid_y": strike_grid,
        "label_x": "Rake",
        "label_y": "Strike",
        "mu1_fixed_y": strike_best1,
        "mu2_fixed_y": strike_best2,
        "label_1d": "Rake",
        "post_1d": np.sum(post_2d_rake, axis=0),
        "mu1_fixed": rake_best1,
        "mu2_fixed": rake_best2,
    }

    # DIP2: x=Dip, y=Strike (sum over rake)
    post_2d_dip2 = np.sum(post_3d, axis=2).T  # -> (dip, strike)
    cases["dip2"] = {
        "post_2d_plot": post_2d_dip2.T,
        "grid_x": dip_grid,
        "grid_y": strike_grid,
        "label_x": "Dip",
        "label_y": "Strike",
        "mu1_fixed_y": strike_best1,
        "mu2_fixed_y": strike_best2,
        "label_1d": "Dip",
        "post_1d": np.sum(post_2d_dip2, axis=1),
        "mu1_fixed": dip_best1,
        "mu2_fixed": dip_best2,
    }

    # RAKE2: x=Rake, y=Dip (sum over strike)
    post_2d_rake2 = np.sum(post_3d, axis=0).T  # -> (rake, dip)
    cases["rake2"] = {
        "post_2d_plot": post_2d_rake2.T,
        "grid_x": rake_grid,
        "grid_y": dip_grid,
        "label_x": "Rake",
        "label_y": "Dip",
        "mu1_fixed_y": dip_best1,
        "mu2_fixed_y": dip_best2,
        "label_1d": "Rake",
        "post_1d": np.sum(post_2d_rake2, axis=1),
        "mu1_fixed": rake_best1,
        "mu2_fixed": rake_best2,
    }

    fig = plt.figure(figsize=(16, 12))

    gs = fig.add_gridspec(3, 3, height_ratios=[1, 2, 1], width_ratios=[0.4, 1.2, 0.4])

    # ROW 0
    ax_empty_00 = fig.add_subplot(gs[0, 0])  # info box
    ax_1d_rake_h = fig.add_subplot(gs[0, 1])  # rake 1D horizontal
    ax_1d_dip_h = fig.add_subplot(gs[0, 2])  # dip 1D horizontal

    # ROW 1
    ax_1d_strike_v = fig.add_subplot(gs[1, 0])  # strike 1D vertical
    ax_2d_rake_strk = fig.add_subplot(gs[1, 1])  # rake-strike 2D
    ax_2d_dip_strk = fig.add_subplot(gs[1, 2])  # dip-strike 2D

    # ROW 2
    ax_1d_dip_v = fig.add_subplot(gs[2, 0])  # dip 1D vertical
    ax_2d_rake_dip = fig.add_subplot(gs[2, 1])  # rake-dip 2D
    ax_empty_22 = fig.add_subplot(gs[2, 2])  # empty

    ax_empty_00.axis("off")
    ax_empty_22.axis("off")

    d = cases["rake1"]
    plot_1d_marginal(
        ax_1d_rake_h,
        d["grid_x"],
        d["post_1d"],
        d["mu1_fixed"],
        d["mu2_fixed"],
        d["label_1d"],
        gs_fits["plane1"]["rake"],
        gs_fits["plane2"]["rake"],
        wrap_range=(-180, 180),
    )
    ax_1d_rake_h.set_yticks([])

    d = cases["dip1"]
    plot_1d_marginal(
        ax_1d_dip_h,
        d["grid_x"],
        d["post_1d"],
        d["mu1_fixed"],
        d["mu2_fixed"],
        d["label_1d"],
        gs_fits["plane1"]["dip"],
        gs_fits["plane2"]["dip"],
    )
    ax_1d_dip_h.yaxis.tick_right()
    ax_1d_dip_h.yaxis.set_label_position("right")
    ax_1d_dip_h.set_yticks([])

    d = cases["strike1"]  # vertical 1D
    plot_1d_marginal_vertical(
        ax_1d_strike_v,
        d["grid_x"],
        d["post_1d"],
        d["mu1_fixed"],
        d["mu2_fixed"],
        d["label_1d"],
        gauss_p1=gs_fits["plane1"]["strike"],
        gauss_p2=gs_fits["plane2"]["strike"],
        wrap_range=(0, 360),
        showlegend=True,
    )
    if use_strike360:
        ax_1d_strike_v.set_ylim(0, 360)
        ax_1d_strike_v.set_yticks(np.arange(0, 361, 30))
    else:
        ax_1d_strike_v.set_ylim(0, 180)
        ax_1d_strike_v.set_yticks(np.arange(0, 181, 30))
    ax_1d_strike_v.set_xticks([])
    ax_1d_strike_v.set_xlabel("")

    d = cases["rake1"]
    plot_2d_marginal(
        ax_2d_rake_strk,
        d["grid_x"],
        d["grid_y"],
        d["post_2d_plot"],
        cmap,
        "rake1",
        use_strike360,
        d["label_x"],
        d["label_y"],
        d["mu1_fixed"],
        d["mu1_fixed_y"],
        d["mu2_fixed"],
        d["mu2_fixed_y"],
    )
    ax_2d_rake_strk.set_yticks([])
    ax_2d_rake_strk.set_ylabel("")
    ax_2d_rake_strk.set_xticks([])
    ax_2d_rake_strk.set_xlabel("")

    d = cases["dip2"]
    plot_2d_marginal(
        ax_2d_dip_strk,
        d["grid_x"],
        d["grid_y"],
        d["post_2d_plot"],
        cmap,
        "dip2",
        use_strike360,
        d["label_x"],
        d["label_y"],
        d["mu1_fixed"],
        d["mu1_fixed_y"],
        d["mu2_fixed"],
        d["mu2_fixed_y"],
    )
    ax_2d_dip_strk.yaxis.tick_right()
    ax_2d_dip_strk.yaxis.set_label_position("right")

    d = cases["dip1"]  # vertical 1D
    plot_1d_marginal_vertical(
        ax_1d_dip_v,
        d["grid_x"],
        d["post_1d"],
        d["mu1_fixed"],
        d["mu2_fixed"],
        d["label_1d"],
        gauss_p1=gs_fits["plane1"]["dip"],
        gauss_p2=gs_fits["plane2"]["dip"],
    )
    ax_1d_dip_v.set_ylim(0, 90)
    ax_1d_dip_v.set_yticks(np.arange(0, 91, 15))
    ax_1d_dip_v.set_xticks([])

    d = cases["rake2"]
    plot_2d_marginal(
        ax_2d_rake_dip,
        d["grid_x"],
        d["grid_y"],
        d["post_2d_plot"],
        cmap,
        "rake2",
        use_strike360,
        d["label_x"],
        d["label_y"],
        d["mu1_fixed"],
        d["mu1_fixed_y"],
        d["mu2_fixed"],
        d["mu2_fixed_y"],
        showlegend=True,
    )
    ax_2d_rake_dip.yaxis.tick_right()
    ax_2d_rake_dip.yaxis.set_label_position("right")

    for ax in fig.axes:
        col = ax.get_subplotspec().colspan.start
        if col == 0:
            xmin, _ = ax.get_xlim()
            ax.set_xlim(xmin, 0)
        if col == 1:  # middle column (RAKE)
            ax.set_xlim(-180, 180)
            ax.set_xticks(np.arange(-180, 181, 45))
        if col == 2:  # right column (DIP)
            ax.set_xlim(0, 90)
            ax.set_xticks(np.arange(0, 91, 15))

        row = ax.get_subplotspec().rowspan.start
        if row == 0:
            _, ymax = ax.get_ylim()
            ax.set_ylim(0, ymax)

    ax_2d_rake_strk.set_yticks([])
    ax_2d_rake_strk.set_ylabel("")
    ax_2d_rake_strk.set_xticks([])
    ax_2d_rake_strk.set_xlabel("")

    for ax in fig.axes:
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    # Info box (top-left cell)
    strike_g1 = gs_fits["plane1"]["strike"]["mu_free"]
    dip_g1 = gs_fits["plane1"]["dip"]["mu_free"]
    rake_g1 = gs_fits["plane1"]["rake"]["mu_free"]
    strike_g2 = gs_fits["plane2"]["strike"]["mu_free"]
    dip_g2 = gs_fits["plane2"]["dip"]["mu_free"]
    rake_g2 = gs_fits["plane2"]["rake"]["mu_free"]

    text = (
        f"Event Code: {evcode}\n\n"
        f"MAP solution: {strike_best1:.1f} / {dip_best1:.1f} / {rake_best1:.1f}\n"
        f"Conj. plane: {strike_best2:.1f} / {dip_best2:.1f} / {rake_best2:.1f}\n\n"
        f"Mean plane 1: {strike_g1:.1f} / {dip_g1:.1f} / {rake_g1:.1f}\n"
        f"Mean plane 2: {strike_g2:.1f} / {dip_g2:.1f} / {rake_g2:.1f}\n"
    )

    ax_empty_00.text(
        -0.2, 1.0, text, ha="left", va="top", fontsize=10, transform=ax_empty_00.transAxes
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
