"""Bayesian grid-search inversion of first-motion polarities and S polarizations.

Parameter space angles (strike, dip, rake) are in radians throughout this module.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.special

from rebayfm.core.radiation import rpgen_mgs_p, rpgen_mgs_s, sanitize_dip, sanitize_rake
from rebayfm.core.spol import predict_s_polarization_angle

logger = logging.getLogger(__name__)


class PPolarityData:
    def __init__(self, station_ids, takeoff_angles, azimuths, observed_signs, mis_pick_probs):
        """
        station_ids    : list/array
        takeoff_angles : array (radians), 0 at nadir (down), increasing upwards
        azimuths       : array (radians), from North clockwise
        observed_signs : array of +1 or -1
        mis_pick_probs : array of per-station mis-pick probabilities
        """
        self.station_ids = np.asarray(station_ids)
        self.takeoff = np.asarray(takeoff_angles)
        self.az = np.asarray(azimuths)
        self.obs = np.asarray(observed_signs)
        self.p_mis = np.asarray(mis_pick_probs)


class SPolarizationData:
    def __init__(self, station_ids, takeoff_angles, azimuths, obs_polarization_angle, sigma_angle):
        """
        station_ids            : list/array
        takeoff_angles         : array (radians), 0 at nadir, increasing upwards
        azimuths               : array (radians), from North clockwise
        obs_polarization_angle : array (radians), horizontal polarization azimuth
                                 from North clockwise, wrapped to [0, pi)
        sigma_angle            : array of 1-sigma errors (radians)
        """
        self.station_ids = np.asarray(station_ids)
        self.takeoff = np.asarray(takeoff_angles)
        self.az = np.asarray(azimuths)
        self.obs = np.asarray(obs_polarization_angle)
        self.sigma = np.asarray(sigma_angle)

        n = len(self.takeoff)
        if not (len(self.obs) == len(self.sigma) == len(self.az) == n):
            raise ValueError("All S-wave arrays must have the same length")


def build_parameter_grid(strike_grid, dip_grid, rake_grid) -> np.ndarray:
    s, d, r = np.meshgrid(strike_grid, dip_grid, rake_grid, indexing="ij")
    return np.column_stack((s.ravel(), d.ravel(), r.ravel()))  # (N_mech, 3)


def dist_calc(lat1, lon1, lat2, lon2):
    """Haversine distance: (km, degrees of arc)."""
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    radius = 6371.0  # km
    distance = radius * 2.0 * np.arcsin(np.sqrt(a))

    return distance, 2.0 * np.rad2deg(np.arcsin(np.sqrt(a)))


def log_likelihood_p(
    theta, p_data: PPolarityData, k_allowed=None, lambda_penalty=None, mismatch_weight=1.0
):
    """P first-motion polarity log-likelihood with per-pick mis-pick probability."""
    strike, dip, rake = theta
    rake = np.deg2rad(sanitize_rake(np.rad2deg(rake)))
    dip = np.deg2rad(sanitize_dip(np.rad2deg(dip)))

    s_pred = np.sign(
        rpgen_mgs_p(
            np.rad2deg(strike),
            np.rad2deg(dip),
            np.rad2deg(rake),
            np.rad2deg(p_data.takeoff),
            np.rad2deg(p_data.az),
        )
    )

    s_obs = p_data.obs
    p_mis = p_data.p_mis

    matches = s_pred == s_obs
    mismatches = ~matches
    n_mismatch = np.sum(mismatches)

    # per-station likelihood
    p_obs = np.empty_like(p_mis, dtype=float)
    p_obs[matches] = 1.0 - p_mis[matches]
    p_obs[mismatches] = p_mis[mismatches] ** mismatch_weight

    eps = 1e-12
    p_obs = np.clip(p_obs, eps, 1.0)

    # optional penalty for mismatches
    if k_allowed is None or lambda_penalty is None:
        penalty = 0.0
    else:
        extra = max(0, n_mismatch - k_allowed)
        penalty = -lambda_penalty * extra

    return np.sum(np.log(p_obs)) + penalty


def log_likelihood_p_amp(
    theta,
    p_data: PPolarityData,
    k_allowed=None,
    lambda_penalty=None,
    p_min=0.05,
    mismatch_weight=1.0,
):
    """Like ``log_likelihood_p`` but weights the theoretical P amplitude in."""
    strike, dip, rake = theta
    rake = np.deg2rad(sanitize_rake(np.rad2deg(rake)))
    dip = np.deg2rad(sanitize_dip(np.rad2deg(dip)))

    amp = rpgen_mgs_p(
        np.rad2deg(strike),
        np.rad2deg(dip),
        np.rad2deg(rake),
        np.rad2deg(p_data.takeoff),
        np.rad2deg(p_data.az),
    )

    s_pred = np.sign(amp)
    s_pred[s_pred == 0] = 1

    s_obs = p_data.obs
    p_mis = p_data.p_mis

    # amplitude-based adjustment of mis-pick probability
    amp_norm = np.abs(amp) / (np.max(np.abs(amp)) + 1e-6)
    w = np.clip(amp_norm, 0.0, 1.0)
    p_mis_eff = (1 - w) * p_mis + w * p_min

    matches = s_pred == s_obs
    mismatches = ~matches
    n_mismatch = np.sum(mismatches)

    p_obs = np.empty_like(p_mis_eff, dtype=float)
    p_obs[matches] = 1.0 - p_mis_eff[matches]
    p_obs[mismatches] = p_mis[mismatches] ** mismatch_weight

    p_obs = np.clip(p_obs, 1e-12, 1.0)

    if k_allowed is None or lambda_penalty is None:
        penalty = 0.0
    else:
        extra = max(0, n_mismatch - k_allowed)
        penalty = -lambda_penalty * extra

    return np.sum(np.log(p_obs)) + penalty


def wrap_angle_diff(angle_diff, period=np.pi):
    """Wrap angle_diff to (-period/2, period/2]."""
    return (angle_diff + period / 2) % period - period / 2


def log_likelihood_s(theta, s_data: SPolarizationData, model, fdepth, period=np.pi, f_spol=1):
    """S-polarization log-likelihood, angles defined modulo ``period``.

    ``f_spol`` selects the misfit distribution: 1 Gaussian, 2 Laplace, 3 Cauchy.
    """
    strike, dip, rake = theta
    rake = np.deg2rad(sanitize_rake(np.rad2deg(rake)))
    dip = np.deg2rad(sanitize_dip(np.rad2deg(dip)))

    # S-wave amplitude, used for likelihood weight
    amp_norm = rpgen_mgs_s(
        np.rad2deg(strike),
        np.rad2deg(dip),
        np.rad2deg(rake),
        np.rad2deg(s_data.takeoff),
        np.rad2deg(s_data.az),
    )

    # apply floor to avoid zero weights
    amp_floor = 0.1
    w = np.clip(amp_norm, amp_floor, 1.0)

    alpha_pred = predict_s_polarization_angle(
        strike, dip, rake, s_data.takeoff, s_data.az, model, fdepth, s_data
    )
    alpha_obs = s_data.obs
    sigma = s_data.sigma

    d_alpha = wrap_angle_diff(alpha_obs - alpha_pred, period=period)

    if f_spol == 1:
        # Gaussian (short-tailed, not very robust)
        term1 = -0.5 * (d_alpha / sigma) ** 2
        term2 = -0.5 * np.log(2 * np.pi * sigma**2)
    elif f_spol == 2:
        # Laplace / double-exponential (long-tailed, more robust to outliers)
        b = sigma
        term1 = -np.abs(d_alpha) / b
        term2 = -np.log(2 * b)
    elif f_spol == 3:
        # Cauchy (very heavy-tailed)
        gamma = sigma
        term1 = -np.log(np.pi * gamma)
        term2 = -np.log(1 + (d_alpha / gamma) ** 2)
    else:
        raise ValueError(f"Unknown f_spol={f_spol}")

    log_l = term1 + term2

    return np.sum(w * log_l)  # weighted by S-wave amplitude


def log_prior(theta):
    """Uniform prior: strike [0, 2pi), dip (0, pi/2), rake [-pi, pi)."""
    strike, dip, rake = theta
    if not 0.0 <= strike < 2 * np.pi:
        return -np.inf
    if not 0.0 < dip < 0.5 * np.pi:
        return -np.inf
    if not -np.pi <= rake < np.pi:
        return -np.inf
    return 0.0


def log_posterior(
    theta,
    model,
    fdepth,
    p_data=None,
    s_data=None,
    period_s=np.pi,
    k_allowed=None,
    lambda_penalty=None,
    fmp_amp=False,
    w_spol=1.0,
    w_fmp=1.0,
    f_spol=1,
):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = 0.0

    if p_data is not None:
        if fmp_amp:
            ll += log_likelihood_p_amp(theta, p_data, k_allowed, lambda_penalty, 0.05, w_fmp)
        else:
            ll += log_likelihood_p(theta, p_data, k_allowed, lambda_penalty, w_fmp)

    if s_data is not None and len(s_data.obs) > 0:
        ll_s = log_likelihood_s(theta, s_data, model, fdepth, period=period_s, f_spol=f_spol)
        ll += w_spol * ll_s  # down-weight Spol

    return lp + ll


def grid_search_map(
    strike_grid,
    dip_grid,
    rake_grid,
    model,
    fdepth,
    p_data: PPolarityData | None = None,
    s_data: SPolarizationData | None = None,
    period_s=np.pi,
    k_allowed=None,
    lambda_penalty=None,
    fmp_amp=False,
    w_spol=1.0,
    w_fmp=1.0,
    f_spol=1,
):
    """Exhaustive grid search; returns (theta_best, log_post_grid, theta_grid)."""
    theta_grid = build_parameter_grid(strike_grid, dip_grid, rake_grid)
    log_post_grid = np.empty(theta_grid.shape[0], dtype=float)

    for i, theta in enumerate(theta_grid):
        log_post_grid[i] = log_posterior(
            theta,
            model,
            fdepth,
            p_data,
            s_data,
            period_s,
            k_allowed,
            lambda_penalty,
            fmp_amp,
            w_spol,
            w_fmp,
            f_spol,
        )

    idx_max = np.argmax(log_post_grid)
    theta_best = theta_grid[idx_max]
    return theta_best, log_post_grid, theta_grid


def grid_search_map_deprange(
    taupmodel,
    stat_df,
    use_spol,
    strike_grid,
    dip_grid,
    rake_grid,
    model,
    fdepth,
    p_data=None,
    s_data=None,
    period_s=np.pi,
    k_allowed=None,
    lambda_penalty=None,
    depth_range_km=1.0,
    depth_step_km=0.2,
    fmp_amp=False,
    w_spol=1.0,
    w_fmp=1.0,
    f_spol=1,
):
    """Grid search marginalized over a range of trial depths around ``fdepth``.

    Station takeoff/incidence angles are recomputed with TauP for each trial
    depth and the posterior is combined over depths with log-sum-exp.
    """
    # make a deep copy so the original DataFrame is not modified
    stat_df = stat_df.copy(deep=True)

    depth_offsets = np.arange(-depth_range_km, depth_range_km + 1e-6, depth_step_km)
    depth_grid = fdepth + depth_offsets

    theta_grid = build_parameter_grid(strike_grid, dip_grid, rake_grid)
    n_theta = theta_grid.shape[0]

    log_post_all_depths = []

    for depth in depth_grid:
        logger.debug("Trial depth: %.2f km", depth)

        # 1. recompute station geometry for this depth
        for idx, row in stat_df.iterrows():
            sta = row["station"]
            distance_deg = row["distance_deg"]

            arrivals = taupmodel.get_ray_paths(
                source_depth_in_km=depth,
                distance_in_degree=distance_deg,
                phase_list=["p", "P", "s", "S"],
            )

            got_p = False
            got_p_down = False
            down_tkoff = down_ain = None

            ivel = np.where(model[:, 1] < depth)[0][-1]

            for arr in arrivals:
                phase = arr.phase.name

                if phase == "p" and not got_p:
                    tkoff = arr.takeoff_angle
                    ain = arr.incident_angle

                    stat_df.at[idx, "takeoff"] = tkoff
                    stat_df.at[idx, "takeoff_rad"] = np.deg2rad(tkoff)
                    stat_df.at[idx, "incidence"] = ain
                    stat_df.at[idx, "incidence_rad"] = np.deg2rad(ain)

                    ainc = np.degrees(
                        np.arcsin(model[0, 0] * np.sin(np.deg2rad(tkoff)) / model[ivel, 0])
                    )

                    stat_df.at[idx, "incidence_ray"] = ainc
                    stat_df.at[idx, "P_rotation1"] = ain - tkoff
                    stat_df.at[idx, "P_rotation2"] = ainc - tkoff

                    got_p = True

                if phase == "P" and not got_p_down:
                    down_tkoff = arr.takeoff_angle
                    down_ain = arr.incident_angle
                    got_p_down = True

            if not got_p:
                logger.debug("[%s] upgoing p not found at depth %.2f km", sta, depth)
                if got_p_down:
                    stat_df.at[idx, "takeoff"] = down_tkoff
                    stat_df.at[idx, "takeoff_rad"] = np.deg2rad(down_tkoff)
                    stat_df.at[idx, "incidence"] = down_ain
                    stat_df.at[idx, "incidence_rad"] = np.deg2rad(down_ain)

                    ainc = np.degrees(
                        np.arcsin(model[0, 0] * np.sin(np.deg2rad(down_tkoff)) / model[ivel, 0])
                    )

                    stat_df.at[idx, "incidence_ray"] = ainc
                    stat_df.at[idx, "P_rotation1"] = down_ain - down_tkoff
                    stat_df.at[idx, "P_rotation2"] = ainc - down_tkoff
                else:
                    logger.debug("[%s] downgoing P not found either", sta)
                    for col in (
                        "takeoff",
                        "takeoff_rad",
                        "incidence",
                        "incidence_rad",
                        "incidence_ray",
                        "P_rotation1",
                        "P_rotation2",
                    ):
                        stat_df.at[idx, col] = None

        # 2. build new p_data and s_data for this depth
        p_data_local = PPolarityData(
            station_ids=stat_df["station"].values,
            takeoff_angles=stat_df["takeoff_rad"].values,
            azimuths=stat_df["azimuth_rad"].values,
            observed_signs=stat_df["P_sign"].values,
            mis_pick_probs=stat_df["mis_pick"].values,
        )

        if use_spol:
            df_s = stat_df[stat_df["Spol"].notna()]
            s_data_local = SPolarizationData(
                station_ids=df_s["station"].values,
                takeoff_angles=df_s["takeoff_rad"].values,
                azimuths=df_s["azimuth_rad"].values,
                obs_polarization_angle=np.deg2rad(df_s["Spol"].values),
                sigma_angle=np.deg2rad(df_s["Spol_sigma"].values),
            )
        else:
            s_data_local = SPolarizationData([], [], [], [], [])

        # 3. compute log posterior for all theta at this depth
        logp_this_depth = np.empty(n_theta)

        for i, theta in enumerate(theta_grid):
            logp_this_depth[i] = log_posterior(
                theta,
                model,
                depth,
                p_data_local,
                s_data_local,
                period_s,
                k_allowed,
                lambda_penalty,
                fmp_amp,
                w_spol,
                w_fmp=w_fmp,
                f_spol=f_spol,
            )

        log_post_all_depths.append(logp_this_depth)

    # 4. combine all depths with log-sum-exp (marginalization)
    log_post_all_depths = np.vstack(log_post_all_depths)  # (Ndepth, Ntheta)
    log_post_marg = scipy.special.logsumexp(log_post_all_depths, axis=0) - np.log(len(depth_grid))

    # 5. MAP of the marginalized posterior
    idx_max = np.argmax(log_post_marg)
    theta_best = theta_grid[idx_max]

    return theta_best, log_post_marg, theta_grid


def normalize_posterior(log_post_grid):
    lp = log_post_grid - np.max(log_post_grid)
    p = np.exp(lp)
    p /= np.sum(p)
    return p


def hpd_region(theta_grid, post, level=0.68):
    """Mechanisms (degrees) inside the highest-posterior-density region.

    Always includes at least the MAP point, even for extremely peaked posteriors.
    """
    idx_sorted = np.argsort(post)[::-1]
    cum_post = np.cumsum(post[idx_sorted])
    cutoff_idx = max(int(np.searchsorted(cum_post, level)), 1)
    idx_hpd = idx_sorted[:cutoff_idx]
    return np.degrees(theta_grid[idx_hpd])
