"""Focal mechanism processor: orchestration of the Bayesian inversion.

Produces the following artifacts per event:

    output/<event_id>/<processed_at>/
        <evcode>_beachball.png
        <evcode>_all_marginal_pdf.png
        <evcode>_MAP.epi
        <evcode>_gmean_plane1.epi
        <evcode>_gmean_plane2.epi
        <evcode>_bayesian_focmec.csv
        <evcode>_station_info.csv
        <evcode>_focmec_info.txt
        <evcode>_processing_info.txt

Reference (.mec / HASH .fps) comparisons are not included because they are
unavailable in real time. The output directory carries the processing timestamp
so reprocessing never overwrites earlier results.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import rebayfm
from rebayfm.config import Config
from rebayfm.core.inversion import (
    PPolarityData,
    SPolarizationData,
    grid_search_map,
    grid_search_map_deprange,
    hpd_region,
    normalize_posterior,
)
from rebayfm.core.kagan import calc_theta
from rebayfm.core.models import EventTask, FMResult
from rebayfm.core.plots import plot_all_marginals, plot_full_beachball_hpd
from rebayfm.core.radiation import (
    f_dec_clvd,
    momtens,
    rmech_fast,
    sanitize_dip,
    sanitize_rake,
    second_plane,
)
from rebayfm.core.rays import build_station_dataframe, load_layer_model, load_taup_model
from rebayfm.core.solution import (
    fit_marginal_mix2,
    fit_marginal_single,
    marginal_1d,
    mean_moment_tensor,
    split_posterior_by_strike,
)

logger = logging.getLogger(__name__)

_EVCODE_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_event_id(event_id: str) -> str:
    """Filesystem-safe event code derived from the SeisComP public ID."""
    return _EVCODE_UNSAFE.sub("_", event_id)


class _RunLog:
    """Collects processing-log lines and mirrors them to the logger."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.lines: list[str] = []

    def __call__(self, msg=""):
        text = str(msg)
        self.lines.append(text)
        logger.debug("[%s] %s", self.prefix, text)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


class FMProcessor:
    """Runs the Bayesian focal mechanism inversion for one event at a time."""

    def __init__(self, config: Config):
        self.config = config
        self.taupmodel = load_taup_model(config.velocity_model.path)
        self.model = load_layer_model(config.velocity_model.path)
        Path(config.output.directory).mkdir(parents=True, exist_ok=True)

    def process(self, task: EventTask) -> FMResult:
        result = FMResult(event_id=task.event_id, origin_id=task.origin_id)
        try:
            self._process(task, result)
        except Exception as exc:
            logger.exception("Processing failed for event %s", task.event_id)
            result.error = f"{type(exc).__name__}: {exc}"
        return result

    def process_batch(self, tasks: list[EventTask]) -> dict[str, FMResult]:
        return {task.event_id: self.process(task) for task in tasks}

    # ------------------------------------------------------------------

    def _process(self, task: EventTask, result: FMResult) -> None:
        cfg = self.config.bayfm

        n_pol = sum(1 for obs in task.observations if obs.polarity is not None)
        result.n_polarities = n_pol
        if n_pol < cfg.min_polarities:
            raise ValueError(
                f"Only {n_pol} P polarities available (min_polarities = {cfg.min_polarities})"
            )

        processed = datetime.now(UTC)
        result.processed_at = processed.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        evcode = sanitize_event_id(task.event_id)
        outdir = Path(self.config.output.directory) / evcode / processed.strftime("%Y%m%dT%H%M%S")
        outdir.mkdir(parents=True, exist_ok=True)
        result.output_dir = str(outdir)

        log = _RunLog(evcode)
        log(f"Event: {task.event_id} (origin {task.origin_id})")
        log(f"Origin time: {task.origin_time}")
        log(
            f"Hypocenter: lat={task.latitude:.4f} lon={task.longitude:.4f} "
            f"depth={task.depth_km:.1f} km, M={task.magnitude}"
        )
        log(" ")

        src_dep = task.depth_km

        # ------------------------------------------------------------------
        # Station geometry / observation table
        # ------------------------------------------------------------------
        df = build_station_dataframe(
            task,
            self.taupmodel,
            self.model,
            mis_pick_impulsive=cfg.mis_pick_impulsive,
            mis_pick_emergent=cfg.mis_pick_emergent,
            spol_sigma_deg=cfg.spol_sigma_deg,
            log=log,
        )

        n_no_ray = int(df["takeoff"].isna().sum())
        if n_no_ray:
            log(f"! Dropping {n_no_ray} station(s) without a TauP ray")
            df = df[df["takeoff"].notna()].reset_index(drop=True)

        # ------------------------------------------------------------------
        # Parameter grids (degrees -> radians)
        # ------------------------------------------------------------------
        if cfg.use_strike360:
            strike_grid = np.deg2rad(np.arange(0, 360, cfg.strike_step))  # [0, 360)
        else:
            strike_grid = np.deg2rad(np.arange(0, 180 + cfg.strike_step, cfg.strike_step))
        dip_grid = np.deg2rad(
            np.vectorize(sanitize_dip)(np.arange(5, 90 + cfg.dip_step, cfg.dip_step))
        )
        rake_grid = np.deg2rad(np.vectorize(sanitize_rake)(np.arange(-180, 180, cfg.rake_step)))

        # ------------------------------------------------------------------
        # Inversion inputs
        # ------------------------------------------------------------------
        p_data = PPolarityData(
            station_ids=df["station"].values,
            takeoff_angles=df["takeoff_rad"].values,
            azimuths=df["azimuth_rad"].values,
            observed_signs=df["P_sign"].values,
            mis_pick_probs=df["mis_pick"].values,
        )

        df_s = df[df["Spol"].notna()]
        if cfg.use_spol:
            s_data = SPolarizationData(
                station_ids=df_s["station"].values,
                takeoff_angles=df_s["takeoff_rad"].values,
                azimuths=df_s["azimuth_rad"].values,
                obs_polarization_angle=np.deg2rad(df_s["Spol"].values.astype(float)),
                sigma_angle=np.deg2rad(df_s["Spol_sigma"].values.astype(float)),
            )
        else:
            s_data = SPolarizationData([], [], [], [], [])

        # ------------------------------------------------------------------
        # Grid search
        # ------------------------------------------------------------------
        if cfg.use_depth_range:
            theta_best_rad, log_post_grid, theta_grid = grid_search_map_deprange(
                self.taupmodel,
                df,
                cfg.use_spol,
                strike_grid,
                dip_grid,
                rake_grid,
                self.model,
                src_dep,
                p_data=p_data,
                s_data=s_data,
                period_s=np.pi,
                k_allowed=cfg.k_allowed,
                lambda_penalty=cfg.lambda_penalty,
                depth_range_km=cfg.depth_range_km,
                depth_step_km=cfg.depth_step_km,
                fmp_amp=cfg.fmp_amp,
                w_spol=cfg.w_spol,
                w_fmp=cfg.w_fmp,
                f_spol=cfg.f_spol,
            )
        else:
            theta_best_rad, log_post_grid, theta_grid = grid_search_map(
                strike_grid,
                dip_grid,
                rake_grid,
                self.model,
                src_dep,
                p_data=p_data,
                s_data=s_data,
                period_s=np.pi,
                k_allowed=cfg.k_allowed,
                lambda_penalty=cfg.lambda_penalty,
                fmp_amp=cfg.fmp_amp,
                w_spol=cfg.w_spol,
                w_fmp=cfg.w_fmp,
                f_spol=cfg.f_spol,
            )

        post = normalize_posterior(log_post_grid)

        # Highest-posterior-density region
        theta_hpd_deg = hpd_region(theta_grid, post, level=cfg.hpd_level)

        strike_best, dip_best, rake_best = np.rad2deg(theta_best_rad)
        log(" ")
        log(
            f"MAP solution (deg): strike={strike_best:.1f}, "
            f"dip={dip_best:.1f}, rake={rake_best:.1f}"
        )

        ns, nd, nr = len(strike_grid), len(dip_grid), len(rake_grid)
        post_3d = post.reshape(ns, nd, nr)

        strike_marginal = marginal_1d(post_3d, "strike")
        dip_marginal = marginal_1d(post_3d, "dip")
        rake_marginal = marginal_1d(post_3d, "rake")

        strike_best_marg = np.rad2deg(strike_grid[np.argmax(strike_marginal)])
        dip_best_marg = np.rad2deg(dip_grid[np.argmax(dip_marginal)])
        rake_best_marg = np.rad2deg(rake_grid[np.argmax(rake_marginal)])
        log("Marginal MAP estimates (deg):")
        log(f"  strike_marg = {strike_best_marg:.1f}")
        log(f"  dip_marg    = {dip_best_marg:.1f}")
        log(f"  rake_marg   = {rake_best_marg:.1f}")

        if cfg.select_marginal_map:
            strike_best, dip_best, rake_best = strike_best_marg, dip_best_marg, rake_best_marg

        # ------------------------------------------------------------------
        # Mean mechanism (probability-weighted moment tensor, DC part)
        # ------------------------------------------------------------------
        mt_mean = mean_moment_tensor(theta_grid, post)
        mdc_mean, _, _ = f_dec_clvd(mt_mean)
        fps, _, _, _, _, _, _ = rmech_fast(mdc_mean)
        strike_mean, dip_mean, rake_mean = fps[0]

        strike_best1, dip_best1, rake_best1 = strike_best, dip_best, rake_best
        strike_best2, dip_best2, rake_best2 = second_plane(strike_best, dip_best, rake_best)
        strike_mean1, dip_mean1, rake_mean1 = strike_mean, dip_mean, rake_mean
        strike_mean2, dip_mean2, rake_mean2 = second_plane(strike_mean, dip_mean, rake_mean)

        theta_best2 = [strike_best1, dip_best1, rake_best1, strike_best2, dip_best2, rake_best2]
        theta_mean2 = [strike_mean1, dip_mean1, rake_mean1, strike_mean2, dip_mean2, rake_mean2]

        # ------------------------------------------------------------------
        # Two-component Gaussian fits of the full marginals
        # ------------------------------------------------------------------
        (
            a1_strike_free,
            sigma1_strike_free,
            a2_strike_free,
            sigma2_strike_free,
            mu1_strike_free,
            mu2_strike_free,
            sigma1_strike_fix,
            sigma2_strike_fix,
        ) = fit_marginal_mix2(strike_marginal, strike_grid, "strike", strike_best1, strike_best2)
        (
            _,
            sigma1_dip_free,
            _,
            sigma2_dip_free,
            mu1_dip_free,
            mu2_dip_free,
            sigma1_dip_fix,
            sigma2_dip_fix,
        ) = fit_marginal_mix2(dip_marginal, dip_grid, "dip", dip_best1, dip_best2)
        (
            _,
            sigma1_rake_free,
            _,
            sigma2_rake_free,
            mu1_rake_free,
            mu2_rake_free,
            sigma1_rake_fix,
            sigma2_rake_fix,
        ) = fit_marginal_mix2(rake_marginal, rake_grid, "rake", rake_best1, rake_best2)

        # Isolate the dip/rake solutions of each plane's strike
        post_3d_plane1, post_3d_plane2 = split_posterior_by_strike(
            post_3d,
            strike_grid,
            a1_strike_free,
            mu1_strike_free,
            sigma1_strike_free,
            a2_strike_free,
            mu2_strike_free,
            sigma2_strike_free,
        )

        # ------------------------------------------------------------------
        # Beachball figure + misfit statistics
        # ------------------------------------------------------------------
        plot_spol_vectors = bool(df["Spol"].apply(lambda v: pd.notna(v) and np.isreal(v)).any())

        bball_log = _RunLog(evcode)
        bball_path = outdir / f"{evcode}_beachball.png"
        (
            rms_kagan,
            ppol_matches,
            ppol_misfits,
            ppol_matches_score,
            ppol_misfits_score,
            len_spol,
            avg_spol_misfits,
        ) = plot_full_beachball_hpd(
            df,
            strike_best,
            dip_best,
            rake_best,
            self.model,
            src_dep,
            evcode,
            theta_hpd_deg,
            plot_spol_vectors,
            save_path=bball_path if self.config.output.save_figures else None,
            log=bball_log,
        )
        (outdir / f"{evcode}_focmec_info.txt").write_text(bball_log.text())

        # ------------------------------------------------------------------
        # Report on results
        # ------------------------------------------------------------------
        log(
            f"MAP solution (best): s/d/r #1 : "
            f"{strike_best1:.1f}/{dip_best1:.1f}/{rake_best1:.1f} , "
            f"s/d/r #2: {strike_best2:.1f}/{dip_best2:.1f}/{rake_best2:.1f}"
        )

        log(" ")
        log("# Bounds based on Gaussian Mixture [FIXED mu at MAP solutions]:")
        log(" ")
        log(
            f"Strike1: mu = {strike_best1:.1f} , sigma = {sigma1_strike_fix:.1f}  "
            f"[{strike_best1 - sigma1_strike_fix:.1f} - {strike_best1 + sigma1_strike_fix:.1f}]"
        )
        log(
            f"Dip1: mu = {dip_best1:.1f} , sigma = {sigma1_dip_fix:.1f}  "
            f"[{dip_best1 - sigma1_dip_fix:.1f} - {dip_best1 + sigma1_dip_fix:.1f}]"
        )
        log(
            f"Rake1: mu = {rake_best1:.1f} , sigma = {sigma1_rake_fix:.1f}  "
            f"[{rake_best1 - sigma1_rake_fix:.1f} - {rake_best1 + sigma1_rake_fix:.1f}]"
        )
        log(",")
        log(
            f"Strike2: mu = {strike_best2:.1f} , sigma = {sigma2_strike_fix:.1f}  "
            f"[{strike_best2 - sigma2_strike_fix:.1f} - {strike_best2 + sigma2_strike_fix:.1f}]"
        )
        log(
            f"Dip2: mu = {dip_best2:.1f} , sigma = {sigma2_dip_fix:.1f}  "
            f"[{dip_best2 - sigma2_dip_fix:.1f} - {dip_best2 + sigma2_dip_fix:.1f}]"
        )
        log(
            f"Rake2: mu = {rake_best2:.1f} , sigma = {sigma2_rake_fix:.1f}  "
            f"[{rake_best2 - sigma2_rake_fix:.1f} - {rake_best2 + sigma2_rake_fix:.1f}]"
        )

        log(" ")
        log("# Bounds based on Gaussian Mixture [FREE mu, UNKNOWN s/d/r combination]:")
        log(
            f"(free) Strike1: mu = {mu1_strike_free:.1f} , sigma = {sigma1_strike_free:.1f}  "
            f"[{mu1_strike_free - sigma1_strike_free:.1f} - "
            f"{mu1_strike_free + sigma1_strike_free:.1f}]"
        )
        log(
            f"(free) Strike2: mu = {mu2_strike_free:.1f} , sigma = {sigma2_strike_free:.1f}  "
            f"[{mu2_strike_free - sigma2_strike_free:.1f} - "
            f"{mu2_strike_free + sigma2_strike_free:.1f}]"
        )
        log(",")
        log(
            f"(free) Dip1: mu = {mu1_dip_free:.1f} , sigma = {sigma1_dip_free:.1f}  "
            f"[{mu1_dip_free - sigma1_dip_free:.1f} - {mu1_dip_free + sigma1_dip_free:.1f}]"
        )
        log(
            f"(free) Dip2: mu = {mu2_dip_free:.1f} , sigma = {sigma2_dip_free:.1f}  "
            f"[{mu2_dip_free - sigma2_dip_free:.1f} - {mu2_dip_free + sigma2_dip_free:.1f}]"
        )
        log(",")
        log(
            f"(free) Rake1: mu = {mu1_rake_free:.1f} , sigma = {sigma1_rake_free:.1f}  "
            f"[{mu1_rake_free - sigma1_rake_free:.1f} - {mu1_rake_free + sigma1_rake_free:.1f}]"
        )
        log(
            f"(free) Rake2: mu = {mu2_rake_free:.1f} , sigma = {sigma2_rake_free:.1f}  "
            f"[{mu2_rake_free - sigma2_rake_free:.1f} - {mu2_rake_free + sigma2_rake_free:.1f}]"
        )

        # ------------------------------------------------------------------
        # Single Gaussian fits per plane-separated posterior
        # ------------------------------------------------------------------
        gs_fits = {"plane1": {}, "plane2": {}}
        plane_mus = {
            "plane1": (strike_best1, dip_best1, rake_best1, post_3d_plane1),
            "plane2": (strike_best2, dip_best2, rake_best2, post_3d_plane2),
        }
        grids = {"strike": strike_grid, "dip": dip_grid, "rake": rake_grid}
        for plane, (mu_s, mu_d, mu_r, post_plane) in plane_mus.items():
            for axis, mu_fixed in zip(("strike", "dip", "rake"), (mu_s, mu_d, mu_r)):
                a_free, mu_free, sigma_free, sigma_fix = fit_marginal_single(
                    marginal_1d(post_plane, axis), grids[axis], axis, mu_fixed
                )
                gs_fits[plane][axis] = {
                    "A_free": a_free,
                    "mu_free": mu_free,
                    "sigma_free": sigma_free,
                    "sigma_fix": sigma_fix,
                }

        strike_g1 = gs_fits["plane1"]["strike"]["mu_free"]
        dip_g1 = gs_fits["plane1"]["dip"]["mu_free"]
        rake_g1 = gs_fits["plane1"]["rake"]["mu_free"]
        strike_g2 = gs_fits["plane2"]["strike"]["mu_free"]
        dip_g2 = gs_fits["plane2"]["dip"]["mu_free"]
        rake_g2 = gs_fits["plane2"]["rake"]["mu_free"]

        strike_s1 = gs_fits["plane1"]["strike"]["sigma_free"]
        dip_s1 = gs_fits["plane1"]["dip"]["sigma_free"]
        rake_s1 = gs_fits["plane1"]["rake"]["sigma_free"]
        strike_s2 = gs_fits["plane2"]["strike"]["sigma_free"]
        dip_s2 = gs_fits["plane2"]["dip"]["sigma_free"]
        rake_s2 = gs_fits["plane2"]["rake"]["sigma_free"]

        # Dip ranges clamped to [0, 90]
        dip1_min = max(0, dip_g1 - dip_s1)
        dip1_max = min(90, dip_g1 + dip_s1)
        dip2_min = max(0, dip_g2 - dip_s2)
        dip2_max = min(90, dip_g2 + dip_s2)

        log(" ")
        log("# Bounds based on Single Gaussians after selecting one plane at a time:")
        log(f"Strike1 / Dip1 / Rake1: {strike_best1:.1f} / {dip_best1:.1f} / {rake_best1:.1f}")
        log(
            f"Strike1: mu = {strike_g1:.1f} , sigma = {strike_s1:.1f}  "
            f"[{strike_g1 - strike_s1:.1f} - {strike_g1 + strike_s1:.1f}]"
        )
        log(f"Dip1: mu = {dip_g1:.1f} , sigma = {dip_s1:.1f}  [{dip1_min:.1f} - {dip1_max:.1f}]")
        log(
            f"Rake1: mu = {rake_g1:.1f} , sigma = {rake_s1:.1f}  "
            f"[{rake_g1 - rake_s1:.1f} - {rake_g1 + rake_s1:.1f}]"
        )
        log(",")
        log(f"Strike2 / Dip2 / Rake2: {strike_best2:.1f} / {dip_best2:.1f} / {rake_best2:.1f}")
        log(
            f"Strike2: mu = {strike_g2:.1f} , sigma = {strike_s2:.1f}  "
            f"[{strike_g2 - strike_s2:.1f} - {strike_g2 + strike_s2:.1f}]"
        )
        log(f"Dip2: mu = {dip_g2:.1f} , sigma = {dip_s2:.1f}  [{dip2_min:.1f} - {dip2_max:.1f}]")
        log(
            f"Rake2: mu = {rake_g2:.1f} , sigma = {rake_s2:.1f}  "
            f"[{rake_g2 - rake_s2:.1f} - {rake_g2 + rake_s2:.1f}]"
        )

        # ------------------------------------------------------------------
        # Marginal PDF figure
        # ------------------------------------------------------------------
        if self.config.output.save_figures:
            plotparams = {"use_strike360": cfg.use_strike360, "cmap": "magma_r"}
            plot_all_marginals(
                post_3d,
                theta_best2,
                theta_mean2,
                strike_grid,
                dip_grid,
                rake_grid,
                plotparams,
                gs_fits,
                save_path=outdir / f"{evcode}_all_marginal_pdf.png",
                evcode=evcode,
            )

        # ------------------------------------------------------------------
        # Kagan angle between the two gmean planes
        # ------------------------------------------------------------------
        m_g1 = momtens(1, strike_g1, dip_g1, rake_g1)
        m_g2 = momtens(1, strike_g2, dip_g2, rake_g2)
        kagan_g1g2 = calc_theta(m_g1, m_g2)

        # ------------------------------------------------------------------
        # Text/CSV artifacts
        # ------------------------------------------------------------------
        ot = datetime.fromisoformat(task.origin_time.replace("Z", "+00:00"))
        year, month, day = ot.year, ot.month, ot.day
        hour, minute = ot.hour, ot.minute
        sec = ot.second + ot.microsecond / 1e6
        mag = float(task.magnitude) if task.magnitude is not None else float("nan")
        src_lat, src_lon = task.latitude, task.longitude

        epi_prefix = (
            f"{year:4d} {month:2d} {day:2d} "
            f"{hour:2d} {minute:2d} {sec:6.3f}  "
            f"{src_lat:7.4f} {src_lon:9.4f} "
            f"{src_dep:5.1f} {mag:3.1f} "
        )
        epiline1 = (
            epi_prefix
            + f"{strike_best1:5.1f} {dip_best1:4.1f} {rake_best1:6.1f} "
            + f"{0:8.2E}  "
            + f"{strike_best2:5.1f} {dip_best2:4.1f} {rake_best2:6.1f}"
        )
        epiline2 = epi_prefix + f"{strike_g1:5.1f} {dip_g1:4.1f} {rake_g1:6.1f}"
        epiline3 = epi_prefix + f"{strike_g2:5.1f} {dip_g2:4.1f} {rake_g2:6.1f}"

        (outdir / f"{evcode}_MAP.epi").write_text(epiline1 + "\n")
        (outdir / f"{evcode}_gmean_plane1.epi").write_text(epiline2 + "\n")
        (outdir / f"{evcode}_gmean_plane2.epi").write_text(epiline3 + "\n")

        n_fmp = ppol_matches + ppol_misfits
        n_fmp_w = ppol_matches_score + ppol_misfits_score

        row = {
            "evcode": evcode,
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "min": minute,
            "sec": sec,
            "lat": src_lat,
            "lon": src_lon,
            "depth": src_dep,
            "mag": mag,
            "strike1_best": round(strike_best1, 1),
            "dip1_best": round(dip_best1, 1),
            "rake1_best": round(rake_best1, 1),
            "strike2_best": round(strike_best2, 1),
            "dip2_best": round(dip_best2, 1),
            "rake2_best": round(rake_best2, 1),
            "strike1_gmean": round(strike_g1, 1),
            "dip1_gmean": round(dip_g1, 1),
            "rake1_gmean": round(rake_g1, 1),
            "strike2_gmean": round(strike_g2, 1),
            "dip2_gmean": round(dip_g2, 1),
            "rake2_gmean": round(rake_g2, 1),
            "strike1_sigma": round(strike_s1, 1),
            "dip1_sigma": round(dip_s1, 1),
            "rake1_sigma": round(rake_s1, 1),
            "strike2_sigma": round(strike_s2, 1),
            "dip2_sigma": round(dip_s2, 1),
            "rake2_sigma": round(rake_s2, 1),
            "strike1_min": round(strike_g1 - strike_s1, 1),
            "strike1_max": round(strike_g1 + strike_s1, 1),
            "dip1_min": round(dip1_min, 1),
            "dip1_max": round(dip1_max, 1),
            "rake1_min": round(rake_g1 - rake_s1, 1),
            "rake1_max": round(rake_g1 + rake_s1, 1),
            "strike2_min": round(strike_g2 - strike_s2, 1),
            "strike2_max": round(strike_g2 + strike_s2, 1),
            "dip2_min": round(dip2_min, 1),
            "dip2_max": round(dip2_max, 1),
            "rake2_min": round(rake_g2 - rake_s2, 1),
            "rake2_max": round(rake_g2 + rake_s2, 1),
            "RMS Kagan": round(rms_kagan, 2),
            "Num FMPs": n_fmp,
            "FMP Matches": ppol_matches,
            "FMP Misfits": ppol_misfits,
            "FMP match perc": round(ppol_matches / n_fmp, 1) if n_fmp else np.nan,
            "FMP Matches wgt": ppol_matches_score,
            "FMP Misfits wgt": ppol_misfits_score,
            "FMP match wgt perc": (round(ppol_matches_score / n_fmp_w, 1) if n_fmp_w else np.nan),
            "Num Spol": len_spol,
            "Avg Spol Misfit": avg_spol_misfits,
            "kagan g1-g2": kagan_g1g2,
        }

        pd.DataFrame([row]).to_csv(outdir / f"{evcode}_bayesian_focmec.csv", index=False)
        df.to_csv(outdir / f"{evcode}_station_info.csv", index=False)

        header = [
            f"REBayFM v{rebayfm.__version__}",
            f"Processed at: {result.processed_at}",
            f"Event: {task.event_id}",
            f"Origin: {task.origin_id}",
            f"Velocity model: {self.config.velocity_model.path}",
            "",
            "# Configuration (bayfm):",
        ]
        header += [f"  {key}: {value}" for key, value in asdict(cfg).items()]
        header += ["", "# Processing log:", ""]
        (outdir / f"{evcode}_processing_info.txt").write_text("\n".join(header) + log.text())

        # ------------------------------------------------------------------
        # Result (plain floats, rounded to publication precision: 0.01 deg
        # is far below the grid/fit resolution)
        # ------------------------------------------------------------------
        def deg2(value):
            return round(float(value), 2)

        result.strike1, result.dip1, result.rake1 = (
            deg2(strike_best1),
            deg2(dip_best1),
            deg2(rake_best1),
        )
        result.strike2, result.dip2, result.rake2 = (
            deg2(strike_best2),
            deg2(dip_best2),
            deg2(rake_best2),
        )
        result.strike1_gmean, result.dip1_gmean, result.rake1_gmean = (
            deg2(strike_g1),
            deg2(dip_g1),
            deg2(rake_g1),
        )
        result.strike2_gmean, result.dip2_gmean, result.rake2_gmean = (
            deg2(strike_g2),
            deg2(dip_g2),
            deg2(rake_g2),
        )
        result.strike1_sigma, result.dip1_sigma, result.rake1_sigma = (
            deg2(strike_s1),
            deg2(dip_s1),
            deg2(rake_s1),
        )
        result.strike2_sigma, result.dip2_sigma, result.rake2_sigma = (
            deg2(strike_s2),
            deg2(dip_s2),
            deg2(rake_s2),
        )
        result.fmp_matches = ppol_matches
        result.fmp_misfits = ppol_misfits
        result.fmp_match_pct = round(100.0 * ppol_matches / n_fmp, 2) if n_fmp else None
        result.fmp_matches_weighted = ppol_matches_score
        result.fmp_misfits_weighted = ppol_misfits_score
        result.fmp_match_weighted_pct = (
            round(100.0 * ppol_matches_score / n_fmp_w, 2) if n_fmp_w else None
        )
        result.n_spol = len_spol
        result.avg_spol_misfit_deg = (
            deg2(avg_spol_misfits) if np.isfinite(avg_spol_misfits) else None
        )
        result.rms_kagan_deg = deg2(rms_kagan) if np.isfinite(rms_kagan) else None
        result.kagan_g1_g2_deg = deg2(kagan_g1g2)
