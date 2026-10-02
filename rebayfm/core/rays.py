"""Velocity model handling and station ray geometry.

The TauP model is built from a .nd file on first use.
The same .nd file also provides the (Vp, layer top depth) array used for the
S-wave polarization rotation correction.

Station geometry uses azimuths via obspy gps2dist_azimuth, haversine distances,
TauP takeoff/incidence angles
(upgoing preferred, downgoing fallback) and Snell-refracted surface incidence.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model

from rebayfm.core.inversion import dist_calc
from rebayfm.core.models import EventTask

logger = logging.getLogger(__name__)


def load_taup_model(model_path: str | Path) -> TauPyModel:
    """Load a TauP model from a .nd file, building the .npz next to it if needed.

    If the model directory is read-only (e.g. Docker volume), the .npz is
    built in a temporary directory instead.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Velocity model file not found: {path}")

    npz_path = path.with_suffix(".npz")
    if not npz_path.exists():
        logger.info("Building TauP model from %s", path)
        try:
            build_taup_model(str(path), str(path.parent), verbose=False)
        except OSError:
            tmpdir = Path(tempfile.gettempdir())
            logger.info("Model directory not writable, building TauP model in %s", tmpdir)
            build_taup_model(str(path), str(tmpdir), verbose=False)
            npz_path = tmpdir / npz_path.name

    return TauPyModel(str(npz_path))


def load_layer_model(model_path: str | Path) -> np.ndarray:
    """(Vp, layer top depth) array from a .nd file, as expected by the science code.

    For each interface depth the velocity of the layer below is kept.
    """
    path = Path(model_path)
    layers: dict[float, float] = {}
    with open(path) as fid:
        for line in fid:
            parts = line.split()
            if not parts:
                continue
            try:
                depth = float(parts[0])
                vp = float(parts[1])
            except (ValueError, IndexError):
                continue  # named discontinuity or comment line
            layers[depth] = vp  # last value at each depth wins (below interface)

    if not layers:
        raise ValueError(f"No layers parsed from velocity model file: {path}")

    model = np.array([[vp, depth] for depth, vp in sorted(layers.items())])
    return model


def _polarity_char(polarity: float | None) -> str:
    if polarity is None:
        return "x"
    return "+" if polarity > 0 else "-"


def build_station_dataframe(
    task: EventTask,
    taupmodel: TauPyModel,
    model: np.ndarray,
    mis_pick_impulsive: float,
    mis_pick_emergent: float,
    spol_sigma_deg: float,
    log=logger.info,
) -> pd.DataFrame:
    """Per-station geometry/observation table.

    Provides columns for inversion data, depth-range search, beachball plots
    and station CSV output.
    """
    src_lat = task.latitude
    src_lon = task.longitude
    src_dep = task.depth_km

    rows = []
    for obs in task.observations:
        polarity = _polarity_char(obs.polarity)
        # clarity: 0 impulsive, 1 emergent (missing onset counts as emergent);
        # dummy 0 for stations without a P polarity
        if polarity == "x":
            clarity = 0
        else:
            clarity = 0 if obs.onset == "I" else 1
        rows.append(
            {
                "station": obs.station,
                "network": obs.network,
                "polarity": polarity,
                "clarity": clarity,
                "sta_lat": obs.latitude,
                "sta_lon": obs.longitude,
                "sta_elev": obs.elevation,
                "spol_meas": obs.spol_deg,
                "spol_class": obs.spol_class,
            }
        )

    df = pd.DataFrame(rows)

    lats = []
    lons = []
    elevs = []
    dists = []
    distdegs = []
    azms = []
    tkos = []  # take-off angles (calculated)
    ains = []  # angle of incidence at the surface (calculated)
    ptts = []  # P travel times
    stts = []  # S travel times
    ainrays = []  # incidence angle from Snell refraction
    ainrot1s = []  # source->surface P rotation from taup incidence
    ainrot2s = []  # source->surface P rotation from Snell incidence
    spols = []  # S-polarization measurements
    spolgrs = []  # S-polarization classes

    ivel = np.where(model[:, 1] < src_dep)[0][-1]

    for row in df.itertuples(index=False):
        sta = row.station
        sta_lat = row.sta_lat
        sta_lon = row.sta_lon
        sta_elev = row.sta_elev

        lats.append(sta_lat)
        lons.append(sta_lon)
        elevs.append(sta_elev)

        # Forward azimuth (source -> station)
        _, az, _ = gps2dist_azimuth(src_lat, src_lon, sta_lat, sta_lon)
        azms.append(az % 360)

        distance_km, distance_deg = dist_calc(src_lat, src_lon, sta_lat, sta_lon)
        dists.append(distance_km)
        distdegs.append(distance_deg)

        arrivals = taupmodel.get_travel_times(
            source_depth_in_km=src_dep,
            distance_in_degree=distance_deg,
            phase_list=["p", "P", "s", "S"],
        )

        got_p = False
        got_s = False
        got_p_down = False
        got_s_down = False
        tkoff = None
        down_tkoff = down_ain = down_ainc = None
        down_rot1 = down_rot2 = down_parr = down_sarr = None

        for arr in arrivals:
            phase = arr.phase.name

            if phase == "p" and not got_p:
                tkoff = arr.takeoff_angle
                ain = arr.incident_angle

                tkos.append(tkoff)
                ains.append(ain)

                ainc = np.degrees(
                    np.arcsin(model[0, 0] * np.sin(np.deg2rad(tkoff)) / model[ivel, 0])
                )
                ainrays.append(ainc)
                ainrot1s.append(ain - tkoff)
                ainrot2s.append(ainc - tkoff)

                ptts.append(arr.time)
                got_p = True

            if phase == "s" and not got_s:
                stts.append(arr.time)
                got_s = True

            if phase == "P" and not got_p_down:
                down_tkoff = arr.takeoff_angle
                down_ain = arr.incident_angle
                down_ainc = np.degrees(
                    np.arcsin(model[0, 0] * np.sin(np.deg2rad(down_tkoff)) / model[ivel, 0])
                )
                down_rot1 = down_ain - down_tkoff
                down_rot2 = down_ainc - down_tkoff
                down_parr = arr.time
                got_p_down = True

            if phase == "S" and not got_s_down:
                down_sarr = arr.time
                got_s_down = True

        if not got_p:
            log(f"[{sta}] upgoing p not found")
            if got_p_down:
                tkoff = down_tkoff  # to check with Spol
                tkos.append(down_tkoff)
                ains.append(down_ain)
                ainrays.append(down_ainc)
                ainrot1s.append(down_rot1)
                ainrot2s.append(down_rot2)
                ptts.append(down_parr)
            else:
                log(f"[{sta}] downgoing P not found either!")
                tkoff = None
                tkos.append(None)
                ains.append(None)
                ainrays.append(None)
                ainrot1s.append(None)
                ainrot2s.append(None)
                ptts.append(None)

        if not got_s:
            log(f"[{sta}] upgoing s not found")
            if got_s_down:
                stts.append(down_sarr)
            else:
                log(f"[{sta}] downgoing S not found either!")
                stts.append(None)

        # S-polarization only attached to upgoing rays
        if tkoff is not None and tkoff > 90 and row.spol_meas is not None:
            spols.append(row.spol_meas)
            spolgrs.append(row.spol_class)
        else:
            spols.append(None)
            spolgrs.append(None)

    df["latitude"] = lats
    df["longitude"] = lons
    df["elevation"] = elevs
    df["azimuth"] = azms
    df["distance"] = dists
    df["distance_deg"] = distdegs
    df["takeoff"] = tkos
    df["incidence"] = ains
    df["incidence_ray"] = ainrays
    df["P_rotation1"] = ainrot1s
    df["P_rotation2"] = ainrot2s
    df["Ptt"] = ptts
    df["Stt"] = stts

    df["Spol"] = spols
    df["Spol_grade"] = spolgrs
    df["Spol_sigma"] = np.where(df["Spol"].notna(), spol_sigma_deg, np.nan)

    df["P_sign"] = df["polarity"].map({"+": 1, "-": -1}).astype(float)

    df["takeoff_rad"] = np.deg2rad(df["takeoff"].astype(float))
    df["azimuth_rad"] = np.deg2rad(df["azimuth"].astype(float))
    df["incidence_rad"] = np.deg2rad(df["incidence_ray"].astype(float))

    df["mis_pick"] = df["clarity"].map({1: mis_pick_emergent, 0: mis_pick_impulsive})

    df = df.drop(columns=["sta_lat", "sta_lon", "sta_elev", "spol_meas", "spol_class"])

    return df
