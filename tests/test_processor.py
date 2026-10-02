"""End-to-end test of FMProcessor with a synthetic event (rebayfm.core.processor)."""

import numpy as np
import pandas as pd
import pytest

from rebayfm.config import load_config
from rebayfm.core.inversion import dist_calc
from rebayfm.core.models import EventTask, Observation
from rebayfm.core.processor import FMProcessor, sanitize_event_id
from rebayfm.core.radiation import rpgen_mgs_p

TRUE_STRIKE, TRUE_DIP, TRUE_RAKE = 40.0, 55.0, -80.0
SRC_LAT, SRC_LON, SRC_DEP = 38.37, 22.02, 8.0


def _config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "velocity_model:\n"
        "  path: velocity_models/model_rigo1996_half.nd\n"
        "bayfm:\n"
        "  strike_step: 20\n"
        "  dip_step: 10\n"
        "  rake_step: 20\n"
        "  min_polarities: 6\n"
        f"output:\n  directory: {tmp_path / 'output'}\n"
    )
    return load_config(cfg)


def _observations(processor):
    """Synthetic stations around the epicenter with true-mechanism polarities."""
    rng = np.random.default_rng(7)
    observations = []
    for i in range(14):
        az = i * (360.0 / 14) + rng.uniform(-5, 5)
        dist_km = rng.uniform(5, 25)
        dlat = dist_km / 111.19 * np.cos(np.deg2rad(az))
        dlon = dist_km / 111.19 * np.sin(np.deg2rad(az)) / np.cos(np.deg2rad(SRC_LAT))
        sta_lat, sta_lon = SRC_LAT + dlat, SRC_LON + dlon

        # polarity from the true mechanism using the actual TauP geometry
        from obspy.geodetics import gps2dist_azimuth

        _, azm, _ = gps2dist_azimuth(SRC_LAT, SRC_LON, sta_lat, sta_lon)
        _, ddeg = dist_calc(SRC_LAT, SRC_LON, sta_lat, sta_lon)
        arr = processor.taupmodel.get_travel_times(
            source_depth_in_km=SRC_DEP, distance_in_degree=ddeg, phase_list=["p"]
        )
        take = arr[0].takeoff_angle
        pol = float(np.sign(rpgen_mgs_p(TRUE_STRIKE, TRUE_DIP, TRUE_RAKE, take, azm % 360)))

        observations.append(
            Observation(
                network="CL",
                station=f"S{i:02d}",
                location="",
                channel="HHZ",
                latitude=sta_lat,
                longitude=sta_lon,
                elevation=300.0,
                polarity=pol,
                onset="I" if i % 2 == 0 else "E",
            )
        )
    return observations


@pytest.fixture(scope="module")
def outcome(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rebayfm")
    config = _config(tmp_path)
    processor = FMProcessor(config)
    task = EventTask(
        event_id="smi:local/test/event#1",
        origin_id="smi:local/test/origin",
        origin_time="2020-06-04T02:32:44.100000",
        latitude=SRC_LAT,
        longitude=SRC_LON,
        depth_km=SRC_DEP,
        magnitude=3.2,
        observations=_observations(FMProcessor(config)),
    )
    result = processor.process(task)
    return config, task, result


def test_result_ok(outcome):
    _, _, result = outcome
    assert result.ok, result.error
    assert result.processed_at
    assert result.n_polarities == 14
    assert result.strike1 is not None
    assert result.fmp_matches + result.fmp_misfits == 14
    # perfect synthetic polarities: solid match rate
    assert result.fmp_match_pct >= 80.0
    assert result.rms_kagan_deg is not None


def test_result_publication_precision(outcome):
    _, _, result = outcome
    published = (
        result.strike1,
        result.dip1,
        result.rake1,
        result.strike2,
        result.dip2,
        result.rake2,
        result.strike1_gmean,
        result.dip1_gmean,
        result.rake1_gmean,
        result.strike2_gmean,
        result.dip2_gmean,
        result.rake2_gmean,
        result.strike1_sigma,
        result.dip1_sigma,
        result.rake1_sigma,
        result.strike2_sigma,
        result.dip2_sigma,
        result.rake2_sigma,
        result.fmp_match_pct,
        result.fmp_match_weighted_pct,
        result.avg_spol_misfit_deg,
        result.rms_kagan_deg,
        result.kagan_g1_g2_deg,
    )
    for value in published:
        if value is not None:
            assert value == round(value, 2)


def test_artifacts_written(outcome):
    _, task, result = outcome
    from pathlib import Path

    outdir = Path(result.output_dir)
    evcode = sanitize_event_id(task.event_id)
    assert outdir.name == pd.Timestamp(result.processed_at).strftime("%Y%m%dT%H%M%S")
    assert outdir.parent.name == evcode

    expected = [
        f"{evcode}_beachball.png",
        f"{evcode}_all_marginal_pdf.png",
        f"{evcode}_MAP.epi",
        f"{evcode}_gmean_plane1.epi",
        f"{evcode}_gmean_plane2.epi",
        f"{evcode}_bayesian_focmec.csv",
        f"{evcode}_station_info.csv",
        f"{evcode}_focmec_info.txt",
        f"{evcode}_processing_info.txt",
    ]
    for name in expected:
        assert (outdir / name).is_file(), f"missing artifact {name}"

    focmec = pd.read_csv(outdir / f"{evcode}_bayesian_focmec.csv")
    assert len(focmec) == 1
    assert focmec["strike1_best"][0] == round(result.strike1, 1)
    assert "RMS Kagan" in focmec.columns

    stations = pd.read_csv(outdir / f"{evcode}_station_info.csv")
    assert len(stations) == 14

    epi = (outdir / f"{evcode}_MAP.epi").read_text().split()
    assert epi[0] == "2020"


def test_min_polarities_gate(tmp_path):
    config = _config(tmp_path)
    processor = FMProcessor(config)
    task = EventTask(
        event_id="ev_too_few",
        origin_id="orig",
        origin_time="2020-06-04T02:32:44",
        latitude=SRC_LAT,
        longitude=SRC_LON,
        depth_km=SRC_DEP,
        observations=[],
    )
    result = processor.process(task)
    assert not result.ok
    assert "min_polarities" in result.error


def test_sanitize_event_id():
    assert sanitize_event_id("smi:local/ev#1 a") == "smi_local_ev_1_a"
    assert sanitize_event_id("gfz2020abcd") == "gfz2020abcd"
