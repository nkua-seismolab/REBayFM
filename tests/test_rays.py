"""Tests for velocity model parsing and station geometry (rebayfm.core.rays)."""

import numpy as np
import pytest

from rebayfm.core.models import EventTask, Observation
from rebayfm.core.rays import build_station_dataframe, load_layer_model, load_taup_model

ND_PATH = "velocity_models/model_rigo1996_half.nd"

# expected Rigo model layers (Vp, layer top depth)
LEGACY_MODEL = np.array(
    [
        [4.8, 0.0],
        [5.2, 4.0],
        [5.8, 7.2],
        [6.1, 8.2],
        [6.3, 10.4],
        [6.5, 15.0],
        [7.0, 30.0],
    ]
)


def test_load_layer_model_matches_legacy():
    model = load_layer_model(ND_PATH)
    # keeps the below-interface velocity per unique depth; the trailing
    # half-space row (6371 km) is harmless for the ivel lookup
    assert np.allclose(model[:7], LEGACY_MODEL)


def test_load_layer_model_missing_file():
    with pytest.raises(FileNotFoundError):
        load_taup_model("velocity_models/nope.nd")


@pytest.fixture(scope="module")
def taupmodel():
    return load_taup_model(ND_PATH)


def _task():
    observations = [
        # station with impulsive positive P polarity
        Observation(
            network="CL",
            station="AAA",
            location="",
            channel="HHZ",
            latitude=38.35,
            longitude=22.10,
            elevation=500.0,
            polarity=1.0,
            onset="I",
        ),
        # station with emergent negative polarity and an S polarization
        Observation(
            network="CL",
            station="BBB",
            location="",
            channel="HHZ",
            latitude=38.45,
            longitude=21.95,
            elevation=200.0,
            polarity=-1.0,
            onset="E",
            spol_deg=42.0,
            spol_class="Good Split",
        ),
        # S-polarization-only station
        Observation(
            network="CL",
            station="CCC",
            location="",
            channel="HHZ",
            latitude=38.25,
            longitude=22.05,
            elevation=100.0,
            spol_deg=-10.0,
            spol_class="Good Split",
        ),
    ]
    return EventTask(
        event_id="test_event",
        origin_id="test_origin",
        origin_time="2020-06-04T02:32:44.100000",
        latitude=38.37,
        longitude=22.02,
        depth_km=8.0,
        magnitude=3.2,
        observations=observations,
    )


def test_build_station_dataframe(taupmodel):
    model = load_layer_model(ND_PATH)
    df = build_station_dataframe(
        _task(),
        taupmodel,
        model,
        mis_pick_impulsive=0.10,
        mis_pick_emergent=0.30,
        spol_sigma_deg=15.0,
    )

    assert list(df["station"]) == ["AAA", "BBB", "CCC"]
    assert list(df["polarity"]) == ["+", "-", "x"]
    assert list(df["clarity"]) == [0, 1, 0]
    assert list(df["mis_pick"]) == [0.10, 0.30, 0.10]

    assert np.allclose(df["P_sign"].values[:2], [1.0, -1.0])
    assert np.isnan(df["P_sign"].values[2])

    # local event: direct upgoing rays with takeoff > 90 deg
    assert (df["takeoff"] > 90).all()
    assert (df["Stt"] > df["Ptt"]).all()
    assert ((df["azimuth"] >= 0) & (df["azimuth"] < 360)).all()

    # S polarizations attached only where measured (upgoing rays)
    assert np.isnan(df["Spol"].values[0])
    assert df["Spol"].values[1] == 42.0
    assert df["Spol"].values[2] == -10.0
    assert df["Spol_sigma"].values[1] == 15.0
