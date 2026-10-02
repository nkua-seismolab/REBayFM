"""Wires the scclient listener to the core focal mechanism processor.

dispatch runs on the main thread (SeisComP objects), the processor's
process_batch runs in the worker process (core objects only), and
finalize runs on the main thread again.
"""

from __future__ import annotations

import logging

from seiscomp import datamodel

from rebayfm import __version__ as rebayfm_version
from rebayfm.config import Config
from rebayfm.core.models import EventTask, Observation
from rebayfm.scclient import OUTPUT_GROUP, fm_io, pick_io

logger = logging.getLogger(__name__)


def _has_focal_mechanism(event) -> bool:
    try:
        return bool(event.preferredFocalMechanismID())
    except Exception:
        return False


def _preferred_magnitude(event) -> float | None:
    try:
        magnitude = datamodel.Magnitude.Find(event.preferredMagnitudeID())
        if magnitude is None:
            return None
        return magnitude.magnitude().value()
    except Exception:
        return None


def _accept_spol_class(quality_class: str | None, config: Config) -> bool:
    """RESS quality classes accepted as S polarization observations."""
    if not config.bayfm.use_spol:
        return False
    if quality_class == "Good Split":
        return True
    if quality_class == "Good Null":
        return config.bayfm.use_nulls
    return False  # "Poor/Noise", unknown or missing


def make_dispatch(config: Config):
    """Build the main-thread callable that turns an event into an EventTask."""

    def dispatch(app, event, origin, picks):
        event_id = event.publicID()
        if _has_focal_mechanism(event) and not config.seiscomp.reprocess:
            logger.info("Event %s already has a focal mechanism, skipping", event_id)
            return [], None

        arrivals = pick_io.arrival_index(origin)

        # one observation per station; P polarity/onset and the latest
        # accepted RESS S polarization are merged onto the same station
        observations: dict[str, Observation] = {}

        def observation_for(pick) -> Observation | None:
            wfid = pick.waveformID()
            key = f"{wfid.networkCode()}.{wfid.stationCode()}"
            if key in observations:
                return observations[key]
            coords = fm_io.station_coords(
                wfid.networkCode(), wfid.stationCode(), pick.time().value()
            )
            if coords is None:
                logger.warning(
                    "No station coordinates for %s, skipping pick", pick_io.seed_id(pick)
                )
                return None
            latitude, longitude, elevation = coords
            observations[key] = Observation(
                network=wfid.networkCode(),
                station=wfid.stationCode(),
                location=wfid.locationCode(),
                channel=wfid.channelCode(),
                latitude=latitude,
                longitude=longitude,
                elevation=elevation,
            )
            return observations[key]

        n_spol = 0
        for pick in picks:
            is_p = pick_io.is_p_pick(pick, arrivals.get(pick.publicID()))
            polarity = fm_io.polarity_value(pick) if is_p else None
            spol = pick_io.spol_from_comments(pick)

            if polarity is None and spol is None:
                continue

            obs = observation_for(pick)
            if obs is None:
                continue

            if polarity is not None:
                obs.polarity = polarity
                obs.onset = pick_io.onset_code(pick)

            if spol is not None:
                spol_deg, spol_class = spol
                if spol_deg is not None and _accept_spol_class(spol_class, config):
                    obs.spol_deg = spol_deg
                    obs.spol_class = spol_class
                    n_spol += 1

        # keep only stations that contribute something
        obs_list = [
            obs
            for obs in observations.values()
            if obs.polarity is not None or obs.spol_deg is not None
        ]
        n_pol = sum(1 for obs in obs_list if obs.polarity is not None)
        logger.info(
            "Event %s: %d polarities, %d S polarizations from %d picks",
            event_id,
            n_pol,
            n_spol,
            len(picks),
        )
        if n_pol < config.bayfm.min_polarities:
            logger.info(
                "Event %s: %d polarities < min_polarities=%d, skipping",
                event_id,
                n_pol,
                config.bayfm.min_polarities,
            )
            return [], None

        task = EventTask(
            event_id=event_id,
            origin_id=origin.publicID(),
            origin_time=origin.time().value().toString("%Y-%m-%dT%H:%M:%S.%f"),
            latitude=origin.latitude().value(),
            longitude=origin.longitude().value(),
            depth_km=origin.depth().value(),
            magnitude=_preferred_magnitude(event),
            observations=obs_list,
        )
        # only plain identifiers cross to finalize; no SeisComP objects are held
        return [task], (event_id, origin.publicID())

    return dispatch


def make_finalize(config: Config):
    """Build the main-thread callable that publishes the focal mechanism."""

    def finalize(app, context, results):
        event_id, origin_id = context
        result = results.get(event_id)
        if result is None:
            logger.info("No focal mechanism result for event %s", event_id)
            return
        if not result.ok:
            logger.warning("Event %s: focal mechanism failed: %s", event_id, result.error)
            return

        ep = datamodel.EventParameters()
        datamodel.Notifier.Enable()
        try:
            fm = fm_io.build_focal_mechanism(origin_id, result, author=app.name())
            # parent-first notifier order: the FM must be added to a parent
            # before its comments, or their notifiers reference a missing FM
            ep.add(fm)
            fm_io.add_quality_comments(fm, result, software="REBayFM", version=rebayfm_version)
        finally:
            msg = datamodel.Notifier.GetMessage()
            datamodel.Notifier.Disable()

        logger.info(
            "Event %s: strike=%.1f dip=%.1f rake=%.1f rms_kagan=%s -> %s",
            event_id,
            result.strike1,
            result.dip1,
            result.rake1,
            f"{result.rms_kagan_deg:.1f}" if result.rms_kagan_deg is not None else "N/A",
            result.output_dir,
        )
        app.send_message(OUTPUT_GROUP, msg)

    return finalize
