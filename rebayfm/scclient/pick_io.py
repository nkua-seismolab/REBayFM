"""Read-side helpers for SeisComP Pick objects.

Reads pick onset/polarity codes and the RESS shear-wave splitting comment
carrying the event-corrected S polarization direction.
"""

from __future__ import annotations

import json
import logging

from seiscomp import datamodel

logger = logging.getLogger(__name__)

# comment ids written by RESS look like <pickID>/comment/ress#<compact UTC time>
SWS_COMMENT_MARKER = "/comment/ress#"

ONSET_CODES = {
    datamodel.IMPULSIVE: "I",
    datamodel.EMERGENT: "E",
}


def seed_id(pick) -> str:
    wfid = pick.waveformID()
    return f"{wfid.networkCode()}.{wfid.stationCode()}.{wfid.locationCode()}.{wfid.channelCode()}"


def arrival_index(origin) -> dict:
    """Map pickID -> Arrival for all arrivals of an origin."""
    return {origin.arrival(i).pickID(): origin.arrival(i) for i in range(origin.arrivalCount())}


def _arrival_phase(arrival) -> str:
    try:
        return arrival.phase().code()
    except Exception:
        return ""


def _pick_phase(pick, arrival) -> str:
    phase = _arrival_phase(arrival) if arrival is not None else ""
    if not phase:
        try:
            phase = pick.phaseHint().code()
        except Exception:
            phase = ""
    return phase


def is_p_pick(pick, arrival) -> bool:
    """A pick is treated as P if its arrival phase (or phase hint) starts with P."""
    return _pick_phase(pick, arrival).upper().startswith("P")


def onset_code(pick) -> str | None:
    """ "I" (impulsive) or "E" (emergent) onset of a pick, None if unset."""
    try:
        return ONSET_CODES.get(pick.onset())
    except Exception:
        return None


def spol_from_comments(pick) -> tuple[float | None, str | None] | None:
    """S polarization from the latest RESS splitting comment on a pick.

    Returns (ev_spol_deg, quality_class) from the comment with the most
    recent timestamp in its id, or None if the pick carries no RESS comment.
    """
    latest = None  # (timestamp string, spol_deg, class)
    for i in range(pick.commentCount()):
        comment = pick.comment(i)
        try:
            comment_id = comment.id()
        except Exception:
            continue
        if SWS_COMMENT_MARKER not in comment_id:
            continue
        # compact ISO timestamps sort lexicographically
        stamp = comment_id.split(SWS_COMMENT_MARKER, 1)[1]
        try:
            payload = json.loads(comment.text())
        except (ValueError, TypeError):
            logger.warning("Pick %s: unparsable RESS comment %r", pick.publicID(), comment_id)
            continue
        splitting = payload.get("splitting", {})
        if latest is None or stamp > latest[0]:
            latest = (stamp, splitting.get("ev_spol_deg"), splitting.get("class"))
    if latest is None:
        return None
    return latest[1], latest[2]
