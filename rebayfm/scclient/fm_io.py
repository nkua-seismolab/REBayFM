"""Conversion between core focal mechanism results and SeisComP objects.

Free of package-internal imports; result objects are duck-typed
(FMResult, see rebayfm.core.models).
"""

from __future__ import annotations

import json
import logging

from seiscomp import client, core, datamodel

logger = logging.getLogger(__name__)

# pick polarity enum -> numeric first motion
POLARITY_VALUES = {
    datamodel.POSITIVE: 1.0,
    datamodel.NEGATIVE: -1.0,
}


def polarity_value(pick) -> float | None:
    """Numeric first motion of a pick (+1 up, -1 down), None if unset/undecidable."""
    try:
        return POLARITY_VALUES.get(pick.polarity())
    except Exception:
        return None


def station_coords(network: str, station: str, time) -> tuple[float, float, float] | None:
    """(latitude, longitude, elevation_m) from the loaded inventory, or None."""
    try:
        sta = client.Inventory.Instance().getStation(network, station, time)
        return (sta.latitude(), sta.longitude(), sta.elevation())
    except Exception:
        return None


def _real_quantity(value: float, uncertainty: float | None = None) -> datamodel.RealQuantity:
    quantity = datamodel.RealQuantity(float(value))
    if uncertainty is not None:
        quantity.setUncertainty(float(uncertainty))
    return quantity


def build_focal_mechanism(origin_id: str, result, author: str) -> datamodel.FocalMechanism:
    """Build a FocalMechanism from an FMResult-like object (see core models).

    Both nodal planes carry the MAP solution; the per-plane Gaussian-fit
    sigmas become the uncertainties of the plane parameters.
    """
    fm = datamodel.FocalMechanism.Create()
    fm.setTriggeringOriginID(origin_id)

    np1 = datamodel.NodalPlane()
    np1.setStrike(_real_quantity(result.strike1, result.strike1_sigma))
    np1.setDip(_real_quantity(result.dip1, result.dip1_sigma))
    np1.setRake(_real_quantity(result.rake1, result.rake1_sigma))
    np2 = datamodel.NodalPlane()
    np2.setStrike(_real_quantity(result.strike2, result.strike2_sigma))
    np2.setDip(_real_quantity(result.dip2, result.dip2_sigma))
    np2.setRake(_real_quantity(result.rake2, result.rake2_sigma))
    planes = datamodel.NodalPlanes()
    planes.setNodalPlane1(np1)
    planes.setNodalPlane2(np2)
    fm.setNodalPlanes(planes)

    if result.fmp_matches is not None and result.fmp_misfits is not None:
        n_fmp = result.fmp_matches + result.fmp_misfits
        fm.setStationPolarityCount(int(n_fmp))
        if n_fmp:
            # fraction of mismatching first-motion polarities
            fm.setMisfit(round(result.fmp_misfits / n_fmp, 4))

    fm.setMethodID("REBayFM")
    fm.setEvaluationMode(datamodel.AUTOMATIC)
    ci = datamodel.CreationInfo()
    ci.setAuthor(author)
    ci.setCreationTime(core.Time.GMT())
    fm.setCreationInfo(ci)
    return fm


def add_quality_comments(fm, result, software: str, version: str) -> None:
    """Attach quality metrics and provenance as a single JSON comment.

    Call only after fm was added to EventParameters: Notifier serialization
    ignores children (Archive::IGNORE_CHILDS), so the comment travels as its
    own notifier and its parent must already exist when it is applied.

    ``stored_at`` reuses the processing timestamp of the result, which also
    names the artifact output subdirectory (provenance linking).
    """
    # flat metrics, units folded into key names; gmean planes are [strike, dip, rake]
    metrics = {
        "rms_kagan_deg": result.rms_kagan_deg,
        "kagan_g1_g2_deg": result.kagan_g1_g2_deg,
        "fmp_match_pct": result.fmp_match_pct,
        "fmp_match_weighted_pct": result.fmp_match_weighted_pct,
        "n_spol": result.n_spol,
        "avg_spol_misfit_deg": result.avg_spol_misfit_deg,
        "gmean_plane1_deg": [result.strike1_gmean, result.dip1_gmean, result.rake1_gmean],
        "gmean_plane2_deg": [result.strike2_gmean, result.dip2_gmean, result.rake2_gmean],
        "output_dir": result.output_dir,
    }
    payload = {
        "provenance": {
            "software": software,
            "version": version,
            "stored_at": result.processed_at,
        },
        **{key: value for key, value in metrics.items() if value is not None},
    }
    comment = datamodel.Comment()
    comment.setId(f"{fm.publicID()}/comment/rebayfm")
    comment.setText(json.dumps(payload, ensure_ascii=True))
    fm.add(comment)
