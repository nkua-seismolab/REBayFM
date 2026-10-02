"""Plain data containers passed between the SeisComP process and workers.

Everything here must be picklable (crosses the ProcessPoolExecutor boundary)
and must not import any SeisComP module.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Observation:
    """Per-station inputs to the inversion.

    ``polarity`` is +1 (compression), -1 (dilatation) or None (station
    contributes only an S-wave polarization). ``onset`` is "I" (impulsive),
    "E" (emergent) or None (treated as emergent). ``spol_deg`` is the
    event-corrected S-wave polarization azimuth from RESS, if accepted.
    """

    network: str
    station: str
    location: str
    channel: str
    latitude: float
    longitude: float
    elevation: float  # meters
    polarity: float | None = None
    onset: str | None = None
    spol_deg: float | None = None
    spol_class: str | None = None


@dataclass
class EventTask:
    """One focal-mechanism inversion job for a single event/origin."""

    event_id: str
    origin_id: str
    origin_time: str  # ISO 8601
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float | None = None
    observations: list[Observation] = field(default_factory=list)


@dataclass
class FMResult:
    """Outcome of one focal-mechanism inversion (per event)."""

    event_id: str
    origin_id: str
    # UTC timestamp of processing, ISO 8601; also names the output subdirectory
    # and is reused verbatim as provenance "stored_at" in the SeisComP comment.
    processed_at: str = ""
    # MAP solution, fault plane 1 and its conjugate (degrees)
    strike1: float | None = None
    dip1: float | None = None
    rake1: float | None = None
    strike2: float | None = None
    dip2: float | None = None
    rake2: float | None = None
    # Gaussian-fit mean solution per plane (degrees)
    strike1_gmean: float | None = None
    dip1_gmean: float | None = None
    rake1_gmean: float | None = None
    strike2_gmean: float | None = None
    dip2_gmean: float | None = None
    rake2_gmean: float | None = None
    # Gaussian-fit standard deviations per plane (degrees)
    strike1_sigma: float | None = None
    dip1_sigma: float | None = None
    rake1_sigma: float | None = None
    strike2_sigma: float | None = None
    dip2_sigma: float | None = None
    rake2_sigma: float | None = None
    # quality metrics
    n_polarities: int | None = None
    fmp_matches: int | None = None
    fmp_misfits: int | None = None
    fmp_match_pct: float | None = None
    fmp_matches_weighted: float | None = None
    fmp_misfits_weighted: float | None = None
    fmp_match_weighted_pct: float | None = None
    n_spol: int | None = None
    avg_spol_misfit_deg: float | None = None
    rms_kagan_deg: float | None = None
    kagan_g1_g2_deg: float | None = None
    # where the artifacts (figures, CSVs, log) were written
    output_dir: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
