"""Configuration loading and validation for REBayFM (config.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import isfinite
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin, get_type_hints

import yaml


class ConfigError(Exception):
    """Raised when config.yaml is missing, malformed or fails validation."""


@dataclass
class SeisCompConfig:
    """Options consumed by the scclient module."""

    host: str | None = None
    database: str | None = None
    wait_time: int = 300
    reprocess: bool = False


@dataclass
class VelocityModelConfig:
    """TauP-compatible layered model (.nd); compiled to .npz on first use.

    The same file provides the Vp/depth layer array used for the S-wave
    polarization rotation correction.
    """

    path: str = "velocity_models/model_rigo1996_half.nd"


@dataclass
class BayFMConfig:
    """Bayesian inversion options."""

    # grid search steps (degrees)
    strike_step: float = 10.0
    dip_step: float = 5.0
    rake_step: float = 10.0
    # search strikes in [0, 360) instead of [0, 180]
    use_strike360: bool = True
    # minimum number of P first-motion polarities; events below are skipped
    min_polarities: int = 8
    # per-pick mis-pick probabilities by onset (missing onset counts as emergent)
    mis_pick_impulsive: float = 0.10
    mis_pick_emergent: float = 0.30
    # use S-wave polarization measurements (RESS pick comments) if available
    use_spol: bool = True
    # also accept measurements from picks classified "Good Null" by RESS
    use_nulls: bool = False
    # fixed 1-sigma angular uncertainty of accepted S-polarization measurements
    spol_sigma_deg: float = 15.0
    # S-polarization likelihood: 1 Gaussian, 2 Laplace, 3 Cauchy
    f_spol: int = 3
    # posterior weights of the S-polarization term and the FMP mismatch penalty
    w_spol: float = 0.2
    w_fmp: float = 2.0
    # weight FMP measurements by P radiation amplitude (downweight near nodal planes)
    fmp_amp: bool = True
    # flat penalty for more than k_allowed FMP mismatches (null disables it)
    k_allowed: int | None = None
    lambda_penalty: float = 5.0
    # use per-axis marginal MAP estimates instead of the joint MAP
    select_marginal_map: bool = False
    # marginalize the posterior over a range of trial depths
    use_depth_range: bool = False
    depth_range_km: float = 1.0
    depth_step_km: float = 0.2
    # cumulative posterior mass of the highest-posterior-density region
    hpd_level: float = 0.68


@dataclass
class OutputConfig:
    """Per-event artifacts (figures, CSVs, processing log) on a mounted volume."""

    directory: str = "output"
    save_figures: bool = True


@dataclass
class LoggingConfig:
    level: str = "DEBUG"
    file: str | None = None


@dataclass
class Config:
    seiscomp: SeisCompConfig = field(default_factory=SeisCompConfig)
    velocity_model: VelocityModelConfig = field(default_factory=VelocityModelConfig)
    bayfm: BayFMConfig = field(default_factory=BayFMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _matches_type(value, annotation) -> bool:
    if get_origin(annotation) is UnionType:
        return any(_matches_type(value, option) for option in get_args(annotation))
    if get_origin(annotation) is list:
        return isinstance(value, list) and all(
            _matches_type(item, get_args(annotation)[0]) for item in value
        )
    if annotation is float:
        return type(value) in (int, float) and isfinite(value)
    return type(value) is annotation


def _build_section(cls, name: str, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown keys in '{name}' section: {sorted(unknown)}")
    hints = get_type_hints(cls)
    for key, value in data.items():
        if not _matches_type(value, hints[key]):
            raise ConfigError(f"{name}.{key} has an invalid type or non-finite value")
    return cls(**data)


def _validate(config: Config) -> None:
    if config.logging.level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ConfigError("logging.level must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
    if config.seiscomp.wait_time < 0:
        raise ConfigError("seiscomp.wait_time must be >= 0")
    if not config.velocity_model.path:
        raise ConfigError("velocity_model.path must point to a .nd velocity model file")
    bayfm = config.bayfm
    if bayfm.min_polarities < 1:
        raise ConfigError("bayfm.min_polarities must be >= 1")
    for key in ("strike_step", "dip_step", "rake_step"):
        if getattr(bayfm, key) <= 0:
            raise ConfigError(f"bayfm.{key} must be > 0")
    if bayfm.spol_sigma_deg <= 0:
        raise ConfigError("bayfm.spol_sigma_deg must be > 0")
    if bayfm.f_spol not in (1, 2, 3):
        raise ConfigError("bayfm.f_spol must be 1 (Gaussian), 2 (Laplace) or 3 (Cauchy)")
    if not 0 < bayfm.hpd_level < 1:
        raise ConfigError("bayfm.hpd_level must be between 0 and 1")
    if bayfm.use_depth_range and (bayfm.depth_range_km <= 0 or bayfm.depth_step_km <= 0):
        raise ConfigError("bayfm.depth_range_km and bayfm.depth_step_km must be > 0")
    if not (0 <= bayfm.mis_pick_impulsive <= 1 and 0 <= bayfm.mis_pick_emergent <= 1):
        raise ConfigError("bayfm mis-pick probabilities must be within [0, 1]")
    if not config.output.directory:
        raise ConfigError("output.directory must be a non-empty path")


def load_config(path: str | Path) -> Config:
    """Load and validate a config.yaml file. Missing sections/keys use defaults."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path) as fid:
            raw = yaml.safe_load(fid)
    except yaml.YAMLError:
        raise ConfigError(f"Invalid YAML in {path}") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")

    sections = {f.name: f.default_factory for f in fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ConfigError(f"Unknown top-level sections: {sorted(unknown)}")

    kwargs = {}
    for f in fields(Config):
        data = raw.get(f.name, {})
        if not isinstance(data, dict):
            raise ConfigError(f"Section '{f.name}' must be a mapping")
        section_cls = f.default_factory
        kwargs[f.name] = _build_section(section_cls, f.name, data)

    config = Config(**kwargs)
    _validate(config)
    return config
