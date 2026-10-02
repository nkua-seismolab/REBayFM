# REBayFM

[![lint](https://github.com/nkua-seismolab/REBayFM/actions/workflows/lint.yml/badge.svg)](https://github.com/nkua-seismolab/REBayFM/actions/workflows/lint.yml)
[![tests](https://github.com/nkua-seismolab/REBayFM/actions/workflows/tests.yml/badge.svg)](https://github.com/nkua-seismolab/REBayFM/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Near real-time Bayesian focal mechanisms from first-motion polarities and
S-wave polarizations for SeisComP.

REBayFM is part of a four-application workflow for automatic weak event source characterization:

[REPOL](https://github.com/nkua-seismolab/REPOL) / [RESS](https://github.com/nkua-seismolab/RESS) -> [REHASH](https://github.com/nkua-seismolab/REHASH) / **[REBayFM](https://github.com/nkua-seismolab/REBayFM)**

## How it works

REBayFM connects to the SeisComP messaging bus and listens for new events.
After the configured wait time (allowing upstream polarity and splitting
measurements to finish) it:

1. Collects P first-motion polarities and onsets from the picks
   (e.g. produced by [REPOL](https://github.com/nkua-seismolab/REPOL)) and
   S-wave polarization directions from the
   [RESS](https://github.com/nkua-seismolab/RESS) JSON comments.
2. Traces rays through a 1-D velocity model for takeoff angles, then runs a
   Bayesian grid search over strike/dip/rake. The posterior combines a first-motion mismatch term -
   onset-dependent mis-pick probabilities, optionally weighted by the P
   radiation amplitude - with an S-polarization misfit term.
3. Reports the MAP solution with uncertainties from the
   highest-posterior-density region, and sends it to SeisComP as a
   `FocalMechanism` (method `REBayFM`) with quality metrics attached as a
   JSON comment. Figures and CSVs of each run are kept under `output/`.

No waveform access is needed - REBayFM works entirely from the SeisComP
database.

## Requirements

- A running SeisComP system.
- Docker Engine with the Compose plugin.
- Resources: ~2 GB disk for the image; roughly 2 GB RAM during
  processing (full strike/dip/rake posterior grid).

Tested with SeisComP 7.3.0 on Ubuntu 24.04.1.

## Installation

```bash
git clone https://github.com/nkua-seismolab/REBayFM.git
cd REBayFM
cp config.example.yaml config.yaml
```

Then:

1. Edit `config.yaml`: set `seiscomp.host` and `seiscomp.database` to your
   SeisComP messaging host and database URL, and point `velocity_model.path`
   at your 1-D model (an example model ships in `velocity_models/`).
2. Start:

   ```bash
   docker compose up -d --build
   docker compose logs -f
   ```

## Configuration

All options live in `config.yaml` and are documented inline in
[config.example.yaml](config.example.yaml). The main sections:

| Section | Purpose |
| ------- | ------- |
| `seiscomp` | Messaging host, database URL, wait time, reprocessing policy |
| `velocity_model` | 1-D velocity model used for ray tracing |
| `bayfm` | Grid resolution, mis-pick probabilities, S-polarization usage, posterior weights, HPD level |
| `output` | Artifact directory and figure generation |
| `logging` | Level and optional log file |

## Output

- A `FocalMechanism` (method `REBayFM`) with nodal planes and uncertainties,
  station polarity count, and misfit, sent to the `FOCMECH` messaging group.
- A JSON comment (id `<fmID>/comment/rebayfm`) with the quality metrics,
  e.g.:

  ```json
  {"provenance": {"software": "REBayFM", "version": "1.0.0", "stored_at": "..."},
   "rms_kagan_deg": 21.4, "fmp_match_pct": 92.3, "n_spol": 6,
   "avg_spol_misfit_deg": 11.8, "output_dir": "output/nkua2020abcd/..."}
  ```

- Figures (beachball, posterior marginals), CSVs, and logs of each run under
  `output/<event id>/<processed at>/`.

## Getting started

See [GETTING_STARTED.md](GETTING_STARTED.md) for a step-by-step demo run with
the example event `nkua2020abcd` from the demo dataset
([doi:10.5281/zenodo.23105466](https://doi.org/10.5281/zenodo.23105466)).

## License

[GPL-3.0](LICENSE)

## Funding

This work is part of the [TRANSFORM²](https://www.transform2-project.eu/) project which aims to improve physical and digital infrastructure across Near-Fault Observatories (NFOs) in Europe.

TRANSFORM² is funded by the European Union under project number 101188365 within the HORIZON-INFRA-2024-DEV-01-01 call.

<div align="center">
  <img src="https://www.transform2-project.eu/wp-content/uploads/2022/08/Logo_TRANSFORM2-round-logo-100x100-1.png" alt="TRANSFORM² logo">
</div>
