# RSNA Knee Abnormality Detection

Competition-grade, configuration-driven infrastructure for the 2026 RSNA Knee
Abnormality Detection challenge.

## Current status

Stage 1 is complete and now enforces a laptop-control-plane/Kaggle-data-plane
architecture. Local mode is limited to at most 20 representative studies;
Kaggle mode discovers and validates the mounted competition input before use.
No full dataset was downloaded, no model was trained, and no submission was
made.

The competition dataset was not present in the workspace during the Stage 1
audit. Exact targets, identifiers, schemas, and rules therefore remain
deliberately unset. See `docs/COMPETITION_AUDIT.md` for the evidence boundary.

## Layout

The repository root is the logical `rsna_knee_abnormality/` directory from the
competition mission. Runtime logic lives in `src/`; notebooks are exploratory
only.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Copy `.env.example` to `.env` only for local convenience. Paths used by code
must still enter through YAML configuration or explicit command-line values.

## Reproducibility contract

Each future experiment must record its merged configuration, configuration
hash, seed, package and hardware details, Git commit (when available), fold,
target order, runtime, and checkpoint metadata. OOF and leaderboard results
must never be invented or inferred.

See [docs/COMPUTE_STRATEGY.md](docs/COMPUTE_STRATEGY.md) for environment,
kernel, artifact, storage, and submission-approval rules.
