# RSNA Knee Abnormality Detection

Competition-grade, configuration-driven infrastructure for the 2026 RSNA Knee
Abnormality Detection challenge.

## Current status

The project now has an executable offline pipeline from raw competition DICOMs
to a validated `submission.csv`:

1. resumable pixel-free DICOM metadata extraction and aggregation;
2. private multilingual report labeling (rules plus MiniLM embeddings);
3. patient-aware grouped folds with nested weak-label blending;
4. geometry-ordered 2.5D stacks and frozen DINOv2-small feature extraction;
5. five-fold plane-aware attention heads, gold-only OOF metrics, and
   cross-fitted monotonic calibration;
6. hidden-test metadata extraction, five-model inference, and strict submission
   validation.

Full-data work runs on Kaggle; only compact audits and model artifacts are
downloaded locally. The complete Stage 2 scan covers 819,078 DICOM instances,
24,371 series, and all 4,407 training studies. Corrected DINOv2 features cover
99.996% of planned stacks. The best deployable leakage-safe local result is an
equal-probability ensemble of two strict image models with gold-only macro ROC
AUC 0.7466; this is not a Kaggle leaderboard score. No competition submission
has been made, and submission remains an explicit user approval boundary.

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

Each experiment records its merged configuration, configuration
hash, seed, package and hardware details, Git commit (when available), fold,
target order, runtime, and checkpoint metadata. OOF and leaderboard results
must never be invented or inferred.

See [docs/COMPUTE_STRATEGY.md](docs/COMPUTE_STRATEGY.md) for environment,
kernel, artifact, storage, and submission-approval rules.
