# Compute, Data, and Kaggle Execution Strategy

Last updated: 2026-08-19  
Status: authenticated Kaggle execution active; full metadata scan running

## Operating model

```text
Local development -> Git/source control -> Kaggle data and compute
                  -> lightweight artifacts -> final inference
                  -> validated submission.csv -> explicit approval -> Kaggle
```

The laptop is the control plane. Kaggle is the data and compute plane. Raw
competition data does not move to the laptop; code moves to the mounted data.

## Verified local resources

- RAM reported by Windows: 15.5 GiB
- Fixed volumes: C 233.5 GiB, D 242.5 GiB, 476.1 GiB total
- Free space on 2026-08-13: C 27.5 GiB, D 11.1 GiB, 38.6 GiB total
- User-supplied competition inventory: approximately 569.76 GB and 819,640 files

Even the nominal SSD capacity is below the supplied raw corpus size. Downloading
or caching the full archive locally is prohibited.

## Environment contract

`configs/local.yaml` enforces a sample of 1–20 studies and defaults to ten.
Production modules remain importable and testable with no competition mount.

`configs/kaggle.yaml` requires:

- An actual Kaggle runtime marker
- Internet disabled
- A runtime no longer than the verified nine-hour limit
- A safety reserve of 20 minutes
- Shallow discovery of exactly one competition mount containing the expected
  CSV files and DICOM directories

No competition input path is accepted merely because the config predicts its
name. `scripts/inspect_environment.py` validates and resolves the mount first.

## Current Kaggle connectivity

Kaggle CLI authentication is configured outside the repository with restricted
file permissions. The private source dataset and private kernels are live.
Completed jobs include environment/table audit and both rules-only and semantic
report labeling. The full DICOM metadata job is running. Credentials are never
stored in Git, configuration, notebooks, artifacts, or logs.

## Deliberate Kaggle kernel stages

The starting separation is:

1. Stage 2 metadata: resumable DICOM-header Parquet parts
2. Stage 2 aggregate: ordered series and study inventories
3. Stage 3 semantic: offline multilingual report probabilities
4. Stage 4 features: one frozen DINOv2-small pass over selected 2.5D stacks
5. Stage 5 CV: grouped folds, five attention heads, OOF, calibration
6. Final inference: hidden-test ordering/features, ensemble, validated
   `submission.csv`

Stages may be merged only after profiling shows that doing so improves runtime
or storage without weakening reproducibility. One enormous notebook is not the
default.

## First full-data operation

Read headers before pixels and persist a compact Parquet inventory. Subsequent
experiments consume the inventory instead of rescanning approximately 819,640
files. Pixel decoding begins only after series selection and cache sizing are
supported by the inventory.

Before materializing any cache, call the storage estimator with observed study,
series, slice, resolution, data-type, and compression values. It reports FP32,
FP16, UINT16, and UINT8 footprints. UINT8 remains an experiment requiring image
fidelity and OOF evidence, not an automatic storage choice.

## Artifact policy

- Tier A: folds, OOF, best checkpoints, target schema, configs, registry, final
  ensemble weights. Preserve and version carefully.
- Tier B: DICOM metadata, report labels, selected-series manifests, processed
  representations. Cache when measured storage permits.
- Tier C: temporary tensors, debug images, batches, failed/redundant checkpoints.
  Delete within the Kaggle job lifecycle and never transfer to Git.

## Fold and timeout policy

Each fold must train independently and support fresh start, latest resume, and
specific-checkpoint resume. A fold checkpoint must bind weights and optimizer,
scheduler, scaler, epoch, global step, best score, fold, merged config and hash,
target order, and seed.

The TimeGuard stops at a safe boundary before the nine-hour limit. Model
training is not implemented in Stage 1, but its checkpoint callback contract is
already tested.

## Submission control

No competition submission is automatic. Later stages may generate, validate,
and hash `submission.csv`, inspect the actual notebook slug/version and daily
budget, and recommend `SUBMIT`, `HOLD`, or `REJECT`. The operation stops at:

```text
READY TO SUBMIT
Awaiting approval.
```

Only explicit user approval authorizes spending a submission.
