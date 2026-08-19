# Strategy

Last updated: 2026-08-12  
Current reviewed boundary: Stage 1 complete; Stage 2 tooling implemented but
competition-data execution blocked; no neural-network training started

## Objective

Maximize private-leaderboard generalization and efficiency through a disciplined
information pipeline: trustworthy labels, leakage-safe CV, correct MRI series
and geometry handling, then efficient representations and evidence-based
ensembling.

## Stage 1 decisions

### Repository root

The current Git root is the logical `rsna_knee_abnormality/` root shown in the
mission. A redundant nested wrapper directory was not added. Runtime code lives
under `src/`; notebooks contain no production logic.

### Evidence before implementation

The dataset is absent, so target names, target count, IDs, grouping keys,
submission schema, rule constraints, and DICOM distributions remain unset.
Empty target lists and `null` schema values are intentional fail-closed
configuration, not incomplete guesses.

### Configuration

YAML files own paths and major hyperparameters. Later files deterministically
deep-merge over earlier files; explicit overrides merge last. `${VARIABLE}`
references are expanded strictly and fail if undefined. No code contains a
competition data path.

### Provenance

Configuration hashes use canonical JSON and SHA-256, independent of mapping
order. Runtime metadata can record Python, platform, installed package versions,
Git commit, and optional PyTorch/CUDA/GPU details without forcing a heavyweight
torch import during lightweight jobs.

### Logging

Operational logs are JSON Lines with UTC timestamps and structured context.
Future experiments will log the experiment ID, config hash, seed, fold, target
schema version, code commit, hardware, runtime, and metrics. `print` is not an
operational logging mechanism.

### Determinism

The seed utility configures Python, NumPy, and, when requested, PyTorch/CUDA.
Deterministic PyTorch algorithms are requested with warnings for unsupported
operations. Determinism limitations must be recorded per experiment.

### Time safety

The wall-clock guard operates at explicit safe-operation boundaries. It refuses
work that cannot fit in the configured reserve or lets the current atomic
operation finish, invokes a checkpoint callback with the latest training state,
flushes logging, and raises a dedicated clean-stop signal. The future training
checkpoint callback owns model, optimizer, scheduler, scaler, epoch, step, best
score, and RNG serialization.

### Dependency discipline

Stage 1 core depends only on PyYAML; NumPy is used when available for seeding.
Heavy data and ML dependencies are separated in optional groups. Exact packages
and offline model artifacts must be reconciled with the actual Kaggle image and
rules before a submission pipeline is frozen.

## Stage 2 implementation decisions

### Schema discovery fails closed

The sample submission defines prediction column order. Identifier-like columns
are excluded before numeric target inference, including numeric study/patient
IDs. Target names may be discovered even when training targets are absent, such
as when labels must be derived from reports. Ambiguity requires an explicit YAML
decision.

### DICOM inventory is extension agnostic

All regular files beneath a configured root are attempted as DICOM, using
pixel-free reads. Corruption is recorded rather than silently dropped. Metadata
and failure outputs are atomic. Geometry ordering uses the orientation-vector
cross product and position projection, never filenames.

### Group-safe multilabel folding

Leakage groups are indivisible. The initial method uses deterministic multi-start
greedy assignment to balance row volume and each target's positive contribution.
It is transparent, dependency-light, and testable. It is not automatically the
final CV method: real fold prevalence and subgroup distributions determine
whether it is retained or revised.

Patient grouping is preferred only after patient identity is verified. Every
candidate split asserts disjoint study IDs and, when available, disjoint patient
IDs. Site, language, vendor, and field strength enter the audit, not the model,
until distribution evidence is available.

Sparse official labels are balanced using both positive counts and label
coverage per target. Unknown labels are never interpreted as negatives. If no
cross-study patient identifier is recoverable from the allowlisted DICOM tags,
`StudyInstanceUID` is the highest verified grouping level.

### Data locality

The full archive will not be downloaded blindly to a volume with only 11.1 GiB
free. Metadata extraction will run inside Kaggle against mounted inputs. Compact
header, study-inventory, table-audit, and fold artifacts will then be transferred
locally for analysis and reproducible configuration updates.

Local execution now fails unless `data.use_sample` is true and `max_studies` is
between 1 and 20. Kaggle execution fails unless a real runtime marker, offline
mode, working directory, and complete competition mount are detected. This
prevents an accidental full-data path from being exercised on the laptop.

Every proposed image cache must be estimated before materialization. The
estimator compares FP32, FP16, UINT16, and UINT8 using observed dimensions and
an explicit compression ratio. Storage reduction never substitutes for OOF
validation of medical signal fidelity.

Kaggle work will use separate audit, label, preprocessing, fold, ensemble, and
final inference kernels initially. Each fold is an independent resumable unit.
Kernel boundaries may change only after runtime/storage profiling.

Competition submission remains a separate approval boundary. Generating and
validating a candidate does not authorize uploading it.

## Experiment decision system

Every ML experiment must have one primary change and record:

- Hypothesis and expected effect
- Fully merged configuration and hash
- Git commit and seed
- OOF macro and per-target AUC
- Fold variance and subgroup results where support is adequate
- Runtime, throughput, parameter count, model size, and peak VRAM/CPU memory
- Comparison against the current validated baseline
- Decision: `KEEP`, `REJECT`, or `INVESTIGATE`

No result is promoted based on one fold or public-leaderboard movement alone.
OOF predictions are immutable artifacts and ensemble weights are fitted only on
OOF evidence.

## Planned sequence after review

1. Stage 2: ingest real files, run complete table/DICOM audit, select the safest
   grouping key, and generate audited folds.
2. Stage 3: build versioned local-only multilingual report labels with evidence,
   negation, uncertainty, and confidence.
3. Stage 4: validate MRI decoding, geometric ordering, normalization,
   orientation, laterality, series categorization, and slice sampling.
4. Stage 5: establish the conservative end-to-end 2.5D OOF baseline.
5. Later stages proceed only through isolated, measured changes described in the
   mission.

## Decisions explicitly deferred

- Exact target schema and loss output width
- Patient/study grouping key and fold algorithm details
- Report terminology and language-specific rules
- Series priorities, slice count, resolution, and normalization
- Official metric behavior and submission validator
- Backbone/pretrained weights and their competition/license eligibility
- External data, TTA, pooling, fusion, ensembling, and efficiency tradeoffs

These decisions require observed data, official rules, or OOF evidence.
