# Data Audit

Status: Stage 2 implementation verified; real-data execution blocked  
Last access check: 2026-08-12

## Actual observed competition data

No competition table, report, DICOM file, or sample submission is present.
Official material now establishes the schemas, hierarchy, targets, and sparse
label design documented in `COMPETITION_AUDIT.md`; actual counts and
distributions remain unavailable.

## Implemented audit pipeline

`scripts/audit_data.py` reads configured train and sample-submission tables and
produces an atomic JSON audit containing:

- Exact rows, columns, dtypes, and submission column order
- Submission-derived identifier and targets, including targets absent from the
  training table
- Candidate patient/study/UID fields with cardinality, missingness, and duplicate
  profiles
- Candidate report-text fields and per-column missingness
- Exact entity counts when identifiers are configured
- Binary target domains, prevalence, and co-occurrence when labels are present

Ambiguous identifiers fail closed and must be configured explicitly.

`scripts/build_metadata.py` recursively considers every file rather than
assuming a filename extension. It reads DICOM headers without pixels, records
each rejected/corrupted file, and atomically emits:

- Slice-level metadata parquet/CSV
- Failure manifest
- Study inventory with series/slice counts and geometry coverage
- Metadata audit with tag missingness and categorical distributions
- Series slice-count distributions
- Geometry-derived ordering reliability
- Relative path depth distribution

Spatial slice coordinates are calculated by projecting
`ImagePositionPatient` onto the normal formed from
`ImageOrientationPatient`; filenames are not used for ordering.

`scripts/build_folds.py` creates deterministic multilabel group folds through
multiple seeded greedy restarts. Patient grouping takes precedence when a
verified patient identifier is configured; otherwise the verified study key is
used. It emits row assignments and a fold audit with study/patient counts,
target counts and prevalence, and configured subgroup distributions. Every fold
is checked for patient and study overlap.

Sparse labels are handled with two separate balancing signals per target:
observed positives and total observed labels. Missing labels remain unknown and
do not enter positive or negative counts. Fold prevalence is computed only
among known labels.

## Local storage decision

Drive D had 11.1 GiB free on 2026-08-12. The authenticated Kaggle file listing
needed to obtain the exact archive size was unavailable. A blind full download
is therefore rejected. Stage 2 extraction should run against Kaggle-mounted
competition data, after which only compact audit Parquet/JSON artifacts should
be copied locally.

## Verification fixtures

Synthetic verification covers numeric and UID identifiers, submission target
order, targets absent from training tables, prevalence/co-occurrence, valid
extensionless MRI DICOM files, corrupted-file manifests, geometry-derived slice
coordinates, study inventories, deterministic group assignments, multilabel
balance, subgroup audits, and explicit leakage assertions.

Synthetic counts and labels are test fixtures only and are not competition
findings.

## Execution gate

Populate the `paths` and `schema` sections of `configs/data.yaml` after adding
the official files. Run table and DICOM audits first, inspect identifier
relationships, select the strongest grouping key, then run fold generation. A
fold artifact produced before that inspection is invalid.
