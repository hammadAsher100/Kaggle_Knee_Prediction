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

## Full-data execution addendum — 2026-08-19

Authenticated inspection established 4,407 unique training studies and 58
observed gold labels per target; reports are complete. The five competition
CSV files were downloaded locally as lightweight, Git-ignored audit inputs.
The approximately 569.76 GB DICOM corpus remains on Kaggle.

The full training-header scan is running in the private
`rsna-knee-stage-2-metadata` kernel. It emits atomic Parquet parts every 25
studies plus a completion manifest and failure ledger. The dependent aggregate
kernel will not start unless that manifest reports `complete: true`. PatientID,
AccessionNumber, vendor/model, field strength, geometry reliability, series
sizes, and descriptor missingness remain pending until aggregation completes.

The current report-language audit contains no raw report text. Its heuristic
counts are: English 1,737; Spanish 681; Turkish 544; Greek 321; Croatian 285;
German 256; Bulgarian 220; Dutch 151; French 81; unknown 131. These language
counts are routing diagnostics, not clinical labels.

After Unicode/whitespace normalization and case folding, 52 exact duplicate
report-text groups contain 198 studies; the largest group contains 37. These
may include generic reporting templates, so report equality is audited but is
not assumed to be a patient identifier. DICOM PatientID remains the preferred
cross-study grouping evidence.

The actual `train.csv` columns do not include `PatientSex`, despite its presence
in the supplied dataset description. The training and subgroup-audit code treats
that field as optional and records it as unavailable; it does not impute sex.
