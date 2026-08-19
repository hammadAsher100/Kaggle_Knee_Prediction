# Stage 2 Kaggle preprocessing

Use Kaggle-mounted competition inputs instead of downloading the DICOM archive
to the local D drive. The tracked `run.py` launcher extracts the private
`hammadasher/rsna-knee-source` dataset, keeps internet disabled, validates the
competition mount and dependencies, and runs the table audit.

Publish the launcher with:

```bash
python -m kaggle kernels push -p kaggle/preprocessing
```

The kernel metadata attaches both the private source snapshot and the
competition input. The setup kernel deliberately stops before the full DICOM
scan so the integration can be verified quickly and cheaply.

From the repository root in the notebook, run:

```bash
python scripts/inspect_environment.py --config configs/kaggle.yaml
python scripts/audit_data.py --config configs/kaggle.yaml --config configs/kaggle_stage2.yaml
python scripts/build_metadata.py --config configs/kaggle.yaml --config configs/kaggle_stage2.yaml --split train
```

The first command discovers the attached competition mount by shallow inspection
and fails unless the expected CSV files and series directories exist. Production
commands reuse the resolved mount; they do not assume a locally downloaded copy.

Outputs are written beneath `/kaggle/working/stage2_artifacts`. Download those
compact JSON/Parquet files and place them under the local `artifacts/metadata/`
directory.

Do not run `build_folds.py` until the metadata audit has established whether a
stable patient identifier survives in the 86-tag allowlist. If one exists,
configure it as `schema.patient_identifier`; otherwise retain
`StudyInstanceUID` as the highest verified grouping key.

The public test files are examples that are replaced during scoring. Training
audit work should not infer the private test distribution from those examples.
