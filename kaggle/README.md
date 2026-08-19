# Kaggle kernel architecture

No Kaggle notebook is linked to this repository yet. Slugs and versions must be
discovered after authentication; they are never invented in configuration.

Initial execution stages:

| Stage | Responsibility | Durable outputs |
|---|---|---|
| 01 Audit | CSV schemas, DICOM headers, quality and storage inventory | Metadata and audit Parquet/JSON |
| 02 Labels | Offline multilingual report processing | Versioned weak labels |
| 03 Preprocess | Selected series/slices and measured image representation | Selection manifests and optional cache |
| 04+ Folds | One independently resumable training job per fold | Best checkpoint, fold OOF, metrics |
| OOF/Ensemble | Evaluation, correlation, ensemble selection | OOF report and ensemble config |
| Final inference | Test preprocessing, prediction, strict validation | `submission.csv` and validation report |

The split is provisional. Merge stages only when profiling demonstrates a clear
runtime or storage benefit. Never combine folds in a way that makes one
nine-hour Kaggle session responsible for the complete CV experiment.

Notebook submission is a later approval-gated operation. Before preparing it,
discover the actual owner/notebook slug and committed version from Kaggle.

