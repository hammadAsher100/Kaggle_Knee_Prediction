# Experiment Log

This file is append-only except for clearly marked factual corrections. ML
metrics are recorded only after actual execution.

## Entries

### 2026-08-12 — STAGE1-ENGINEERING-AUDIT

- Type: engineering foundation; not an ML experiment
- Hypothesis: strict configuration, provenance, structured logging,
  deterministic seeding, and checkpoint-first time control reduce invalid and
  irreproducible future experiments
- Primary change: initial repository scaffold and Stage 1 audit
- Config: `configs/base.yaml` plus component skeletons
- Config hash: recorded by tests/commands when a merged experiment is launched;
  no experiment run was launched in Stage 1
- OOF macro ROC AUC: N/A — no training performed
- Per-target AUC: N/A — targets unavailable and no training performed
- Runtime/VRAM: N/A — no model profiling performed
- Public leaderboard: N/A — no submission made
- Decision: KEEP
- Notes: competition data and official Kaggle rule/evaluation exports were not
  present; exact schemas remain deliberately unset

### 2026-08-12 — STAGE2-PIPELINE-IMPLEMENTATION

- Type: data engineering verification; not an ML experiment
- Hypothesis: exact sample-driven schema discovery, geometry-based DICOM
  inventory, and deterministic indivisible-group fold assignment will expose
  invalid assumptions before GPU experiments
- Primary change: Stage 2 table audit, DICOM metadata audit, study inventory,
  and multilabel grouped-fold implementation
- Data: synthetic fixtures only; competition data remained unavailable
- OOF macro ROC AUC: N/A — no training performed
- Per-target AUC: N/A — actual targets unavailable
- Runtime/VRAM: N/A — no model profiled
- Public leaderboard: N/A — no submission made
- Decision: INVESTIGATE — implementation passes synthetic verification but
  cannot be accepted for competition use until run and reviewed on real data
- Notes: Kaggle CLI authentication and official competition files are still
  absent; no synthetic value is treated as a competition finding

### 2026-08-13 — STAGE1-COMPUTE-ARCHITECTURE

- Type: engineering and execution-boundary verification; not an ML experiment
- Hypothesis: explicit local/Kaggle modes and pre-cache sizing prevent accidental
  local corpus downloads, storage explosions, and invalid environment runs
- Primary change: runtime mount discovery, local sample cap, cache estimator,
  credential/binary Git protection, and Kaggle kernel/artifact policy
- Kaggle MCP: not exposed
- Kaggle CLI: installed, unauthenticated, competition listing unavailable
- Local resources: 15.5 GiB RAM; 476.1 GiB fixed-volume capacity; 38.6 GiB free
- Dataset: user-supplied estimate 569.76 GB and 819,640 files; not downloaded
- OOF macro ROC AUC: N/A — no training performed
- Runtime/VRAM: N/A — no model profiled
- Public leaderboard: N/A — no submission made
- Decision: KEEP
- Notes: local and Kaggle execution boundaries are synthetic-testable; actual
  Kaggle mount discovery awaits an authenticated notebook runtime
