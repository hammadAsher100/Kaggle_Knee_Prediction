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

### 2026-08-19 — STAGE3-RULES-V1

- Type: report weak-label development
- Data: 4,407 reports; 58 gold studies per target
- Model: multilingual terminology, negation, and uncertainty rules
- Gold macro ROC AUC: 0.6692662575
- Public leaderboard: N/A — no submission made
- Decision: INVESTIGATE — useful for several structural findings, weak for OA
  and effusion

### 2026-08-19 — STAGE3-SEMANTIC-BLEND-V1

- Type: offline report weak-label development
- Primary change: multilingual MiniLM sentence similarities blended with rules
- Candidate rule weights: 0, 0.25, 0.5, 0.75, 1
- Selected development weight: 0.5
- Gold development macro ROC AUC: 0.6881405384
- Per-target AUC: ACL 0.8487; MCL 0.8481; medial meniscus 0.7200; lateral
  meniscus 0.6783; medial OA 0.4969; lateral OA 0.6557; PF OA 0.5714;
  effusion 0.5801; synovitis 0.7276; Baker's 0.7210; contusion 0.6579;
  fracture 0.7521
- Runtime: about 134 seconds in the Kaggle T4 job including model load and
  notebook finalization
- Public leaderboard: N/A — no submission made
- Decision: KEEP as the current weak-label source
- Leakage control: image CV reselects blend weight within each development
  partition; the global 0.6881 value is not represented as unbiased image OOF
