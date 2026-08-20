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

### 2026-08-19 — STAGE4-DINOV2-SMOKE

- Type: real-DICOM pipeline validation; not a predictive experiment
- Data: three example test studies, 15 series, 557 slices
- Geometry: 15/15 series ordered from orientation/position metadata
- Header failures and empty series: 0
- Sampled representation: 12 three-slice stacks per study, DINOv2-small,
  384-dimensional FP16 embeddings
- Pixel-decode valid-stack fraction: 1.0
- Feature shape: `(3, 12, 384)`; all values finite
- Runtime: 2.34 seconds for study extraction after model load; about 52 seconds
  for the whole Kaggle kernel
- Decision: KEEP — full feature extraction is authorized after metadata audit

### 2026-08-20 — STAGE4-FULL-COVERAGE-CORRECTION

- Type: data-artifact correction
- Root cause: the first aggregate consumed only 34 of 177 metadata parts,
  covering 850 of 4,407 studies; downstream placeholders made the first image
  result invalid
- Corrected metadata: 819,078 DICOM rows; 24,371 series; 4,407 studies; 177
  non-empty parts
- Corrected features: `(4407, 12, 384)` FP16; 0.999962 valid-stack fraction;
  all values finite
- Guard: aggregate and resume jobs now require the exact manifest part count
- Decision: KEEP correction; RETRACT the earlier 0.5317 partial-data image score

### 2026-08-20 — STAGE5-CORRECTED-IMAGE-CV — SUPERSEDED

- Type: full-data frozen-feature image cross-validation
- Gold evaluation rows: 58 per target
- Candidates: three plane-aware attention heads with different gold weights,
  dropout, hidden dimensions, and seeds
- Selected: `attention_w2_d01_h64`
- Gold-only OOF macro ROC AUC: 0.6678028453
- Bootstrap 95% interval: 0.6187806578 to 0.7139717237
- Candidate macro AUCs: 0.6678028453, 0.6355716003, 0.6411136629
- Public leaderboard: N/A — no submission made
- Factual correction: checkpoint selection repeatedly observed the evaluated
  fold's gold AUC. The 0.6678 value is biased and superseded by the strict
  fixed-epoch result below.
- Decision: RETRACT as a valid OOF estimate

### 2026-08-20 — STAGE5-NESTED-MULTIMODAL-FUSION — DIAGNOSTIC ONLY

- Type: leakage-safe nested OOF fusion
- Modalities: corrected image OOF, semantic report probability, rule report
  probability
- Selection: one simplex weight vector chosen on the development gold rows of
  each outer fold, never on that fold's evaluated labels; grid step 0.05
- Gold-only nested OOF macro ROC AUC: 0.7470527673
- Per-target AUC: ACL 0.8824; MCL 0.9161; medial meniscus 0.7548; lateral
  meniscus 0.6671; medial OA 0.6775; lateral OA 0.6867; PF OA 0.6422;
  effusion 0.6373; synovitis 0.7539; Baker's 0.8225; contusion 0.6910;
  fracture 0.8333
- Public leaderboard: N/A — no submission made
- Factual correction: the image component inherited checkpoint-selection bias,
  and test studies do not include reports. This result is neither a strict
  image OOF estimate nor a deployable submission model.
- Decision: RETRACT as a submission candidate; retain only as a modality
  complementarity diagnostic
- Caveat: only 58 gold rows; high statistical uncertainty. The 0.7597
  non-nested sensitivity blend is optimistic and is not a validated score.

### 2026-08-20 — STAGE5-REJECTED-FUSION-AND-PROBE

- Target-specific nested fusion, grid step 0.1: macro AUC 0.7233101249
- Nested masked-mean DINOv2 ridge probe: macro AUC 0.6066855785
- Public leaderboard: N/A — no submission made
- Decision: REJECT both; the extra target-wise flexibility is unstable with 58
  labels, and the linear probe underperforms the attention image head

### 2026-08-20 — STAGE5-STRICT-FIXED-EPOCH-BASELINE

- Type: corrected image-only OOF evaluation
- Correction: checkpoint duration fixed at 40 epochs; held-out fold gold is
  evaluated once and never used for checkpoint selection
- Teacher: original nested rule/MiniLM report labels
- Gold-only OOF macro ROC AUC: 0.6594587777
- Bootstrap 95% interval: 0.6035531879 to 0.7134158392
- Public leaderboard: N/A — no submission made
- Decision: KEEP as strict baseline; supersedes the biased 0.6678 result

### 2026-08-20 — STAGE5-HYBRID-TEACHER-SHARED-ATTENTION

- Type: strict image-only OOF evaluation
- Primary change: validated CC0 hybrid report-label teacher covering all 4,407
  studies; no report input at inference
- Teacher sparse-gold audit macro AUC: 0.899131
- Gold-only image OOF macro ROC AUC: 0.7292457997
- Per-target AUC: ACL 0.6642; MCL 0.6508; medial meniscus 0.6502; lateral
  meniscus 0.7491; medial OA 0.8372; lateral OA 0.8124; PF OA 0.7889;
  effusion 0.8534; synovitis 0.7204; Baker's 0.7844; contusion 0.6815;
  fracture 0.5583
- Public leaderboard: N/A — no submission made
- Decision: KEEP

### 2026-08-20 — STAGE5-TARGET-ATTENTION

- Type: strict image-only OOF architecture ablation
- Primary change: separate slice-attention distribution and classifier for each
  target; same hybrid teacher and deterministic folds as the shared head
- Gold-only OOF macro ROC AUC: 0.7044570900
- Public leaderboard: N/A — no submission made
- Decision: REJECT from macro ensemble; useful target-specific gains do not
  offset losses in MCL, contusion, and fracture

### 2026-08-20 — STAGE5-STRICT-IMAGE-ENSEMBLE

- Type: fixed, deployable OOF ensemble
- Inputs: strict original-teacher shared head and strict hybrid-teacher shared
  head
- Equal probability macro ROC AUC: 0.7465859983
- Equal rank macro ROC AUC: 0.7212415688
- Three-model probability ensemble including target attention: 0.7376146069
- Public leaderboard: N/A — no submission made
- Decision: KEEP the fixed 50/50 probability ensemble; reject the rank and
  three-model variants
- Inference: both checkpoint families are versioned as private Kaggle datasets;
  reports are not used at inference
