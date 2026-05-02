# Seed Sensitivity Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a concise evidence record for current seed coverage, pre-submission gates, and confirmed seed-sensitive training/evaluation areas.

**Architecture:** Add one focused record under `docs/30_records/` so the paper decision state is separate from operational runbooks. Use existing result files, DLC worker records, and artifact manifests as evidence; avoid claiming root cause where only sensitivity has been measured.

**Tech Stack:** Markdown documentation, existing PTA result artifacts, `git diff --check` for formatting verification.

---

### Task 1: Add Seed Sensitivity Record

**Files:**
- Create: `docs/30_records/SEED_SENSITIVITY_STATUS.md`

- [ ] **Step 1: Add the record file**

Create `docs/30_records/SEED_SENSITIVITY_STATUS.md` with these sections:

```markdown
# Seed Sensitivity Status

Date: 2026-05-02

## Purpose

## Evidence Sources

## Current Seed Coverage

## Pre-Submission Gate Runs

## Confirmed Seed Sensitivity

## Not Yet Confirmed As Seed Sensitivity

## Implication For G3/G4
```

- [ ] **Step 2: Populate factual coverage tables**

Use:

```text
results/ood_eval_per_seed.csv
results/presub/audit_encoder_m7_pta_s42_ood_elastoplastic.json
results/presub/audit_probe_*_seed123.json
results/dlc/runs/*.json
/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/**/artifact_manifest.json
```

Record that `m1_reactive`, `m7_pta`, `m7_noprobe`, and `m7_nobelief` have policy seeds `0, 1, 42` across five splits; `m8_teacher` has seed `42`; G1 used audit seed `123`; G2 used policy seed `42` with encoder seeds `11, 22, 33`.

- [ ] **Step 3: State sensitivity classification conservatively**

Write these classifications:

```text
Confirmed seed-sensitive evaluation: m7_pta policy seed 42 under encoder-sensitivity audit on ood_elastoplastic.
Reported high-variance policy-seed evidence: m1_reactive, m7_pta, m7_noprobe, and m7_nobelief on ood_elastoplastic in the three-seed OOD table.
Not newly diagnosed by G2: m1_reactive, m7_noprobe, m7_nobelief, m8_teacher.
```

- [ ] **Step 4: Verify formatting**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Do not commit automatically**

Leave the documentation change unstaged unless the user explicitly requests a commit.
