# NIPS2026 Submission Materials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible NeurIPS/NIPS 2026 Full Bundle containing the anonymous PDF, flattened LaTeX source zip, manifest, readiness report, and supplemental/reproducibility checklist.

**Architecture:** Keep `paper/venues/neurips` as the source of truth. Add one Python packaging script that calls the existing flattening flow, compiles the flattened source, creates the archive, and writes reports under an ignored `paper/submissions/nips2026/` directory.

**Tech Stack:** GNU Make, Python 3 standard library, latexmk/pdflatex/BibTeX, existing NeurIPS 2026 LaTeX style.

---

### Task 1: Packaging Entry Point

**Files:**
- Modify: `paper/Makefile`
- Modify: `paper/.gitignore`
- Create: `paper/scripts/package_nips2026.py`

- [ ] **Step 1: Ignore generated submission bundle**

Add `submissions/` to `paper/.gitignore` so generated package outputs do not appear as source changes.

- [ ] **Step 2: Add Makefile target**

Add `nips2026` to `.PHONY` and make it depend on `neurips` before running `python3 scripts/package_nips2026.py`.

- [ ] **Step 3: Add packaging script**

Implement `paper/scripts/package_nips2026.py` with these responsibilities: clean/create `submissions/nips2026/`, copy the built NeurIPS PDF, run `scripts/flatten.py --venue neurips`, copy `neurips_2026.sty` into flattened source, compile flattened source with latexmk, zip the flattened source, scan logs/aux files, and write `manifest.json`, `READINESS_REPORT.md`, and `SUPPLEMENTAL_CHECKLIST.md`.

- [ ] **Step 4: Verify syntax**

Run: `python3 -m py_compile scripts/package_nips2026.py`

Expected: exit 0.

### Task 2: Bundle Generation

**Files:**
- Generated: `paper/submissions/nips2026/`

- [ ] **Step 1: Generate bundle**

Run: `make nips2026`

Expected: exit 0 and generated output under `paper/submissions/nips2026/`.

- [ ] **Step 2: Inspect generated artifacts**

Confirm these files exist: `pta_nips2026_main.pdf`, `pta_nips2026_source.zip`, `manifest.json`, `READINESS_REPORT.md`, `SUPPLEMENTAL_CHECKLIST.md`, and `source/main.tex`.

### Task 3: Final Verification

**Files:**
- Read: `paper/venues/neurips/build/main.log`
- Read: `paper/venues/neurips/build/main.blg`
- Read: `paper/venues/neurips/build/main.aux`
- Read: `paper/submissions/nips2026/READINESS_REPORT.md`

- [ ] **Step 1: Run source diff whitespace check**

Run: `git diff --check`

Expected: no output and exit 0.

- [ ] **Step 2: Check git status**

Run: `git status --short --branch`

Expected: source changes only for the spec, plan, Makefile, `.gitignore`, and packaging script; generated `paper/submissions/` remains ignored.

- [ ] **Step 3: Report evidence**

Report exact artifact paths, verification commands, and any remaining warnings or skipped optional checks.
