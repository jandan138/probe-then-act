# NIPS2026 Submission Materials Design

## Goal

Prepare a reproducible Full Bundle for the NeurIPS/NIPS 2026 anonymous submission from the existing dual-venue paper tree.

## Scope

The bundle must include the submission PDF, flattened LaTeX source package, source manifest, build/readiness report, and supplemental/reproducibility checklist. Generated bundle artifacts stay out of git by default; the reproducible packaging script and Makefile target are source changes.

## Approach

Use the existing `paper/venues/neurips` entry point as the source of truth. Add a small packaging script under `paper/scripts/` and a `make nips2026` target that builds the NeurIPS paper, flattens the source, validates the result, and writes artifacts under `paper/submissions/nips2026/`.

## Components

- `paper/scripts/package_nips2026.py`: orchestrates bundle creation, validation, and report generation.
- `paper/Makefile`: exposes `make nips2026` as the user-facing packaging command.
- `paper/submissions/nips2026/`: ignored generated output containing the PDF, source zip, manifest, readiness report, and supplemental checklist.

## Validation

The package command verifies that the NeurIPS PDF and flattened source compile, scans LaTeX and BibTeX logs for errors, undefined references, undefined citations, and suspicious stale wording, checks expected figure and conclusion labels in `main.aux`, checks anonymous-submission markers, records file sizes and paths, and reports unavailable external PDF tools instead of pretending those checks ran.

## Success Criteria

- `make nips2026` exits with status 0 when all required checks pass.
- The generated bundle contains a main PDF, source zip, manifest, readiness report, and supplemental/reproducibility checklist.
- The readiness report records exact commands, artifact paths, warnings, and any skipped optional PDF-tool checks.
- The workflow does not modify scientific claims, results, citations, or paper wording except for packaging metadata.

## Non-Goals

- Do not create camera-ready author metadata.
- Do not fabricate supplemental experiments, checklist answers, or missing venue requirements.
- Do not track bulky generated submission artifacts unless explicitly requested.
