# Docs Taxonomy Migration Plan

> **Date:** 2026-04-09
> **Purpose:** Reorganize `docs/` by document authority and lifecycle rather than by a flat mixed list.

## 1. Taxonomy rule

Only `00_foundation/` and `10_protocols/` should be treated as source-of-truth documentation.

- `20_planning/` = live coordination state
- `30_records/` = historical execution records
- `40_investigations/` = exploratory diagnostics and side-tracks
- `50_reports/` = synthesized analytical reports

Anything outside foundation / protocols must not silently override project constraints.

## 2. Old → new mapping

| Old path | New path |
|---|---|
| `docs/00_PROJECT_BRIEF.md` | `docs/00_foundation/00_PROJECT_BRIEF.md` |
| `docs/01_REPO_BLUEPRINT.md` | `docs/00_foundation/01_REPO_BLUEPRINT.md` |
| `docs/02_EXECUTION_PLAYBOOK.md` | `docs/10_protocols/02_EXECUTION_PLAYBOOK.md` |
| `docs/03_EXPERIMENT_PROTOCOL.md` | `docs/10_protocols/03_EXPERIMENT_PROTOCOL.md` |
| `docs/04_VALIDATION_GATES.md` | `docs/10_protocols/04_VALIDATION_GATES.md` |
| `docs/05_TINY_TASK_OVERFIT_PROTOCOL.md` | `docs/10_protocols/05_TINY_TASK_OVERFIT_PROTOCOL.md` |
| `docs/06_NOVELTY_CHECK_REPORT.md` | `docs/50_reports/06_NOVELTY_CHECK_REPORT.md` |
| `docs/07_CURRENT_BLOCKERS_AND_ACTIONS.md` | `docs/20_planning/07_CURRENT_BLOCKERS_AND_ACTIONS.md` |
| `docs/08_48HR_SPRINT_RESULTS.md` | `docs/30_records/08_48HR_SPRINT_RESULTS.md` |
| `docs/09_NEXT_STEPS_PLAN.md` | `docs/20_planning/09_NEXT_STEPS_PLAN.md` |
| `docs/10_TASK_DESIGN_INVESTIGATION.md` | `docs/40_investigations/10_TASK_DESIGN_INVESTIGATION.md` |
| `docs/11_BOWL_TOOL_INVESTIGATION.md` | `docs/40_investigations/11_BOWL_TOOL_INVESTIGATION.md` |
| `docs/12_BOWL_TRANSPORT_DIAGNOSIS_RUNBOOK.md` | `docs/10_protocols/12_BOWL_TRANSPORT_DIAGNOSIS_RUNBOOK.md` |
| `docs/GATE4_TRAINING_REPORT.md` | `docs/30_records/GATE4_TRAINING_REPORT.md` |
| `docs/IK_MINIMAL_REPRO.md` | `docs/40_investigations/IK_MINIMAL_REPRO.md` |
| `docs/Week1_environment_bootstrap.md` | `docs/30_records/Week1_environment_bootstrap.md` |

## 3. Reference update scope

Files that must be updated during the migration:

- `README.md`
- `CLAUDE.md`
- `REPO_DOCS_INDEX.md`
- `RESEARCH_BRIEF.md`
- `docs/00_foundation/01_REPO_BLUEPRINT.md`
- `docs/10_protocols/02_EXECUTION_PLAYBOOK.md`
- `docs/10_protocols/04_VALIDATION_GATES.md`
- `docs/10_protocols/05_TINY_TASK_OVERFIT_PROTOCOL.md`
- `docs/20_planning/07_CURRENT_BLOCKERS_AND_ACTIONS.md`
- `docs/20_planning/09_NEXT_STEPS_PLAN.md`
- `docs/30_records/08_48HR_SPRINT_RESULTS.md`
- `docs/30_records/Week1_environment_bootstrap.md`
- `docs/40_investigations/10_TASK_DESIGN_INVESTIGATION.md`
- `docs/40_investigations/11_BOWL_TOOL_INVESTIGATION.md`
- `pta/envs/wrappers/joint_residual_wrapper.py`
- `pta/configs/overfit/sand_tiny_task.yaml`

## 4. Validation checklist

- every migrated document exists exactly once at its new path;
- repo-wide references to old flat `docs/...` paths are removed or intentionally preserved only here;
- the bowl execution runbook link from `11_BOWL_TOOL_INVESTIGATION.md` still resolves;
- `README.md`, `CLAUDE.md`, `REPO_DOCS_INDEX.md`, and `RESEARCH_BRIEF.md` point to the new taxonomy paths;
- `01_REPO_BLUEPRINT.md` no longer shows a flat `docs/` tree.
