# CLAUDE.md — Probe-Then-Act Project Conventions

## Project Overview
Paper-first research codebase for: "Probe-Then-Act: Active Tactile System Identification for Robust Cross-Material Robot Tool Use in Multi-Physics Simulation"
- Target: IEEE T-RL (IROS 2026 / CASE 2026 transfer, deadline 2026-04-30)
- Simulator: Genesis (MPM + Rigid coupling)
- Robot: Franka Panda with scoop tools
- Training: Teacher-student (privileged RL → distillation + fine-tuning)

## Environment Setup
```bash
# Activate venv
source /home/zhuzihou/dev/Genesis/.venv/bin/activate

# Required env vars for WSL2 headless rendering
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

## Code Structure
See `docs/00_foundation/01_REPO_BLUEPRINT.md` for canonical structure. Key modules:
- `pta/envs/` — Genesis environments, materials, sensors, rewards, metrics
- `pta/models/` — Encoders, probe policy, belief encoder, task policy
- `pta/training/` — RL, IL, distillation pipelines
- `pta/eval/` — OOD evaluation, tables, figures
- `pta/configs/` — All experiment configs (YAML)

## Conventions
- English-only code surface (comments, variable names, commit messages)
- Experiment names: `{task}.{method}.{split}.{seed}` (e.g., `scoop_transfer.pta.ood_material.seed3`)
- Checkpoint names: `{task}_{method}_{stage}_{step}.pt`
- Every result must trace to a config file
- Student observation must NEVER access privileged material parameters
- Report mean ± std over seeds, never single best seed

## Running Experiments
```bash
# Sanity check
python pta/scripts/sanity_check_env.py

# Train teacher
python pta/scripts/train_teacher.py --config pta/configs/train/teacher_rl.yaml

# Train student
python pta/scripts/train_student.py --config pta/configs/train/student_bc.yaml

# Evaluate
python pta/scripts/run_eval_main.py --config pta/configs/eval/paper_main.yaml
```

## ARIS Integration
- `RESEARCH_BRIEF.md` — Input for `/research-lit` and `/novelty-check`
- `refine-logs/EXPERIMENT_PLAN.md` — Input for `/experiment-bridge`
- `NARRATIVE_REPORT.md` — Input for `/paper-writing`
- `docs/30_records/CRON_ARIS_ORCHESTRATOR_RUNBOOK.md` — Cron install, state, log, and recovery steps
- `docs/30_records/DLC_EXECUTION_RUNBOOK.md` — DSW/PAI-DLC submitter and worker usage for bounded train/eval jobs
- `docs/superpowers/specs/2026-04-26-dlc-execution-layer-design.md` — DLC/ARIS/Auto repo boundary decision
- See `Auto-claude-code-research-in-sleep/projects/probe-then-act/AUTOMATION_PLAN.md` for full schedule

## Key Design Documents
1. `docs/00_foundation/00_PROJECT_BRIEF.md` — Research vision, method, baselines
2. `docs/00_foundation/01_REPO_BLUEPRINT.md` — Repo architecture
3. `docs/10_protocols/02_EXECUTION_PLAYBOOK.md` — Week-by-week execution
4. `docs/10_protocols/03_EXPERIMENT_PROTOCOL.md` — Hypotheses, experiment matrix, metrics
5. `docs/10_protocols/04_VALIDATION_GATES.md` — Gate 0–5 workflow (Gate 0 **PASSED**)
6. `docs/10_protocols/05_TINY_TASK_OVERFIT_PROTOCOL.md` — Tiny-task overfit before scale-up

## Current Status (2026-04-26)
- **Corrected OOD v2 COMPLETE**: `results/ood_eval_per_seed.csv` has 35/35 expected rows and `results/main_results.csv` has 15 aggregate rows.
- **Result-to-claim verdict**: original broad Probe-Then-Act claims are **not supported** by the corrected OOD table.
- **Main finding**: M7 improves only on `ood_elastoplastic`; it is worse than M1 on ID, snow, and sand parameter shifts, and worse on all-OOD transfer/spill average.
- **Direction decision**: choose Option 1, **Ablation-First Diagnostic**, before any paper writing or broader experiment expansion.
- **DLC acceleration path**: use the probe repo DLC layer for bounded `smoke_env`, `train_ablation`, and `eval_ood` jobs when the repos are uploaded to a DSW machine; do not run cron/ARIS/agent tooling inside DLC workers.
- **Do not write paper claims yet**: M2/RNN, M6/uncertainty, and M7 ablations are missing.
- **Next**: run approved `m7_noprobe` and `m7_nobelief` seeds `42/0/1`, rerun corrected resumable OOD, then run result-to-claim again before claiming any PTA mechanism.
