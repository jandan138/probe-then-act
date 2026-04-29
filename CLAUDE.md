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

## Current Status (2026-04-29)
- **Corrected OOD v2 + ablation OOD COMPLETE**: `results/ood_eval_per_seed.csv` has `65` per-seed rows and `results/main_results.csv` has `25` aggregate rows.
- **Post-ablation result-to-claim verdict**: broad Probe-Then-Act OOD robustness is **not supported**.
- **Main finding**: M7 full, `m7_noprobe`, and `m7_nobelief` all underperform M1 on all-OOD average transfer/spill. Full M7 only improves on `ood_elastoplastic`, and that signal is seed-unstable.
- **Ablation interpretation**: probing helps relative to `m7_noprobe`; belief helps transfer/success relative to `m7_nobelief` but not all-OOD spill. Neither component produces broad robustness over M1 or localizes a simple repair.
- **Direction decision**: no-go for the current PTA paper claim. Do not launch M2/RNN, M6/uncertainty, elastoplastic expansion, or paper-writing automatically.
- **DLC status**: R001 finished locally; R002-R006 finished via the DLC handoff; R007 ablation OOD finished locally. DLC workers remain bounded compute workers only.
- **Automation status**: local ARIS cron entries are intentionally paused; do not rely on local cron for automatic advancement.
- **Next**: require an explicit human choice between a lightweight failure-analysis writeup, a narrow elastoplastic salvage plan, or a new method/pivot before spending more compute.
