# Probe-Then-Act

Probe-Then-Act: Active Tactile System Identification for Robust Cross-Material Robot Tool Use in Multi-Physics Simulation.

## Quick Start

```bash
# Activate Genesis environment
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Sanity check
python pta/scripts/sanity_check_env.py

# Train teacher (privileged upper bound)
python pta/scripts/train_teacher.py --total-timesteps 500000 --seed 42

# Train baseline
python pta/scripts/train_student.py --method reactive_ppo --seed 42
```

## Documentation

Read in this order:
1. `docs/00_foundation/00_PROJECT_BRIEF.md` — Research vision
2. `docs/00_foundation/01_REPO_BLUEPRINT.md` — Repository architecture
3. `docs/10_protocols/02_EXECUTION_PLAYBOOK.md` — Execution plan
4. `docs/10_protocols/03_EXPERIMENT_PROTOCOL.md` — Experiment protocol
5. `docs/50_reports/06_NOVELTY_CHECK_REPORT.md` — Literature survey results

## Project Status

See `results/manifests/runs.csv` for experiment tracking.
