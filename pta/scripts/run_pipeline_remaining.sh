#!/bin/bash
# Run all remaining experiments sequentially.
# Usage: nohup bash pta/scripts/run_pipeline_remaining.sh > logs/pipeline_remaining.log 2>&1 &

set -e

source /home/zhuzihou/dev/Genesis/.venv/bin/activate
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/zhuzihou/dev/probe-then-act:$PYTHONPATH
cd /home/zhuzihou/dev/probe-then-act

echo "============================================================"
echo "PROBE-THEN-ACT: Remaining Pipeline"
echo "Started: $(date)"
echo "============================================================"

# ============================================================
# Phase A3 remainder: M8 Teacher PPO
# ============================================================
for seed in 0 1; do
    echo ">>> M8 seed=$seed started: $(date)"
    python pta/scripts/train_baselines.py --method m8 --seed $seed --total-timesteps 500000
    echo ">>> M8 seed=$seed completed: $(date)"
done

# ============================================================
# Phase B: Core Method M7
# ============================================================
for seed in 42 0 1; do
    echo ">>> M7 seed=$seed started: $(date)"
    python pta/scripts/train_m7.py --seed $seed --total-timesteps 500000
    echo ">>> M7 seed=$seed completed: $(date)"
done

# ============================================================
# Phase B2: Ablations
# ============================================================
for seed in 42 0; do
    echo ">>> M7 No-Probe seed=$seed started: $(date)"
    python pta/scripts/train_m7.py --ablation no_probe --seed $seed --total-timesteps 500000
    echo ">>> M7 No-Probe seed=$seed completed: $(date)"
done

for seed in 42 0; do
    echo ">>> M7 No-Belief seed=$seed started: $(date)"
    python pta/scripts/train_m7.py --ablation no_belief --seed $seed --total-timesteps 500000
    echo ">>> M7 No-Belief seed=$seed completed: $(date)"
done

# ============================================================
# Phase C: OOD Evaluation
# ============================================================
echo ">>> OOD Evaluation started: $(date)"
python pta/scripts/run_ood_eval_v2.py
echo ">>> OOD Evaluation completed: $(date)"

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "Finished: $(date)"
echo "============================================================"
