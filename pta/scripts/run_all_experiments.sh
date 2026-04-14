#!/bin/bash
# Run all experiments for Days 5-13: Gate 4 → M1 → M8 → M7 → Ablations → OOD Eval
#
# Usage:
#   cd /home/zhuzihou/dev/probe-then-act
#   nohup bash pta/scripts/run_all_experiments.sh > logs/run_all.log 2>&1 &
#
# Monitor:
#   tail -f logs/run_all.log

set -e

# Environment setup
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/zhuzihou/dev/probe-then-act:$PYTHONPATH
cd /home/zhuzihou/dev/probe-then-act

echo "============================================================"
echo "PROBE-THEN-ACT: Full Experiment Pipeline"
echo "Started: $(date)"
echo "============================================================"

# ============================================================
# Phase A: Gate 4 + Baselines (Day 5-8)
# ============================================================

echo ""
echo ">>> SKIPPING: Gate 4 already completed (Apr 8)"
echo ">>> SKIPPING: M1 seed=42 already completed (Apr 8-9, 500K)"

echo ""
echo ">>> Phase A2: M1 Reactive PPO (remaining seeds × 500K)"
for seed in 0 1; do
    echo ">>> M1 seed=$seed started: $(date)"
    python pta/scripts/train_baselines.py --method m1 --seed $seed --total-timesteps 500000
    echo ">>> M1 seed=$seed completed: $(date)"
done

echo ""
echo ">>> Phase A3: M8 Teacher PPO (3 seeds × 500K)"
for seed in 42 0 1; do
    echo ">>> M8 seed=$seed started: $(date)"
    python pta/scripts/train_baselines.py --method m8 --seed $seed --total-timesteps 500000
    echo ">>> M8 seed=$seed completed: $(date)"
done

# ============================================================
# Phase B: Core Method M7 (Day 8-13)
# ============================================================

echo ""
echo ">>> Phase B: M7 Probe-Then-Act (3 seeds × 500K)"
for seed in 42 0 1; do
    echo ">>> M7 seed=$seed started: $(date)"
    python pta/scripts/train_m7.py --seed $seed --total-timesteps 500000
    echo ">>> M7 seed=$seed completed: $(date)"
done

# ============================================================
# Phase B2: Ablations (Day 13)
# ============================================================

echo ""
echo ">>> Phase B2: Ablation - No Probe (2 seeds × 500K)"
for seed in 42 0; do
    echo ">>> No-Probe seed=$seed started: $(date)"
    python pta/scripts/train_m7.py --ablation no_probe --seed $seed --total-timesteps 500000
    echo ">>> No-Probe seed=$seed completed: $(date)"
done

echo ""
echo ">>> Phase B3: Ablation - No Belief (2 seeds × 500K)"
for seed in 42 0; do
    echo ">>> No-Belief seed=$seed started: $(date)"
    python pta/scripts/train_m7.py --ablation no_belief --seed $seed --total-timesteps 500000
    echo ">>> No-Belief seed=$seed completed: $(date)"
done

# ============================================================
# Phase C: OOD Evaluation (Day 14-16)
# ============================================================

echo ""
echo ">>> Phase C: OOD Evaluation"
echo ">>> Started: $(date)"
python pta/scripts/run_ood_eval_v2.py
echo ">>> OOD Evaluation completed: $(date)"

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "Finished: $(date)"
echo "Results: results/main_results.csv"
echo "============================================================"
