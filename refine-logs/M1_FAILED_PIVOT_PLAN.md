# M1 Failed → Pivot Plan

**Project:** Probe-Then-Act  
**Date:** 2026-04-06  
**Status:** **M1 gate failed** — pivot required before more large-scale RL runs.

---

## 1) Decision Summary

We will **stop treating this as a reward-tuning problem** and instead treat it as a **feasibility + task-definition problem**.

Current evidence indicates that the main bottleneck is likely a combination of:
1. **End-effector / geometry mismatch** for scooping bulk material
2. **Over-hard control formulation** (7-D action at high control frequency)
3. **Reward and entropy settings that favor approach-only behavior**
4. **No verified scripted feasibility baseline**

**Implication:** Do **not** launch more 1M–10M step runs on the current setup.

---

## 2) What Failed in M1

### Observed facts
- All trained methods are still at **0% success** after the first training round.
- OOD evaluation also shows **0% success** across methods.
- Teacher v2 entered an **approach-only local optimum**, plateaued early, and never discovered real scoop-transfer behavior.
- Action std increased substantially during Teacher v2, consistent with overly noisy exploration.
- Physical feasibility of the current **Franka gripper-only** setup has **not yet been verified** with a scripted trajectory.

### Working diagnosis
The current setup is most consistent with a **geometry-and-feasibility bottleneck**, compounded by an unnecessarily difficult control formulation and unstable exploration.

---

## 3) Pivot Objective

Replace the current question:
> “Can PPO learn scoop-transfer in the current setup?”

with the new question:
> “Can a physically plausible scoop-tool setup complete scoop-transfer under a simplified task, and can RL improve on a working scripted controller?”

This is now the shortest path to a publishable and defensible result.

---

## 4) Immediate Pivot Actions (next 48 hours)

### A. Fix physical plausibility first
1. **Add a scoop attachment** to the robot end-effector.
   - Simple rigid concave geometry is enough for v1.
   - Do **not** continue with stock Panda fingers as the primary scooping geometry.

2. **Run a scripted feasibility test** before any new RL campaign.
   - Hand-code an open-loop scoop trajectory.
   - Record: `penetration depth`, `lifted mass`, `transferred mass`, `spill`, `success`.

3. **Make sand-only the mainline environment.**
   - Drop liquid from the main training track for now.
   - Freeze material family, container geometry, and initial pose distribution for M1-reboot.

### B. Simplify control
4. **Reduce action space** from 7-D EE delta to a simpler form.
   - Preferred v1: **3-D position-only action** with phase-fixed orientation.
   - Better if easy: waypoint / trajectory-parameter action instead of per-step 7-D control.

5. **Lower policy frequency / add action repeat.**
   - Keep physics fast.
   - Slow policy decisions so the agent makes tens of meaningful decisions per episode, not hundreds of fragile micro-decisions.

6. **Increase effective episode physical duration.**
   - Ensure the robot has enough time to perform a full approach → insert → drag → lift → move → dump sequence.

### C. Fix training setup only after A/B succeed
7. **Set PPO entropy pressure to minimal.**
   - Remove the current high-entropy setting.
   - Start from lower exploration scale.

8. **Rebalance reward to make scooping/lifting/transfer dominant.**
   - Approach reward should become guidance-only.
   - Transfer-related rewards and success bonus should dominate.

9. **Use scripted controller as the base policy.**
   - Preferred next step: **residual RL** on top of a working scripted scoop.
   - Demos / BC / DAPG are secondary options if residual RL underperforms.

---

## 5) Acceptance Gates

### Gate G1 — Feasibility Gate (must pass)
The pivot is only valid if the new scoop-tool setup can pass a scripted baseline test.

**Minimum pass criteria:**
- Non-zero `lifted mass`
- Non-zero `transferred mass`
- Repeatable success on a small fixed evaluation set

If G1 fails, **do not** proceed to RL. Rework geometry, tool pose, friction, or task setup.

### Gate G2 — Learnability Gate
Only after G1 passes, run a small teacher / residual training test.

**Pass criteria:**
- Training beats the scripted baseline on at least one metric **or**
- Training clearly improves robustness / consistency over the scripted policy

If G2 fails, reduce task complexity again before adding more methods.

---

## 6) What We Will Not Do

- No more long PPO runs on the current gripper-only setup
- No more liquid as a core benchmark in this phase
- No more interpretation of `mean return` as the primary success signal
- No expansion to broader OOD benchmarking until scripted feasibility is proven

---

## 7) Metrics That Matter From Now On

Primary metrics:
- `scripted_success_rate`
- `lifted_mass`
- `transferred_mass`
- `spill_fraction`
- `contact_duration`

Secondary metrics:
- episode return
- explained variance
- training FPS

Rule: **task-stage metrics beat reward curves** during this recovery phase.

---

## 8) Deliverables for the Team

### By end of Pivot Day 1
- Scoop attachment asset integrated
- Scripted scoop trajectory implemented
- Feasibility evaluation script producing stage-wise metrics

### By end of Pivot Day 2
- Simplified sand-only environment frozen
- Reduced action-space version running
- PPO config cleaned up
- Go / no-go decision for residual RL

---

## 9) One-Line Management Summary

**M1 failed because the current task is not yet a good learning problem. The pivot is to first prove physical feasibility with a scoop tool and scripted baseline, then restart learning on a simplified, physically plausible setup.**

---

## Source Documents
- `refine-logs/LITERATURE_RESEARCH_REPORT.md` — 5-agent parallel research (reward, demo, action space, Genesis, PPO)
- `refine-logs/EXPERIMENT_PLAN.md` — original experiment blocks + decision gates
- `EXECUTION_LOG.md` (in Auto-claude-code-research-in-sleep) — full timeline + Phase 3.5 post-mortem
