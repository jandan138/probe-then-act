# Research Findings

## 2026-04-26 Result-to-Claim: OOD Main Claims Not Supported

**Status:** `claim_supported=no` (`pending Codex MCP review`; corroborated by primary + auxiliary research reviewers).

### Evidence Checked

- Fresh corrected OOD artifacts are complete: `results/ood_eval_per_seed.csv` has `35/35` expected rows and `results/main_results.csv` has `15` aggregate rows.
- Coordinator reconciliation reports OOD complete with no missing or extra keys.
- No evaluated row reported failed Genesis episodes: `n_failed_episodes_sum=0` throughout.
- Evaluated systems: `M1 reactive PPO` (3 seeds), `M7 Probe-Then-Act` (3 seeds), `M8 privileged teacher` (1 seed).
- Missing systems for original claims: `M2 RNN-PPO`, `M6 uncertainty ablation`, M7 component ablations.

### Main Numeric Verdict

| Split | M7-M1 Transfer | M7-M1 Spill | M7-M1 Success | Verdict |
|---|---:|---:|---:|---|
| ID sand | -0.2214 | +0.2194 | -0.0667 | M7 worse |
| OOD snow | -0.1329 | +0.1440 | -0.1667 | M7 worse |
| OOD elastoplastic | +0.1471 | -0.1471 | +0.1667 | M7 better but seed-unstable |
| OOD sand soft | -0.2221 | +0.2220 | 0.0000 | M7 worse on transfer/spill |
| OOD sand hard | -0.1351 | +0.1385 | 0.0000 | M7 worse on transfer/spill |
| All OOD average | -0.0858 | +0.0894 | 0.0000 | M7 worse overall |

### Claim Verdicts

- **Claim A: Active probing improves cross-material robustness.** Not supported. M7 is not consistently better than M1 on OOD materials, and M2 is absent.
- **Claim B: Explicit belief beats passive memory.** Not supported / untestable. M2 RNN-PPO is absent.
- **Claim C: Uncertainty improves failure avoidance.** Not supported / untestable. M6 and uncertainty diagnostics are absent, and the current OOD run has zero failed episodes.
- **Claim D: Method generalizes beyond the training setup.** Not supported. M7 improves only on elastoplastic, not at least two OOD splits, and that improvement is unstable across seeds.

### Postmortem

- The broad paper narrative is contradicted by the corrected OOD table: M7 hurts ID performance and most OOD parameter/material settings compared with M1.
- The only positive split is elastoplastic, where seed 42 drives most of the advantage; seed 0 is worse than M1 and seed 1 is roughly tied.
- The current experiment does not isolate whether the probe phase, belief latent, or PPO training instability causes the degradation.
- Because M2 and M6 were not run, no passive-memory or uncertainty claims can be made.

### Constraints for Future Attempts

- Do not write a paper claiming broad Probe-Then-Act robustness from the current OOD table.
- Do not claim belief or uncertainty contributions without M2/M6 or direct ablations.
- Do not use M8 as a clean upper bound; it has only one seed and fails on elastoplastic.
- Treat elastoplastic as a hypothesis-generating signal, not confirmed evidence.

### Next Research Direction

Focus the next cycle on diagnosis and salvage. The selected strategy is **Option 1: Ablation-First Diagnostic**.

### 2026-04-26 Direction Decision

- **Selected**: Option 1, ablation-first diagnosis.
- **Reason**: It is the highest-value next step because the corrected OOD table already refutes broad PTA robustness, and only component ablations can tell whether the probe/belief mechanism is salvageable or should be abandoned.
- **Do first**: train `m7_noprobe` and `m7_nobelief` for seeds `42`, `0`, and `1`, then rerun corrected resumable OOD.
- **Stop / pivot gate**: if neither ablation explains or repairs M7's regressions, pivot away from broad PTA robustness rather than adding more baselines or paper-writing claims.
- **Deferred alternatives**: elastoplastic-only claim and negative/failure-analysis framing remain fallback options after ablation evidence.

### 2026-04-27 R001 Local Ablation Result

- `m7_noprobe seed=42` completed locally for `500000` timesteps.
- Training/eval curve is positive on the ID training setting: best eval `25735.26 +/- 1.74 @440k`, final eval `23634.73 +/- 2.03 @500k`.
- Compared with prior seed-42 training curves, `m7_noprobe` is stronger and more stable than full M7 (`24629.75 +/- 37.24` best, `17328.08 +/- 2747.15` final) and competitive with M1 (`24840.94 +/- 53.29` best, `22951.69 +/- 383.32` final).
- Interpretation: this is an encouraging ID/training sanity result and suggests the probe phase may contribute to full M7's degradation. It is not OOD evidence yet; do not claim mechanism support until corrected ablation OOD finishes.
- Local ARIS cron entries are paused, so no local automatic OOD or next-run launch should occur after R001.

## 2026-04-29 Post-Ablation Result-to-Claim: Broad PTA Robustness Rejected

**Status:** `claim_supported=no` for broad Probe-Then-Act OOD robustness (available result-to-claim review completed; Codex MCP-specific review remains unavailable and is not blocking routing).

### Evidence Checked

- Corrected OOD v2 is now expanded with `m7_noprobe` and `m7_nobelief`: `results/ood_eval_per_seed.csv` has `65` per-seed rows and `results/main_results.csv` has `25` aggregate rows.
- All rows report `n_failed_episodes=0`; the verdict is not caused by Genesis crash accounting.
- Ablation OOD used the intended final checkpoints: `checkpoints/m7_pta_noprobe_seed{seed}/m7_pta_final.zip` and `checkpoints/m7_pta_nobelief_seed{seed}/m7_pta_final.zip`.
- Metrics remain transfer higher is better, spill lower is better, success higher is better.

### Post-Ablation Numeric Verdict

| System | All-OOD Transfer Delta vs M1 | All-OOD Spill Delta vs M1 | All-OOD Success Delta vs M1 | Verdict |
|---|---:|---:|---:|---|
| M7 full | -0.0858 | +0.0894 | 0.0000 | Worse overall; only elastoplastic improves |
| M7 no-probe | -0.2733 | +0.1365 | -0.3333 | Much worse; removing probe damages M7 |
| M7 no-belief | -0.1279 | +0.0693 | -0.2167 | Better than no-probe on some splits, still worse than M1 |

### Component Interpretation

- The ablations do not salvage the original claim. Every M7 variant underperforms M1 on all-OOD average transfer, and every M7 variant has worse all-OOD spill than M1.
- Removing the probe hurts substantially, so the probe is not simply the source of M7's degradation. It helps within the M7 design but does not produce broad robustness over M1.
- Removing the belief sometimes improves snow/ID-like behavior in seeds `42` and `0`, but seed `1` collapses and the aggregate remains below M1.
- Full M7's elastoplastic advantage remains the only positive split, but it is seed-unstable and cannot support a broad OOD claim.
- Failure is best described as architecture/training instability or a poor robustness/efficiency tradeoff relative to the reactive PPO baseline, not a single removable module.

### Reviewer Consensus

- Reject the broad claim that active Probe-Then-Act improves OOD robustness over reactive PPO.
- A narrow internal-mechanism statement is supportable only with metric scope: probing helps relative to `m7_noprobe`, while belief helps transfer/success relative to `m7_nobelief` but not all-OOD spill.
- A paper-facing PTA robustness story is not currently supportable; M2/RNN would not rescue the broad claim unless a new repaired method is introduced.
- Additional elastoplastic-only runs are only worth doing if the project explicitly pivots to a narrow dynamics-adaptation story.

### Updated Routing

1. Stop the broad PTA robustness paper path for the current method.
2. Do not launch M2/RNN, uncertainty/M6, or paper writing as automatic next steps.
3. If continuing this project, choose explicitly between a lightweight failure-analysis note and a new method/pivot.
4. If trying to salvage a narrow elastoplastic claim, first require a concrete hypothesis and extra-seed confirmation plan; otherwise treat the signal as hypothesis-generating only.
