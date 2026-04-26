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

1. Train M7 ablations (`no_probe`, `no_belief`) to identify whether probing/belief causes the observed regressions.
2. Run or implement M2 RNN-PPO only if the paper will retain an explicit-belief-vs-memory claim.
3. Re-test elastoplastic with additional seeds only if ablations suggest the M7 gain is mechanistic rather than random.
4. If broad OOD robustness remains negative, narrow the contribution to a failure analysis or pivot away from the current PTA mechanism.
