# Novelty Check & Literature Survey Report

> **Date:** 2026-04-03
> **Status:** PASSED — no direct competitor found

---

## Verdict

**Probe-Then-Act is novel enough for T-RL submission.** No existing paper combines ALL of:
1. Active probing for material identification
2. Latent physical belief (z) + calibrated uncertainty (sigma)
3. Manipulation of deformable/granular materials
4. Uncertainty-aware policy conditioning + risk head
5. Cross-material benchmark with OOD evaluation axes

---

## Closest Partial Competitors (must cite and differentiate)

| Paper | Year/Venue | What it shares | What it lacks |
|-------|-----------|---------------|--------------|
| **SCONE** | CoRL 2023 | Active perception (stir) + food scooping | No latent belief model, no uncertainty, fixed probing |
| **AdaptiGraph** | RSS 2024 | Cross-material adaptive dynamics (GNN) | Passive adaptation, no active probing, no uncertainty |
| **RoboPack** | RSS 2024 | Tactile + latent physics (recurrent GNN) | Rigid objects only, no probing, no uncertainty |
| **ASID** | ICLR 2024 Oral | Active exploration for system ID | Refines simulator params, not latent beliefs for policy; rigid-body only |
| **PrivilegedDreamer** | ICRA 2025 | Latent belief conditioned policy (HIP-MDP) | Locomotion/rigid control, no deformable, no active probing |

## Key Differentiation Argument

> "Unlike prior work that either uses active probing without belief models (SCONE), or infers latent physics passively (AdaptiGraph, RoboPack), or handles hidden parameters only in rigid settings (ASID, PrivilegedDreamer), Probe-Then-Act is the first framework that explicitly learns an active probing policy to build a latent physical belief with calibrated uncertainty, and conditions an uncertainty-aware manipulation policy on this belief, enabling robust cross-material tool use."

---

## Architectural Inspirations to Cite

| Paper | Year/Venue | What to borrow |
|-------|-----------|---------------|
| **SITT** | ICLR 2025 Spotlight | Teacher-student training with asymmetry awareness |
| **GoFlow** | ICML 2025 | Learned domain randomization via normalizing flows |
| **ActivePusher** | 2025 | Uncertainty-aware kinodynamic planning |
| **CURA-PPO** | 2026 | Distributional uncertainty for active perception |
| **Latent Intuitive Physics** | ICLR 2024 | Latent prior distribution for hidden physics |

## Benchmark Landscape

| Benchmark | Materials | MPM | Cross-material OOD | Our advantage |
|-----------|----------|-----|-------------------|---------------|
| SoftGym | Rope/cloth/fluid | No | No | We cover granular+liquid+elastoplastic |
| FluidLab | Liquids | Partial | No | We have systematic OOD axes |
| PlasticineLab | Plasticine | DiffTaichi | No | We cover multiple material families |
| DaXBench | Multi | JAX | No | We use MPM + standardized tasks |
| **Ours** | Sand/Snow/ElastoPlastic/Liquid | Genesis MPM | 5 OOD axes | First cross-material MPM benchmark |

---

## Recommended Related Work Structure

1. **Active sensing for manipulation** — SCONE, Contact SLAM, ASID, GmClass
2. **Belief-conditioned RL under hidden dynamics** — PrivilegedDreamer, Latent Intuitive Physics, belief-space planning
3. **Cross-material/deformable manipulation** — AdaptiGraph, RoboPack, FluidLab
4. **Uncertainty-aware robot learning** — CURA-PPO, ActivePusher, GUAPO
5. **Teacher-student learning** — SITT, privileged info locomotion
6. **Multi-physics simulation benchmarks** — Genesis, SoftGym, FluidLab, DaXBench
