# 01_REPO_BLUEPRINT

## 1. Purpose
This document defines the **canonical repository structure**, **module boundaries**, **naming conventions**, and **engineering rules** for the `probe-then-act` project.

This repo is **not** a generic robotics playground. It is a **paper-first research codebase** whose job is to produce:
1. a stable benchmark,
2. a strong method,
3. a reproducible evaluation protocol,
4. and publication-ready artifacts.

---

## 2. Repository design principles

### 2.1 Paper-first structure
Every directory should map to one of:
- `environment / benchmark`
- `method / model`
- `training`
- `evaluation`
- `reproducibility`
- `paper assets`

If a directory does not help one of those, it should probably not exist.

### 2.2 Strict separation of concerns
Do **not** mix:
- scene building and reward code,
- model code and experiment config,
- evaluation code and training code,
- plotting code and metric computation.

### 2.3 Reproducibility over convenience
Anything that influences a result must be either:
- in a config file,
- in a checked-in script,
- or logged to disk.

No “I changed it in the notebook” workflow for core experiments.

### 2.4 English-only code surface
The following must stay in English:
- code comments,
- folder names,
- experiment names,
- table names,
- metric names,
- figure names,
- commit titles,
- PR titles.

Chinese may appear in internal discussion documents, but not in the code surface.

---

## 3. Canonical repository tree

```text
probe-then-act/
  README.md
  docs/
    00_PROJECT_BRIEF.md
    01_REPO_BLUEPRINT.md
    02_EXECUTION_PLAYBOOK.md
    03_EXPERIMENT_PROTOCOL.md

  pta/
    __init__.py

    envs/
      __init__.py
      builders/
        scene_builder.py
        robot_builder.py
        material_builder.py
        tool_builder.py
        sensor_builder.py
        container_builder.py
      tasks/
        base_task.py
        scoop_transfer.py
        level_fill.py
      materials/
        material_family.py
        mpm_materials.py
        material_sampling.py
      tools/
        tool_library.py
        tool_randomization.py
      randomization/
        domain_randomizer.py
        observation_noise.py
        geometry_randomizer.py
      sensors/
        camera_obs.py
        tactile_obs.py
        proprio_obs.py
        observation_stack.py
      rewards/
        task_reward.py
        risk_penalty.py
        shaping_terms.py
      metrics/
        task_metrics.py
        spill_metrics.py
        contact_metrics.py
        calibration_metrics.py
      wrappers/
        gym_wrapper.py
        vector_env.py
      debug/
        overlays.py
        event_recorder.py
        state_dump.py

    models/
      __init__.py
      encoders/
        vision_encoder.py
        tactile_encoder.py
        proprio_encoder.py
        multimodal_fusion.py
      probe/
        probe_policy.py
        probe_action_space.py
      belief/
        latent_belief_encoder.py
        uncertainty_head.py
        auxiliary_heads.py
      policy/
        task_policy.py
        action_head.py
        risk_head.py
      teachers/
        privileged_teacher.py
      students/
        student_policy.py
        distillation_losses.py

    training/
      __init__.py
      rl/
        train_teacher.py
        train_task_policy.py
        rollout_storage.py
      il/
        collect_teacher_data.py
        train_student.py
      distill/
        online_distill.py
        offline_distill.py
      curriculum/
        curriculum_scheduler.py
      launch/
        launch_local.py
        launch_slurm.py
      utils/
        seed.py
        checkpoint_io.py
        logger.py

    eval/
      __init__.py
      runners/
        eval_policy.py
        eval_probe.py
        eval_ood.py
      splits/
        split_id.py
        split_ood_material.py
        split_ood_tool.py
        split_ood_container.py
        split_ood_sensor.py
      analysis/
        aggregate_results.py
        summarize_failures.py
        build_tables.py
        build_figures.py
      videos/
        record_rollouts.py
        montage.py

    configs/
      env/
        scoop_transfer/
          id.yaml
          ood_material.yaml
          ood_tool.yaml
          ood_container.yaml
          ood_sensor.yaml
        level_fill/
          id.yaml
          ood_material.yaml
          ood_tool.yaml
          ood_container.yaml
          ood_sensor.yaml
      model/
        reactive.yaml
        rnn.yaml
        probe_then_act.yaml
        no_uncertainty.yaml
        material_router.yaml
      train/
        teacher_rl.yaml
        student_bc.yaml
        finetune_rl.yaml
      eval/
        default.yaml
        paper_main.yaml
        paper_ablation.yaml
      ablation/
        no_probe.yaml
        random_probe.yaml
        no_tactile.yaml
        no_uncertainty.yaml
        no_teacher_student.yaml

    scripts/
      sanity_check_env.py
      visualize_scene.py
      run_probe_debug.py
      train_teacher.py
      train_student.py
      run_eval_main.py
      run_eval_ablation.py
      export_paper_videos.py
      export_tables.py

    utils/
      registry.py
      io.py
      paths.py
      typing.py

  assets/
    meshes/
    urdf/
    scene_assets/
    tool_meshes/

  results/
    manifests/
    tables/
    figures/
    videos/

  checkpoints/
    teacher/
    student/
    ablations/

  logs/
    tensorboard/
    system/
    run_metadata/

  tests/
    test_scene_build.py
    test_observation_shapes.py
    test_reward_signs.py
    test_metric_consistency.py
    test_eval_splits.py

  third_party/
    README.md
```

---

## 4. What each directory owns

### `pta/envs/`
Owns all **environment-side logic**:
- scene construction,
- task reset,
- reward,
- observations,
- metrics,
- domain randomization,
- debug utilities.

It does **not** own:
- network architectures,
- PPO implementation,
- distillation losses.

### `pta/models/`
Owns all **learning-side logic**:
- multimodal encoders,
- latent belief encoder,
- uncertainty head,
- probe policy,
- task policy,
- teacher/student wrappers.

It does **not** own:
- scene randomization,
- reward computation,
- metric aggregation.

### `pta/training/`
Owns all **optimization pipelines**:
- RL stage,
- IL / BC stage,
- distillation,
- launchers,
- checkpointing,
- logging.

### `pta/eval/`
Owns all **evaluation-only code**:
- test-time rollouts,
- OOD split definitions,
- aggregation,
- table generation,
- figure generation,
- video generation.

Evaluation code must be callable **without** importing training launch logic.

### `pta/configs/`
Owns all experiment settings.
A result that cannot be traced back to a config file should not appear in the paper.

---

## 5. Module boundaries that must not be violated

### 5.1 Reward vs metric
- **Reward** is for optimization.
- **Metric** is for reporting.

Never report “training reward” as a main paper result.

### 5.2 Task logic vs randomization
Task success definition must not live inside domain randomization code.

### 5.3 Observation vs privileged state
Student observation and privileged teacher state must be physically separated in code.
Do not let the student accidentally read hidden material parameters.

### 5.4 Mainline vs ablation
Ablation hacks must not pollute the default path.
Every ablation should be toggled via config.

---

## 6. Naming conventions

### Experiment names
Use this format:

```text
{task}.{method}.{split}.{seed}
```

Example:
```text
scoop_transfer.pta.ood_material.seed3
```

### Checkpoint names
```text
{task}_{method}_{stage}_{step}.pt
```

Example:
```text
scoop_transfer_pta_student_00800000.pt
```

### Table names
```text
main_results.csv
ablation_results.csv
ood_breakdown.csv
```

### Figure names
```text
fig_main_ood_breakdown.png
fig_ablation_probe_vs_no_probe.png
fig_failure_gallery.png
```

---

## 7. Config rules

### 7.1 Single source of truth
Every run must load:
- one env config,
- one model config,
- one train config,
- one eval config.

### 7.2 No hidden constants
Do not hard-code:
- task horizon,
- reward weights,
- material ranges,
- noise levels,
- number of probe steps,
- seed count.

These belong in config.

### 7.3 Stable IDs
Assign stable IDs to:
- material families,
- tool geometries,
- container families,
- sensor noise presets.

This matters for reproducible train/test splits.

---

## 8. Results and artifact policy

### 8.1 What gets committed
Commit:
- configs,
- scripts,
- small CSV summaries,
- tables,
- figure-generation code,
- markdown notes.

Do **not** commit:
- huge checkpoints,
- raw videos from every run,
- giant particle dumps,
- ad-hoc notebooks with unpublished logic.

### 8.2 What every finished run must write
Each run must write:
- `config_snapshot.yaml`
- `metrics.json`
- `stdout.log`
- `seed.txt`
- `git_commit.txt`
- optional TensorBoard event file

### 8.3 Result manifest
Create one lightweight run registry:
```text
results/manifests/runs.csv
```

Columns:
- `run_name`
- `task`
- `method`
- `split`
- `seed`
- `status`
- `best_checkpoint`
- `notes`

This file becomes the ground truth for paper aggregation.

---

## 9. Minimal code quality bar

Before any large training run:
1. `sanity_check_env.py` passes
2. observation shapes are correct
3. reward signs are checked
4. at least one scripted / oracle baseline runs end-to-end
5. one evaluation video looks physically plausible

If any of those fail, do not launch RL.

---

## 10. Testing checklist

### Must-have tests
- scene builds in headless mode
- reset produces finite observations
- rewards are finite and bounded
- metrics are monotonic where expected
- OOD split loader excludes training IDs
- student observation excludes privileged fields

### Nice-to-have tests
- deterministic reset with fixed seed
- checkpoint load / resume
- video export smoke test

---

## 11. Recommended branch policy

Use lightweight but disciplined branches:
- `main`: stable paper-quality code only
- `dev`: integrated but not fully frozen code
- `feature/env-*`
- `feature/model-*`
- `feature/eval-*`

Each merged branch should leave behind:
- code,
- a short markdown note,
- and at least one reproducible command line.

---

## 12. Pull request checklist

Every PR should answer:
1. What problem does this PR solve?
2. Which config or experiment depends on it?
3. How was it tested?
4. What new files were added?
5. Does this change affect previous results?

No “just some cleanup” PRs for core logic during active experiment weeks.

---

## 13. Definition of done by subsystem

### Environment subsystem done
- two tasks build reliably
- resets are diverse but stable
- metrics are trustworthy
- videos show meaningful interactions

### Method subsystem done
- reactive baseline trains
- teacher trains
- student trains
- Probe-Then-Act mainline trains
- ablation toggles work from config

### Evaluation subsystem done
- ID + OOD rollouts run in batch
- tables export automatically
- videos can be reproduced
- failure taxonomy script works

---

## 14. Anti-patterns to avoid
- starting with end-to-end PPO before environment instrumentation
- mixing debug code into training loops
- using notebook-only analysis for paper tables
- changing train/test splits midway without versioning
- reporting cherry-picked seeds
- reporting only success rate without spill/contact metrics
- introducing a third task before the first two are stable

---

## 15. Recommended `README.md` at repo root
The root `README.md` should be short:
- what this repo is,
- the paper hypothesis,
- quick setup,
- reading order for docs,
- main commands,
- current status.

Do **not** turn the root README into a long lab notebook.

---

## 16. Final instruction to the team
Treat this repo as a **scientific instrument**.
If a result cannot be reconstructed from the repo, it does not count.
