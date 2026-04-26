# Resumable OOD Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pta/scripts/run_ood_eval_v2.py` survive cron restarts and OOM kills by persisting each completed `(method, seed, split)` row immediately and skipping completed rows on restart.

**Architecture:** Keep the existing evaluator and orchestrator shape. Add small persistence helpers to `run_ood_eval_v2.py`, make resume the default behavior, and write aggregate outputs after every persisted row so `main_results.csv` remains useful even when the full sweep has not finished. The coordinator can keep relaunching the same command; progress lives in `results/ood_eval_per_seed.csv`.

**Tech Stack:** Python 3.11, `csv`, `pathlib`, `numpy`, existing pytest suite.

---

## Current Evidence

**Execution status (2026-04-26):** this plan has been implemented. The corrected resumable OOD sweep completed with `35/35` per-seed rows and `15` aggregate rows. Result-to-claim found the original broad PTA claims unsupported, so the next research step is the ablation-first diagnostic plan in `refine-logs/EXPERIMENT_PLAN.md`.

Planning-time state: training was complete for `M1` seeds `42/0/1`, `M7` seeds `42/0/1`, and `M8` seed `42`, while corrected OOD evaluation was not complete because the evaluator only wrote CSVs after the full sweep. This historical blocker has since been resolved by the implementation described here.

Kernel logs show process-level OOM kills, not just episode-level Genesis NaNs:

```text
2026-04-25 21:43 HKT: killed Python PID 1159227, anon RSS 12133604kB
2026-04-26 03:45 HKT: killed Python PID 1241153, anon RSS 12328420kB
2026-04-26 09:20 HKT: killed Python PID 1339114, anon RSS 12361992kB
```

The latest attempt reached `m7_pta seed=0 ood_snow`, but no CSV was written because the current script writes only after the full sweep finishes.

---

## Files

- Modify: `pta/scripts/run_ood_eval_v2.py`
- Modify: `tests/test_run_ood_eval_v2.py`
- Verify: `tests/test_cron_aris_orchestrator.py`
- Verify: `tests/test_cron_shell_contract.py`
- Docs already updated: `docs/30_records/CRON_ARIS_ORCHESTRATOR_RUNBOOK.md`
- Docs already updated: `docs/10_protocols/04_VALIDATION_GATES.md`
- Docs already updated: `docs/20_planning/11_HOTFIX_TASK_BRIEF.md`

---

### Task 1: Add Result Persistence Helpers

**Files:**
- Modify: `pta/scripts/run_ood_eval_v2.py`
- Test: `tests/test_run_ood_eval_v2.py`

- [ ] **Step 1: Write failing tests for row keys and append persistence**

Add these tests to `tests/test_run_ood_eval_v2.py`:

```python
def test_result_key_uses_method_seed_split():
    from pta.scripts.run_ood_eval_v2 import result_key

    row = {"method": "m1_reactive", "seed": 42, "split": "ood_snow"}

    assert result_key(row) == ("m1_reactive", 42, "ood_snow")


def test_append_result_row_writes_header_once(tmp_path):
    from pta.scripts.run_ood_eval_v2 import RESULT_FIELDNAMES, append_result_row

    path = tmp_path / "ood_eval_per_seed.csv"
    row = {field: 0 for field in RESULT_FIELDNAMES}
    row.update(
        {
            "method": "m1_reactive",
            "seed": 42,
            "split": "id_sand",
            "mean_reward": 1.0,
            "std_reward": 0.0,
            "mean_transfer": 0.2,
            "std_transfer": 0.0,
            "mean_spill": 0.1,
            "std_spill": 0.0,
            "success_rate": 1.0,
            "n_failed_episodes": 0,
        }
    )

    append_result_row(path, row)
    append_result_row(path, {**row, "seed": 0})

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",") == RESULT_FIELDNAMES
    assert len(lines) == 3
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
source "/home/zhuzihou/dev/Genesis/.venv/bin/activate" && pytest tests/test_run_ood_eval_v2.py::test_result_key_uses_method_seed_split tests/test_run_ood_eval_v2.py::test_append_result_row_writes_header_once -v
```

Expected: import errors for `result_key`, `RESULT_FIELDNAMES`, and `append_result_row`.

- [ ] **Step 3: Implement minimal helpers**

Add this near `AGGREGATE_METRICS` in `pta/scripts/run_ood_eval_v2.py`:

```python
RESULT_FIELDNAMES = [
    "method",
    "seed",
    "split",
    "mean_reward",
    "std_reward",
    "mean_transfer",
    "std_transfer",
    "mean_spill",
    "std_spill",
    "success_rate",
    "n_failed_episodes",
]


def result_key(row):
    return (row["method"], int(row["seed"]), row["split"])


def append_result_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in RESULT_FIELDNAMES})
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
source "/home/zhuzihou/dev/Genesis/.venv/bin/activate" && pytest tests/test_run_ood_eval_v2.py::test_result_key_uses_method_seed_split tests/test_run_ood_eval_v2.py::test_append_result_row_writes_header_once -v
```

Expected: both tests pass.

---

### Task 2: Load Completed Rows Safely

**Files:**
- Modify: `pta/scripts/run_ood_eval_v2.py`
- Test: `tests/test_run_ood_eval_v2.py`

- [ ] **Step 1: Write failing tests for resume loading and stale CSV rejection**

Add these tests:

```python
def test_load_completed_rows_reads_existing_csv(tmp_path):
    from pta.scripts.run_ood_eval_v2 import append_result_row, load_completed_rows

    path = tmp_path / "ood_eval_per_seed.csv"
    row = {
        "method": "m1_reactive",
        "seed": 42,
        "split": "id_sand",
        "mean_reward": 1.0,
        "std_reward": 0.0,
        "mean_transfer": 0.2,
        "std_transfer": 0.0,
        "mean_spill": 0.1,
        "std_spill": 0.0,
        "success_rate": 1.0,
        "n_failed_episodes": 0,
    }
    append_result_row(path, row)

    rows, keys = load_completed_rows(path, resume=True)

    assert rows == [row]
    assert keys == {("m1_reactive", 42, "id_sand")}


def test_load_completed_rows_ignores_file_when_resume_disabled(tmp_path):
    from pta.scripts.run_ood_eval_v2 import append_result_row, load_completed_rows

    path = tmp_path / "ood_eval_per_seed.csv"
    row = {
        "method": "m1_reactive",
        "seed": 42,
        "split": "id_sand",
        "mean_reward": 1.0,
        "std_reward": 0.0,
        "mean_transfer": 0.2,
        "std_transfer": 0.0,
        "mean_spill": 0.1,
        "std_spill": 0.0,
        "success_rate": 1.0,
        "n_failed_episodes": 0,
    }
    append_result_row(path, row)

    rows, keys = load_completed_rows(path, resume=False)

    assert rows == []
    assert keys == set()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
source "/home/zhuzihou/dev/Genesis/.venv/bin/activate" && pytest tests/test_run_ood_eval_v2.py::test_load_completed_rows_reads_existing_csv tests/test_run_ood_eval_v2.py::test_load_completed_rows_ignores_file_when_resume_disabled -v
```

Expected: import error for `load_completed_rows`.

- [ ] **Step 3: Implement CSV loading with type conversion**

Add this helper:

```python
def load_completed_rows(path: Path, resume: bool = True):
    if not resume or not path.exists():
        return [], set()

    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {
                "method": raw["method"],
                "seed": int(raw["seed"]),
                "split": raw["split"],
                "mean_reward": float(raw["mean_reward"]),
                "std_reward": float(raw["std_reward"]),
                "mean_transfer": float(raw["mean_transfer"]),
                "std_transfer": float(raw["std_transfer"]),
                "mean_spill": float(raw["mean_spill"]),
                "std_spill": float(raw["std_spill"]),
                "success_rate": float(raw["success_rate"]),
                "n_failed_episodes": int(float(raw["n_failed_episodes"])),
            }
            rows.append(row)
    return rows, {result_key(row) for row in rows}
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
source "/home/zhuzihou/dev/Genesis/.venv/bin/activate" && pytest tests/test_run_ood_eval_v2.py::test_load_completed_rows_reads_existing_csv tests/test_run_ood_eval_v2.py::test_load_completed_rows_ignores_file_when_resume_disabled -v
```

Expected: both tests pass.

---

### Task 3: Write Aggregate Results After Each Row

**Files:**
- Modify: `pta/scripts/run_ood_eval_v2.py`
- Test: `tests/test_run_ood_eval_v2.py`

- [ ] **Step 1: Write failing test for aggregate CSV writing**

Add this test:

```python
def test_write_aggregate_results_creates_main_results_csv(tmp_path):
    from pta.scripts.run_ood_eval_v2 import write_aggregate_results

    rows = [
        {
            "method": "m1_reactive",
            "seed": 42,
            "split": "id_sand",
            "mean_reward": 2.0,
            "std_reward": 0.0,
            "mean_transfer": 0.4,
            "std_transfer": 0.0,
            "mean_spill": 0.1,
            "std_spill": 0.0,
            "success_rate": 1.0,
            "n_failed_episodes": 0,
        }
    ]
    path = tmp_path / "main_results.csv"

    agg_rows = write_aggregate_results(path, rows)

    assert path.exists()
    assert agg_rows[0]["method"] == "m1_reactive"
    assert agg_rows[0]["n_failed_episodes_sum"] == 0
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
source "/home/zhuzihou/dev/Genesis/.venv/bin/activate" && pytest tests/test_run_ood_eval_v2.py::test_write_aggregate_results_creates_main_results_csv -v
```

Expected: import error for `write_aggregate_results`.

- [ ] **Step 3: Implement aggregate writer**

Add this helper:

```python
def write_aggregate_results(path: Path, all_rows: list[dict]):
    agg_rows = aggregate_results(all_rows)
    if not agg_rows:
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)
    return agg_rows
```

- [ ] **Step 4: Run test and verify it passes**

Run:

```bash
source "/home/zhuzihou/dev/Genesis/.venv/bin/activate" && pytest tests/test_run_ood_eval_v2.py::test_write_aggregate_results_creates_main_results_csv -v
```

Expected: test passes.

---

### Task 4: Resume Main Loop Without Changing Cron Command

**Files:**
- Modify: `pta/scripts/run_ood_eval_v2.py`
- Test: `tests/test_run_ood_eval_v2.py`

- [ ] **Step 1: Add parser test for default resume behavior**

Add this test:

```python
def test_parse_args_resumes_by_default(monkeypatch):
    from pta.scripts.run_ood_eval_v2 import parse_args

    monkeypatch.setattr(sys, "argv", ["run_ood_eval_v2.py"])
    args = parse_args()

    assert args.resume is True
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
source "/home/zhuzihou/dev/Genesis/.venv/bin/activate" && pytest tests/test_run_ood_eval_v2.py::test_parse_args_resumes_by_default -v
```

Expected: `AttributeError` because `resume` is missing.

- [ ] **Step 3: Add `--no-resume` argument**

In `parse_args()`, add:

```python
parser.add_argument(
    "--no-resume",
    dest="resume",
    action="store_false",
    help="Ignore existing OOD CSV progress and start from scratch.",
)
parser.set_defaults(resume=True)
```

- [ ] **Step 4: Wire persistence into `main()`**

Replace the end-only per-seed write pattern with immediate persistence:

```python
per_seed_path = results_dir / "ood_eval_per_seed.csv"
main_results_path = results_dir / "main_results.csv"
all_rows, completed_keys = load_completed_rows(per_seed_path, resume=args.resume)
```

Inside the `(method_name, seed, split_name)` loop, before making the env:

```python
key = (method_name, seed, split_name)
if key in completed_keys:
    print(f"    {split_name}... SKIP existing", flush=True)
    continue
```

After `all_rows.append(row)`, add:

```python
append_result_row(per_seed_path, row)
completed_keys.add(key)
write_aggregate_results(main_results_path, all_rows)
```

Delete the old final-only per-seed CSV write block. Keep final summary printing, using:

```python
agg_rows = write_aggregate_results(main_results_path, all_rows)
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
source "/home/zhuzihou/dev/Genesis/.venv/bin/activate" && pytest tests/test_run_ood_eval_v2.py -v
```

Expected: all `test_run_ood_eval_v2.py` tests pass.

---

### Task 5: Verify Orchestrator Compatibility

**Files:**
- Verify: `pta/scripts/cron_aris_orchestrator.py`
- Verify: `pta/scripts/run_cron_aris_orchestrator.sh`
- Test: `tests/test_cron_aris_orchestrator.py`
- Test: `tests/test_cron_shell_contract.py`

- [ ] **Step 1: Confirm no cron command change is required**

Run:

```bash
rg -n "run_ood_eval_v2.py" pta/scripts/cron_aris_orchestrator.py tests/test_cron_aris_orchestrator.py
```

Expected: command remains `python pta/scripts/run_ood_eval_v2.py --residual-scale 0.05`. Resume is default, so no orchestrator command update is required.

- [ ] **Step 2: Run relevant regression tests**

Run:

```bash
source "/home/zhuzihou/dev/Genesis/.venv/bin/activate" && pytest tests/test_run_ood_eval_v2.py tests/test_cron_aris_orchestrator.py tests/test_cron_shell_contract.py -q
```

Expected: all tests pass.

---

### Task 6: Restart OOD Safely

**Files:**
- Runtime only, no source edits.

- [ ] **Step 1: Check for live evaluator**

Run:

```bash
ps -eo pid,etimes,%cpu,%mem,cmd | rg "run_ood_eval_v2.py|PID"
```

Expected: either no evaluator process or a single evaluator process.

- [ ] **Step 2: If an old evaluator is live, stop it**

Run only if Step 1 shows a live evaluator that started before this patch:

```bash
kill -TERM <PID>
sleep 5
ps -eo pid,etimes,%cpu,%mem,cmd | rg "run_ood_eval_v2.py|PID"
```

Expected: no old evaluator process remains.

- [ ] **Step 3: Launch via coordinator**

Run:

```bash
bash pta/scripts/run_cron_aris_orchestrator.sh
sleep 3
ps -eo pid,etimes,%cpu,%mem,cmd | rg "run_ood_eval_v2.py|PID"
```

Expected: one evaluator process starts, or coordinator logs `wait` if one was already running.

- [ ] **Step 4: Verify first row persists**

After the first split finishes, run:

```bash
ls -l results/ood_eval_per_seed.csv results/main_results.csv
```

Expected: both files exist before the full sweep finishes.

---

## Self-Review

Spec coverage: the plan addresses process-level OOM by adding immediate per-row persistence, restart-time skip logic, and aggregate refreshes after each row.

Placeholder scan: no TBD placeholders remain; every implementation task includes exact file paths, commands, and expected results.

Type consistency: row keys use `(method, seed, split)` consistently, and `n_failed_episodes` remains an integer in loaded rows.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-26-resumable-ood-eval.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
