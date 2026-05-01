# NeurIPS Figure Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder NeurIPS main-text figures so Fig. 1 is the workflow/architecture figure and the uploaded result-summary image appears later as Fig. 5.

**Architecture:** Keep all figure assets unchanged. Move only LaTeX figure blocks and nearby references/captions: workflow moves from shared Method into NeurIPS Introduction, teaser/result-summary moves from Introduction into late Results.

**Tech Stack:** LaTeX, Makefile-driven `latexmk`, existing paper figure assets.

---

### Task 1: Establish Red Figure-Order Check

**Files:**
- Read: `paper/venues/neurips/build/main.aux`

- [ ] **Step 1: Run the figure-order assertion before editing**

Run:
```bash
python3 -c "from pathlib import Path; aux=Path('paper/venues/neurips/build/main.aux').read_text(); expected={'fig:pta_workflow':'1','fig:scene_schematic':'2','fig:main':'3','fig:effect_delta':'4','fig:teaser':'5'}; labels={}; [labels.setdefault(k, line.split('{{',1)[1].split('}',1)[0]) for line in aux.splitlines() for k in expected if line.startswith(chr(92)+'newlabel{'+k+'}')]; missing=[k for k in expected if k not in labels]; wrong=[(k, labels[k], v) for k,v in expected.items() if k in labels and labels[k]!=v]; assert not missing and not wrong, f'missing={missing} wrong={wrong}'"
```

Expected: FAIL because the current build has `fig:teaser` as Fig. 1 and `fig:pta_workflow` as Fig. 2.

### Task 2: Move Workflow To Introduction

**Files:**
- Modify: `paper/venues/neurips/sections/1_introduction.tex`
- Modify: `paper/shared/sections/3_method.tex`

- [ ] **Step 1: Remove the opening teaser figure from the Introduction**

Delete the `fig0_teaser.png` figure block from `paper/venues/neurips/sections/1_introduction.tex`.

- [ ] **Step 2: Insert the workflow figure in the Introduction**

Insert this figure block after the second motivation paragraph:
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.74\textwidth]{figures/fig1_workflow.pdf}
    \caption{\textbf{\PTA{} workflow.} A fixed probe yields a response trace, an encoder maps it to $(z,\sigma)$, and a residual PPO policy conditions on $(s_t,z,\sigma)$.}
    \label{fig:pta_workflow}
\end{figure}
```

- [ ] **Step 3: Remove the NeurIPS-only workflow block from Method**

Delete the `\ifdefined\isneurips ... \fi` block containing `fig1_workflow.pdf` in `paper/shared/sections/3_method.tex`.

### Task 3: Move Uploaded Summary Image To Late Results

**Files:**
- Modify: `paper/venues/neurips/sections/1_introduction.tex`
- Modify: `paper/venues/neurips/sections/5_results.tex`

- [ ] **Step 1: Update Introduction text references**

Change the Introduction text so it references `\cref{fig:pta_workflow}` for the method overview and does not reference `\cref{fig:teaser}`.

- [ ] **Step 2: Add the uploaded summary figure after the recoverability interpretation paragraph in Results**

Insert this block at the end of `paper/venues/neurips/sections/5_results.tex`:
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.92\textwidth]{figures/fig0_teaser.png}
    \caption{\textbf{Qualitative summary of the information--intervention tradeoff.} The 2D schematic summarizes the descriptive pattern after the quantitative results: the fixed probe can reveal material response but also perturbs the particle state. In this benchmark, \PTA{} improves elastoplastic transfer (+14.7pp versus M1) and underperforms on the other splits; the M8 value is a single-seed elastoplastic diagnostic. Exact metrics appear in \cref{tab:main_results,fig:main,fig:effect_delta}.}
    \label{fig:teaser}
\end{figure}
```

### Task 4: Rebuild And Verify Green

**Files:**
- Read: `paper/venues/neurips/build/main.aux`
- Read: `paper/venues/neurips/build/main.log`
- Read: `paper/venues/ieee-trl/build/main.log`

- [ ] **Step 1: Rebuild both venues**

Run:
```bash
make all
```

Expected: both venue PDFs build successfully.

- [ ] **Step 2: Rerun the figure-order assertion**

Run the Python assertion from Task 1.

Expected: PASS.

- [ ] **Step 3: Scan logs and source for stale wording**

Run:
```bash
rg --no-ignore -n "Warning|undefined|Overfull|Citation|multiply defined|LaTeX Error" "paper/venues/neurips/build/main.log" "paper/venues/ieee-trl/build/main.log" || true
```

Expected: no NeurIPS issues; IEEE may still report the known caption package warning only.

Run:
```bash
rg -n "M8 collapsing|Means over 3 training seeds|all three seeds|across three seeds|std \\\$0\\.0\\\\$|averaged over three random seeds\\. We|dagger|magenta callout" paper --glob '*.tex' || true
```

Expected: no matches.

- [ ] **Step 4: Check diff whitespace**

Run:
```bash
git diff --check
```

Expected: no output.

### Task 5: Review And Handoff

**Files:**
- Read: final git diff and status.

- [ ] **Step 1: Dispatch final review**

Ask a reviewer to check that Fig. 1 is workflow, the uploaded summary is late Results, M8 remains single-seed diagnostic, and NeurIPS conclusion remains on page 9.

- [ ] **Step 2: Report commit boundary**

Report verification evidence and changed/untracked files. Do not create a commit unless explicitly requested.
