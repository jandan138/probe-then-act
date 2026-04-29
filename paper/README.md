# Probe-Then-Act Paper

Dual-venue LaTeX paper structure supporting IEEE T-RL and NeurIPS submission.

## Directory Structure

```
paper/
├── shared/               # Cross-venue shared content
│   ├── sections/         # Default sections
│   ├── figures/          # Shared figures + generation scripts
│   ├── math_commands.tex # Shared math macros
│   ├── references.bib    # Shared bibliography
│   └── venue_macros.tex  # Venue-conditional macros
├── venues/
│   ├── ieee-trl/         # IEEE T-RL entry point
│   └── neurips/          # NeurIPS entry point
└── scripts/
    └── flatten.py        # arXiv flattening
```

## Build

```bash
# Build IEEE T-RL version
make ieee-trl

# Build NeurIPS version
make neurips

# Build both
make all

# Clean build artifacts
make clean

# Check page counts
make check
```

## How Overrides Work

Each venue's `main.tex` explicitly declares which files to include:
- `../../shared/sections/X.tex` = shared content
- `sections/X.tex` = venue-specific override

If a venue doesn't need to override a section, it references the shared version.

## Adding a Venue-Specific Section Override

1. Copy from shared: `cp shared/sections/X.tex venues/<venue>/sections/`
2. Modify `venues/<venue>/sections/X.tex`
3. Ensure `venues/<venue>/main.tex` references `sections/X.tex` (not `../../shared/sections/X.tex`)

## arXiv Submission

```bash
make neurips  # Generates .bbl
make arxiv    # Flattens to arxiv-submission/
```

The flattened directory contains a single `main.tex` with all inputs inlined and figures copied.
