#!/usr/bin/env python3
"""Flatten a dual-venue LaTeX paper for arXiv/Overleaf submission.

Usage:
    python scripts/flatten.py --venue neurips --output arxiv-submission/

Prerequisites:
    1. Build the venue first: make neurips
    2. This generates venues/<venue>/build/main.bbl via latexmk + bibtex
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Optional


def resolve_input_path(input_path: str, base_dir: Path, venues_dir: Path) -> Optional[Path]:
    """Resolve a \input{path} to an actual file."""
    # Compute paper root from venues_dir (venues/<venue>/ -> paper/)
    paper_dir = venues_dir.parent.parent
    # Try relative to base_dir first
    p = (base_dir / input_path).resolve()
    if p.exists():
        return p
    # Try relative to venues/<venue>/
    p = (venues_dir / input_path).resolve()
    if p.exists():
        return p
    # Try with .tex extension
    for base in [base_dir, venues_dir]:
        p = (base / (input_path + ".tex")).resolve()
        if p.exists():
            return p
    # Try paper root (for paths like figures/... from shared/sections/)
    p = (paper_dir / input_path).resolve()
    if p.exists():
        return p
    p = (paper_dir / (input_path + ".tex")).resolve()
    if p.exists():
        return p
    # Try shared/figures/ for TABLE_*.tex files (input may already have figures/ prefix)
    shared_figures = paper_dir / "shared" / "figures"
    # Strip leading figures/ if present to avoid double path
    clean_path = input_path
    if clean_path.startswith("figures/"):
        clean_path = clean_path[8:]
    p = (shared_figures / clean_path).resolve()
    if p.exists():
        return p
    p = (shared_figures / (clean_path + ".tex")).resolve()
    if p.exists():
        return p
    return None


def resolve_figure_path(fig_path: str, base_dir: Path, venues_dir: Path) -> Optional[Path]:
    """Resolve \includegraphics{path} using graphicspath."""
    paper_dir = venues_dir.parent.parent
    # Common figure extensions
    extensions = ["", ".pdf", ".png", ".jpg", ".eps"]
    # Try relative to base_dir
    for ext in extensions:
        p = (base_dir / (fig_path + ext)).resolve()
        if p.exists():
            return p
    # Try relative to venues/
    for ext in extensions:
        p = (venues_dir / (fig_path + ext)).resolve()
        if p.exists():
            return p
    # Try paper root (for paths like figures/... from shared/sections/)
    for ext in extensions:
        p = (paper_dir / (fig_path + ext)).resolve()
        if p.exists():
            return p
    # Try shared/figures/ (fig_path may already have figures/ prefix)
    shared_figures = paper_dir / "shared" / "figures"
    clean_fig = fig_path
    if clean_fig.startswith("figures/"):
        clean_fig = clean_fig[8:]
    for ext in extensions:
        p = (shared_figures / (clean_fig + ext)).resolve()
        if p.exists():
            return p
    return None


def inline_file(content: str, base_dir: Path, venues_dir: Path, processed: set) -> str:
    """Recursively inline \input{} commands."""
    # Pattern: \input{path} (no extension or with .tex)
    pattern = r"\\input\{([^}]+)\}"

    def replace_input(match):
        input_path = match.group(1)
        resolved = resolve_input_path(input_path, base_dir, venues_dir)
        if resolved is None:
            print(f"Warning: could not resolve \\input{{{input_path}}}", file=sys.stderr)
            return match.group(0)  # Keep original

        resolved = resolved.resolve()
        if resolved in processed:
            return f"% Already inlined: {input_path}\n"
        processed.add(resolved)

        inner_content = resolved.read_text(encoding="utf-8")
        # Recursively process nested inputs
        inner_content = inline_file(inner_content, resolved.parent, venues_dir, processed)
        return f"% --- BEGIN {input_path} ---\n{inner_content}\n% --- END {input_path} ---\n"

    return re.sub(pattern, replace_input, content)


def process_bibliography(content: str, venue_dir: Path) -> str:
    """Replace \bibliography{} with inlined .bbl content."""
    build_dir = venue_dir / "build"
    bbl_file = build_dir / "main.bbl"

    if not bbl_file.exists():
        print(f"Error: {bbl_file} not found. Build the venue first with 'make <venue>'.", file=sys.stderr)
        sys.exit(1)

    bbl_content = bbl_file.read_text(encoding="utf-8")
    # Replace \bibliography{...} and \bibliographystyle{...} with .bbl content
    # Use str.replace to avoid regex escape issues with bbl content
    content = re.sub(r"\\bibliographystyle\{[^}]+\}\n?", "", content)
    # Find and replace bibliography command with bbl content
    bib_match = re.search(r"\\bibliography\{[^}]+\}\n?", content)
    if bib_match:
        content = content[:bib_match.start()] + "\n" + bbl_content + "\n" + content[bib_match.end():]
    return content


def copy_figures(content: str, output_dir: Path, venues_dir: Path) -> str:
    """Copy figures to output dir and rewrite paths."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    pattern = r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"

    def replace_fig(match):
        fig_path = match.group(1)
        resolved = resolve_figure_path(fig_path, venues_dir, venues_dir)
        if resolved is None:
            print(f"Warning: could not resolve figure {fig_path}", file=sys.stderr)
            return match.group(0)

        # Copy to output figures dir
        dest = fig_dir / resolved.name
        shutil.copy2(resolved, dest)
        return match.group(0).replace(fig_path, f"figures/{resolved.name}")

    return re.sub(pattern, replace_fig, content)


def strip_comments(content: str) -> str:
    """Remove LaTeX comments (but not escaped %)."""
    lines = []
    for line in content.split("\n"):
        # Remove comments, but keep \%
        result = []
        i = 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            result.append(line[i])
            i += 1
        cleaned = "".join(result).rstrip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Flatten LaTeX paper for arXiv")
    parser.add_argument("--venue", required=True, choices=["ieee-trl", "neurips"])
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    paper_dir = Path(__file__).parent.parent.resolve()
    venue_dir = paper_dir / "venues" / args.venue
    main_tex = venue_dir / "main.tex"
    output_dir = Path(args.output).resolve()

    if not main_tex.exists():
        print(f"Error: {main_tex} not found", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read main.tex
    content = main_tex.read_text(encoding="utf-8")

    # Inline all \input{} commands recursively
    content = inline_file(content, venue_dir, venue_dir, set())

    # Process bibliography (inline .bbl)
    content = process_bibliography(content, venue_dir)

    # Copy figures and rewrite paths
    content = copy_figures(content, output_dir, venue_dir)

    # Strip comments
    content = strip_comments(content)

    # Write flattened main.tex
    output_tex = output_dir / "main.tex"
    output_tex.write_text(content, encoding="utf-8")

    print(f"Flattened paper written to: {output_dir}/")
    print(f"  - {output_tex}")
    print(f"  - {output_dir}/figures/")


if __name__ == "__main__":
    main()
