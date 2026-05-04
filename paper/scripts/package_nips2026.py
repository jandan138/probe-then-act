#!/usr/bin/env python3
"""Build and verify the NIPS/NeurIPS 2026 submission bundle."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_LABELS = {
    "fig:pta_workflow": ("1", "2"),
    "fig:scene_schematic": ("2", "5"),
    "fig:main": ("3", "7"),
    "fig:effect_delta": ("4", "7"),
    "fig:teaser": ("5", "9"),
    "sec:conclusion": ("6", "9"),
    "sec:conclusion:end": ("6", "9"),
}
MIN_EXPECTED_TOTAL_PAGES = 14

LOG_PATTERNS = {
    "LaTeX errors": r"(?m)^! (?:LaTeX Error|Package .* Error|Emergency stop|Fatal error)",
    "Undefined references": r"(?i)(?:LaTeX Warning: Reference `[^']+' on page \d+ undefined|There were undefined references)",
    "Undefined citations": r"(?i)(?:Citation `[^']+' undefined|There were undefined citations|Package natbib Warning: Citation)",
}

BLG_PATTERN = r"(?i)(?:Warning--|I couldn't open database|Repeated entry|empty year|empty journal|empty booktitle|error)"

STALE_PATTERNS = {
    "overclaim: broadly robust": r"broadly robust",
    "overclaim: competitive method": r"competitive method",
    "overclaim: validated law": r"validated law",
    "stale seed wording": r"(?:three-seed diagnostic|3-seed diagnostic)",
    "stale snow rescue wording": r"post-hoc rescue",
    "stale ablation wording": r"both ablations fall below",
}


@dataclass
class Check:
    name: str
    status: str
    required: bool
    details: str


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def redact(text: str, redactions: list[tuple[str, str]]) -> str:
    for raw, replacement in redactions:
        if raw:
            text = text.replace(raw, replacement)
    return text


def run_command(
    command: list[str],
    cwd: Path,
    command_log: Path,
    redactions: list[tuple[str, str]],
    required: bool = True,
) -> tuple[int, str]:
    display = shell_join(command)
    safe_display = redact(display, redactions)
    safe_cwd = redact(str(cwd), redactions)
    command_log.parent.mkdir(parents=True, exist_ok=True)
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {safe_display}\n[cwd] {safe_cwd}\n")

    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(redact(result.stdout, redactions))
        handle.write(f"\n[exit] {result.returncode}\n")

    if required and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit {result.returncode}: {safe_display}")

    return result.returncode, result.stdout


def add_check(checks: list[Check], name: str, passed: bool, details: str, required: bool = True) -> None:
    checks.append(Check(name=name, status="PASS" if passed else "FAIL", required=required, details=details))


def add_skip(checks: list[Check], name: str, details: str) -> None:
    checks.append(Check(name=name, status="SKIP", required=False, details=details))


def add_warn(checks: list[Check], name: str, details: str) -> None:
    checks.append(Check(name=name, status="WARN", required=False, details=details))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_info(path: Path, base: Path) -> dict[str, object]:
    return {
        "path": rel(path, base),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def scan_text_file(path: Path, patterns: dict[str, str], checks: list[Check], prefix: str, base: Path) -> None:
    display_path = rel(path, base)
    if not path.exists():
        add_check(checks, f"{prefix}: file exists", False, f"Missing {display_path}")
        return

    text = path.read_text(encoding="utf-8", errors="replace")
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        add_check(
            checks,
            f"{prefix}: {name}",
            not matches,
            "no matches" if not matches else f"{len(matches)} matches in {display_path}",
        )


def check_blg(path: Path, checks: list[Check], prefix: str, base: Path) -> None:
    display_path = rel(path, base)
    if not path.exists():
        add_check(checks, f"{prefix}: BibTeX log exists", False, f"Missing {display_path}")
        return

    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(BLG_PATTERN, text)
    add_check(
        checks,
        f"{prefix}: BibTeX warnings/errors",
        not matches,
        "no matches" if not matches else f"{len(matches)} matches in {display_path}",
    )


def parse_aux_label(aux_text: str, label: str) -> tuple[str, str] | None:
    match = re.search(r"\\newlabel\{" + re.escape(label) + r"\}\{\{([^}]*)\}\{([^}]*)\}", aux_text)
    if not match:
        return None
    return match.group(1), match.group(2)


def check_aux(aux_path: Path, checks: list[Check], base: Path) -> None:
    if not aux_path.exists():
        add_check(checks, "NeurIPS aux exists", False, f"Missing {rel(aux_path, base)}")
        return

    text = aux_path.read_text(encoding="utf-8", errors="replace")
    for label, expected in EXPECTED_LABELS.items():
        actual = parse_aux_label(text, label)
        add_check(
            checks,
            f"NeurIPS label {label}",
            actual == expected,
            f"expected number/page {expected}, found {actual}",
        )

    page_match = re.search(r"\\gdef \\@abspage@last\{([^}]*)\}", text)
    actual_pages = page_match.group(1) if page_match else None
    actual_page_count = int(actual_pages) if actual_pages and actual_pages.isdigit() else None
    add_check(
        checks,
        "NeurIPS total PDF pages from aux",
        actual_page_count is not None and actual_page_count >= MIN_EXPECTED_TOTAL_PAGES,
        f"expected at least {MIN_EXPECTED_TOTAL_PAGES} pages with refs/appendix/checklist, found {actual_pages}",
    )


def check_anonymity(main_tex: Path, checks: list[Check]) -> None:
    text = main_tex.read_text(encoding="utf-8", errors="replace")
    style_match = re.search(r"\\usepackage(?:\[([^]]*)\])?\{neurips_2026\}", text)
    options = style_match.group(1) if style_match else ""
    forbidden_options = {"final", "preprint", "nonanonymous"}
    active_forbidden = [option for option in forbidden_options if option in (options or "")]
    add_check(
        checks,
        "Anonymous NeurIPS style options",
        style_match is not None and not active_forbidden,
        f"style options={options!r}; forbidden={active_forbidden}",
    )
    add_check(
        checks,
        "Anonymous author placeholder",
        "Anonymous Author" in text,
        "main.tex contains Anonymous Author placeholder" if "Anonymous Author" in text else "Anonymous Author placeholder missing",
    )


def check_stale_wording(paper_dir: Path, checks: list[Check]) -> None:
    tex_files = list((paper_dir / "venues" / "neurips").rglob("*.tex")) + list((paper_dir / "shared").rglob("*.tex"))
    combined = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in tex_files)
    for name, pattern in STALE_PATTERNS.items():
        matches = re.findall(pattern, combined, flags=re.IGNORECASE)
        add_check(checks, f"Stale wording: {name}", not matches, "no matches" if not matches else f"{len(matches)} matches")


def check_neurips_checklist(venue_dir: Path, source_main_tex: Path, checks: list[Check]) -> None:
    checklist_tex = venue_dir / "sections" / "B_checklist.tex"
    if not checklist_tex.exists():
        add_check(checks, "NeurIPS paper checklist source exists", False, f"Missing {checklist_tex}")
        return

    checklist = checklist_tex.read_text(encoding="utf-8", errors="replace")
    source = source_main_tex.read_text(encoding="utf-8", errors="replace") if source_main_tex.exists() else ""
    todo_matches = re.findall(r"\\answerTODO|\\justificationTODO|\bTODO\b", checklist)
    answers = re.findall(r"(?m)^\s*\\item\[\] Answer:", checklist)
    questions = re.findall(r"(?m)^\\item \{\\bf ", checklist)
    add_check(
        checks,
        "NeurIPS paper checklist heading",
        "NeurIPS Paper Checklist" in checklist and "NeurIPS Paper Checklist" in source,
        "heading present in checklist source and flattened source"
        if "NeurIPS Paper Checklist" in checklist and "NeurIPS Paper Checklist" in source
        else "heading missing from checklist source or flattened source",
    )
    add_check(
        checks,
        "NeurIPS paper checklist answers complete",
        len(questions) == 16 and len(answers) == 16 and not todo_matches,
        f"questions={len(questions)}, answers={len(answers)}, todo_markers={len(todo_matches)}",
    )


def check_pdf_tools(pdf_path: Path, checks: list[Check], command_log: Path, redactions: list[tuple[str, str]]) -> dict[str, object]:
    results: dict[str, object] = {}

    if shutil.which("pdfinfo"):
        _, output = run_command(["pdfinfo", str(pdf_path)], cwd=pdf_path.parent, command_log=command_log, redactions=redactions)
        pages = re.search(r"(?m)^Pages:\s+(\d+)", output)
        results["pdfinfo_pages"] = pages.group(1) if pages else None
        add_check(
            checks,
            "pdfinfo page count",
            pages is not None and int(pages.group(1)) >= MIN_EXPECTED_TOTAL_PAGES,
            f"expected at least {MIN_EXPECTED_TOTAL_PAGES}, found {pages.group(1) if pages else None}",
        )
    else:
        add_skip(checks, "pdfinfo page count", "pdfinfo is not installed; aux page check was used instead")

    if shutil.which("pdffonts"):
        _, output = run_command(["pdffonts", str(pdf_path)], cwd=pdf_path.parent, command_log=command_log, redactions=redactions)
        lines = [line for line in output.splitlines()[2:] if line.strip()]
        bad = []
        for line in lines:
            cols = line.split()
            if len(cols) >= 5 and cols[-5] != "yes":
                bad.append(line)
        results["pdffonts_checked"] = True
        add_check(checks, "PDF fonts embedded", not bad, "all fonts embedded" if not bad else f"non-embedded font rows: {bad}")
    else:
        add_skip(checks, "PDF fonts embedded", "pdffonts is not installed; font embedding was not checked")

    return results


def create_source_zip(source_dir: Path, zip_path: Path) -> None:
    include_suffixes = {".tex", ".sty", ".pdf", ".png", ".jpg", ".jpeg"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            if "build" in path.relative_to(source_dir).parts:
                continue
            if path.suffix.lower() not in include_suffixes:
                continue
            archive.write(path, path.relative_to(source_dir))


def write_supplemental_checklist(path: Path) -> None:
    path.write_text(
        """# NIPS2026 Supplemental / Reproducibility Checklist

## Included Artifacts

- Main anonymous submission PDF: `pta_nips2026_main.pdf`
- Flattened LaTeX source archive: `pta_nips2026_source.zip`
- Source manifest and readiness evidence: `manifest.json`, `READINESS_REPORT.md`, `command.log`
- Appendix: included in the main PDF after the bibliography

## Supplement Status

- No separate supplementary experiment results are added in this bundle.
- No unreported numbers, citations, or claims are introduced by packaging.
- If OpenReview requires a separate supplement upload, use this checklist plus the source archive as the supplemental package manifest.

## Reproducibility Pointers

- Paper source of truth: `paper/venues/neurips/main.tex`
- Shared paper sections and figures: `paper/shared/`
- Experiment code: `pta/`, `scripts/`, and `tests/`
- Recorded results and generated paper figures: `results/` and `paper/shared/figures/`
- Rebuild command: `cd paper && make nips2026`

## Anonymity Notes

- The NeurIPS style is loaded without `final`, `preprint`, or `nonanonymous` options.
- The author block uses the anonymous placeholder in the source package.
""",
        encoding="utf-8",
    )


def write_readiness_report(path: Path, checks: list[Check], artifacts: dict[str, dict[str, object]], metadata: dict[str, object]) -> None:
    def escape(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# NIPS2026 Submission Readiness Report",
        "",
        f"Generated UTC: `{metadata['created_at_utc']}`",
        "Source state: VCS branch and commit omitted from anonymous bundle metadata.",
        "",
        "## Artifacts",
        "",
    ]
    for name, info in artifacts.items():
        lines.append(f"- {name}: `{info['path']}` ({info['bytes']} bytes, sha256 `{info['sha256']}`)")

    lines.extend([
        "",
        "## Generated Metadata",
        "",
        f"- readiness_report: `{metadata['readiness_report_path']}`",
        f"- manifest: `{metadata['manifest_path']}`",
        "",
        "Report and manifest hashes are not listed in this report to avoid self-referential hash metadata.",
        "The manifest records the finalized readiness report hash and omits a manifest self-hash for the same reason.",
        "",
        "## Checks",
        "",
        "| Status | Required | Check | Details |",
        "| --- | --- | --- | --- |",
    ])
    for check in checks:
        lines.append(f"| {check.status} | {check.required} | {escape(check.name)} | {escape(check.details)} |")

    lines.extend([
        "",
        "## Commands",
        "",
        "Full command output is recorded in `command.log`.",
        "",
        "## Notes",
        "",
        "Generated artifacts are intentionally under `paper/submissions/`, which is ignored by git.",
        "Skipped checks mean the external tool was unavailable in this environment; they are not reported as passed.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    paper_dir = Path(__file__).resolve().parents[1]
    repo_root = paper_dir.parent
    venue_dir = paper_dir / "venues" / "neurips"
    build_dir = venue_dir / "build"
    output_dir = paper_dir / "submissions" / "nips2026"
    source_dir = output_dir / "source"
    command_log = output_dir / "command.log"
    redactions = [
        (sys.executable, "python3"),
        (str(repo_root), "<repo>"),
        (str(Path.home()), "<home>"),
    ]
    checks: list[Check] = []

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    artifacts: dict[str, dict[str, object]] = {}
    metadata: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_state": "VCS branch and commit omitted from anonymous bundle metadata.",
        "venue": "neurips_2026",
    }
    readiness_report = output_dir / "READINESS_REPORT.md"
    manifest_path = output_dir / "manifest.json"
    metadata["readiness_report_path"] = rel(readiness_report, paper_dir)
    metadata["manifest_path"] = rel(manifest_path, paper_dir)

    try:
        main_pdf = build_dir / "main.pdf"
        add_check(checks, "Built NeurIPS PDF exists", main_pdf.exists(), rel(main_pdf, paper_dir))
        if not main_pdf.exists():
            raise RuntimeError("NeurIPS PDF is missing; run make neurips first")

        packaged_pdf = output_dir / "pta_nips2026_main.pdf"
        shutil.copy2(main_pdf, packaged_pdf)
        add_check(checks, "Packaged PDF size", packaged_pdf.stat().st_size > 100_000, f"{packaged_pdf.stat().st_size} bytes")

        run_command([sys.executable, "scripts/flatten.py", "--venue", "neurips", "--output", str(source_dir)], cwd=paper_dir, command_log=command_log, redactions=redactions)
        shutil.copy2(venue_dir / "neurips_2026.sty", source_dir / "neurips_2026.sty")
        add_check(checks, "Flattened source main.tex exists", (source_dir / "main.tex").exists(), rel(source_dir / "main.tex", paper_dir))
        add_check(checks, "NeurIPS style copied to source", (source_dir / "neurips_2026.sty").exists(), rel(source_dir / "neurips_2026.sty", paper_dir))

        run_command(["latexmk", "-pdf", "-outdir=build", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=source_dir, command_log=command_log, redactions=redactions)
        source_build_pdf = source_dir / "build" / "main.pdf"
        source_build_log = source_dir / "build" / "main.log"
        add_check(checks, "Flattened source PDF compiles", source_build_pdf.exists(), "latexmk produced build/main.pdf before cleanup")

        scan_text_file(build_dir / "main.log", LOG_PATTERNS, checks, "NeurIPS build log", paper_dir)
        check_blg(build_dir / "main.blg", checks, "NeurIPS build", paper_dir)
        scan_text_file(source_build_log, LOG_PATTERNS, checks, "Flattened source log", paper_dir)
        check_neurips_checklist(venue_dir, source_dir / "main.tex", checks)
        if (source_dir / "build").exists():
            shutil.rmtree(source_dir / "build")

        source_zip = output_dir / "pta_nips2026_source.zip"
        create_source_zip(source_dir, source_zip)
        add_check(checks, "Source zip size", source_zip.stat().st_size > 10_000, f"{source_zip.stat().st_size} bytes")

        check_aux(build_dir / "main.aux", checks, paper_dir)
        check_anonymity(venue_dir / "main.tex", checks)
        check_stale_wording(paper_dir, checks)
        pdf_tool_results = check_pdf_tools(packaged_pdf, checks, command_log, redactions)
        metadata.update(pdf_tool_results)

        supplemental = output_dir / "SUPPLEMENTAL_CHECKLIST.md"
        write_supplemental_checklist(supplemental)

        artifacts = {
            "main_pdf": artifact_info(packaged_pdf, paper_dir),
            "source_zip": artifact_info(source_zip, paper_dir),
            "flattened_main_tex": artifact_info(source_dir / "main.tex", paper_dir),
            "supplemental_checklist": artifact_info(supplemental, paper_dir),
            "command_log": artifact_info(command_log, paper_dir),
        }
    except Exception as exc:
        add_check(checks, "Package command completed", False, str(exc))
    else:
        add_check(checks, "Package command completed", True, "all packaging steps completed")

    write_readiness_report(readiness_report, checks, artifacts, metadata)

    manifest = {
        **metadata,
        "artifacts": artifacts,
        "generated_metadata": {
            "readiness_report": artifact_info(readiness_report, paper_dir),
            "manifest": {
                "path": rel(manifest_path, paper_dir),
                "sha256": "omitted: self-referential",
            },
        },
        "checks": [asdict(check) for check in checks],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed = [check for check in checks if check.required and check.status == "FAIL"]
    print(f"NIPS2026 submission bundle written to {rel(output_dir, paper_dir)}")
    print(f"Readiness report: {rel(readiness_report, paper_dir)}")
    if failed:
        print("Required checks failed:", file=sys.stderr)
        for check in failed:
            print(f"- {check.name}: {check.details}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
