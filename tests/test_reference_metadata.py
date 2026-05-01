"""Regression checks for paper bibliography metadata."""

from __future__ import annotations

from pathlib import Path


REFERENCES = Path(__file__).resolve().parents[1] / "paper" / "shared" / "references.bib"


def _parse_entries() -> dict[str, dict[str, str]]:
    text = REFERENCES.read_text(encoding="utf-8")
    entries: dict[str, dict[str, str]] = {}
    current_key: str | None = None
    current_fields: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("@"):
            current_key = line.split("{", 1)[1].split(",", 1)[0]
            current_fields = {}
            entries[current_key] = current_fields
            continue
        if current_key is None or "=" not in line:
            continue
        name, value = line.split("=", 1)
        current_fields[name.strip()] = value.strip().rstrip(",").strip("{}")

    return entries


def test_reported_reference_metadata_matches_verified_records() -> None:
    entries = _parse_entries()

    assert entries["bauza2020tactile"]["author"] == (
        "Bauza, Maria and Rodriguez, Alberto and Lim, Bryan and Valls, Eric and Sechopoulos, Theo"
    )
    assert entries["bauza2018data"]["author"] == "Bauza, Maria and Hogan, Francois R. and Rodriguez, Alberto"
    assert entries["bauza2018data"]["booktitle"] == "Proceedings of The 2nd Conference on Robot Learning"
    assert entries["freeman2021brax"]["booktitle"] == (
        "Advances in Neural Information Processing Systems ({NeurIPS}) Datasets and Benchmarks Track"
    )
    assert entries["millard2023granulargym"]["author"] == (
        "Millard, David R. and Pastor, Daniel and Bowkett, Joseph and Backes, Paul and Sukhatme, Gaurav S."
    )
    assert entries["yang2025differentiable"]["title"] == (
        "Differentiable Physics-based System Identification for Robotic Manipulation of Elastoplastic Materials"
    )
    assert entries["lin2021softgym"]["booktitle"] == (
        "Proceedings of the 2020 Conference on Robot Learning"
    )
