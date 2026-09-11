from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_profile_manifest_matches_bundled_artifacts() -> None:
    manifest = json.loads((ROOT / "docs/benchmarks/profile_manifest.json").read_text())
    assert len(manifest) == 4
    for item in manifest:
        profile = ROOT / "src/siftsc/profiles" / f"{item['profile']}.json"
        assert profile.is_file()
        assert _sha256(profile) == item["artifact_sha256"]


def test_benchmark_tables_have_expected_release_grid() -> None:
    with (ROOT / "docs/benchmarks/per_task.csv").open(newline="") as handle:
        per_task = list(csv.DictReader(handle))
    with (ROOT / "docs/benchmarks/pooled_gate.csv").open(newline="") as handle:
        pooled = list(csv.DictReader(handle))
    assert len(per_task) == 8
    assert len(pooled) == 4
    assert {row["dataset"] for row in per_task} == {"GSM8K", "MATH-500"}
    assert all(int(row["n"]) == 200 for row in per_task)
    assert all(int(row["n"]) == 400 for row in pooled)


def test_readme_relative_links_resolve() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    markdown_targets = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", readme)
    html_targets = re.findall(r'(?:href|src)="([^"]+)"', readme)
    targets = markdown_targets + html_targets
    local_targets = [
        target for target in targets if "://" not in target and not target.startswith("#")
    ]
    missing = [target for target in local_targets if not (ROOT / target).is_file()]
    assert missing == []


def test_readme_discloses_bundled_demo_platform() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "currently require an Apple-Silicon Mac" in readme
    assert "built-in Linux or Windows model runner is not included yet" in readme
