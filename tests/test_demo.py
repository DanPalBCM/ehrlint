"""The demo and the committed example data.

The demo is the promise on the README: `uv sync && uv run python demo.py` works
and shows the tool working. It is tested here so it cannot rot silently, and
because it is the only place the committed `examples/` data is exercised
end to end through the real MEDS reader.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from ehrlint.audit import run_audit
from ehrlint.context import AuditContext

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
CLEAN = EXAMPLES / "clean_meds"
LEAKY = EXAMPLES / "leaky_meds"


def load(root: Path) -> AuditContext:
    features = root / "features.parquet"
    return AuditContext.from_meds(
        root,
        task_spec=root / "task.yaml",
        labels_path=root / "labels.parquet",
        splits_path=root / "splits.json",
        features_path=features if features.exists() else None,
    )


class TestCommittedData:
    @pytest.mark.parametrize("root", [CLEAN, LEAKY], ids=["clean", "leaky"])
    def test_the_dataset_is_present_and_complete(self, root: Path) -> None:
        for relative in (
            "data/shard_0.parquet",
            "metadata/dataset.json",
            "labels.parquet",
            "splits.json",
            "task.yaml",
            "manifest.json",
        ):
            assert (root / relative).exists(), f"{root.name}/{relative}"

    @pytest.mark.parametrize("root", [CLEAN, LEAKY], ids=["clean", "leaky"])
    def test_the_manifest_says_the_data_is_synthetic(self, root: Path) -> None:
        """Committed data in a clinical repository needs to say what it is."""
        manifest = json.loads((root / "manifest.json").read_text())
        assert "synthetic" in manifest["note"].lower()
        assert "No real patient record" in manifest["note"]

    @pytest.mark.parametrize("root", [CLEAN, LEAKY], ids=["clean", "leaky"])
    def test_every_subject_id_is_a_surrogate(self, root: Path) -> None:
        import re

        events = load(root).events
        pattern = re.compile(r"^S\d{6}(-DUP)?$")
        for subject in events["subject_id"].unique().to_list():
            assert pattern.match(subject), subject

    @pytest.mark.parametrize("root", [CLEAN, LEAKY], ids=["clean", "leaky"])
    def test_no_free_text_is_committed(self, root: Path) -> None:
        events = load(root).events
        assert events["text_value"].null_count() == events.height

    def test_the_examples_stay_small_enough_to_commit(self) -> None:
        total = sum(p.stat().st_size for p in EXAMPLES.rglob("*") if p.is_file())
        assert total < 2 * 1024 * 1024, f"{total} bytes of example data"


class TestCleanExample:
    def test_audits_to_exit_code_zero_with_nothing_skipped(self) -> None:
        """The baseline the demo rests on; if it drifts, the demo is a lie."""
        report = run_audit(load(CLEAN))
        assert report.skipped == [], [s.reason for s in report.skipped]
        assert report.fired == [], [(f.check_id, f.message) for f in report.fired]
        assert report.exit_code == 0

    def test_the_task_spec_weakens_no_check(self) -> None:
        from ehrlint.schemas.task import TaskSpec

        assert TaskSpec.from_yaml(CLEAN / "task.yaml").warnings() == []

    def test_it_has_no_feature_matrix(self) -> None:
        assert not (CLEAN / "features.parquet").exists()


class TestLeakyExample:
    def test_the_ground_truth_is_committed_beside_the_data(self) -> None:
        planted = json.loads((LEAKY / "planted_leaks.json").read_text())
        assert set(planted) == {"LK001", "LK002", "LK004", "LK010"}
        assert all(len(v) == 5 for v in planted.values())

    def test_every_planted_leak_is_found_with_the_right_subjects(self) -> None:
        planted = json.loads((LEAKY / "planted_leaks.json").read_text())
        report = run_audit(load(LEAKY))
        assert report.exit_code == 1
        for check_id, subjects in planted.items():
            finding = report.finding(check_id)
            assert finding is not None and finding.fired, check_id
            assert set(subjects) <= set(finding.subjects), check_id

    def test_nothing_fires_that_was_not_planted(self) -> None:
        """Four leaks were planted; a fifth finding would be a false positive."""
        planted = set(json.loads((LEAKY / "planted_leaks.json").read_text()))
        report = run_audit(load(LEAKY))
        assert {f.check_id for f in report.fired} == planted

    def test_it_ships_a_feature_matrix_so_lk001_is_runnable(self) -> None:
        features = pl.read_parquet(LEAKY / "features.parquet")
        assert "subject_id" in features.columns
        assert features.height > 0


class TestDemoScript:
    def test_the_demo_runs_and_exits_zero(self, tmp_path: Path) -> None:
        """Run as the README says, from the repository root."""
        result = subprocess.run(
            [sys.executable, "demo.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Done." in result.stdout

    def test_the_demo_verifies_itself_against_the_ground_truth(self) -> None:
        result = subprocess.run(
            [sys.executable, "demo.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert "found all 5 planted subject(s)" in result.stdout
        assert "FAIL" not in result.stdout

    def test_the_demo_writes_the_reports_it_advertises(self) -> None:
        subprocess.run(
            [sys.executable, "demo.py"], cwd=REPO, capture_output=True, timeout=600, check=True
        )
        out = REPO / "demo_output"
        for subdir in ("clean", "leaky"):
            for name in ("report.html", "report.md", "report.json", "findings.jsonl"):
                assert (out / subdir / name).exists(), f"{subdir}/{name}"

    def test_the_demo_output_is_not_committed(self) -> None:
        """It is generated, gitignored, and would otherwise go stale in the repo."""
        gitignore = (REPO / ".gitignore").read_text()
        assert "/demo_output/" in gitignore
