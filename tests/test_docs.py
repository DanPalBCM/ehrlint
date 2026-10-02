"""The documentation, where it makes checkable claims.

Two kinds of assertion here. The generated check pages must match what the
generator produces, so prose cannot drift from the code it describes. And the
hand-written pages must not reference a check, a CLI flag, or a nav entry that
does not exist — a docs example that no longer runs is a bug report from the
future.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
CHECKS = DOCS / "checks"
GENERATOR = REPO / "scripts" / "generate_check_docs.py"

CHECK_IDS = [f"LK{n:03d}" for n in range(1, 11)]


class TestGeneratedCheckPages:
    def test_there_is_a_page_per_check_plus_an_index(self) -> None:
        pages = sorted(p.name for p in CHECKS.glob("*.md"))
        expected = sorted(["index.md", *(f"{c.lower()}.md" for c in CHECK_IDS)])
        assert pages == expected

    def test_the_committed_pages_match_the_generator(self) -> None:
        """So the rationale, query, severity, and example cannot drift."""
        before = {p: p.read_text(encoding="utf-8") for p in sorted(CHECKS.glob("*.md"))}
        result = subprocess.run(
            [sys.executable, str(GENERATOR)],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        after = {p: p.read_text(encoding="utf-8") for p in sorted(CHECKS.glob("*.md"))}

        stale = sorted(p.name for p in before if before[p] != after.get(p))
        assert not stale, (
            f"these generated pages are out of date: {stale}. "
            "Run: uv run python scripts/generate_check_docs.py"
        )

    @pytest.mark.parametrize("check_id", CHECK_IDS)
    def test_each_page_documents_what_the_spec_asks_for(self, check_id: str) -> None:
        """SPEC §4.3: rationale, exact query, severity default, worked example."""
        text = (CHECKS / f"{check_id.lower()}.md").read_text(encoding="utf-8")
        assert "## Why this matters" in text
        assert "## The query it runs" in text
        assert "default severity" in text
        assert "## Worked example" in text
        assert "```sql" in text

    @pytest.mark.parametrize("check_id", CHECK_IDS)
    def test_each_page_shows_the_check_both_quiet_and_firing(self, check_id: str) -> None:
        """A page showing only the finding does not show that it is specific."""
        text = (CHECKS / f"{check_id.lower()}.md").read_text(encoding="utf-8")
        assert "did not fire" in text
        assert f"{check_id} " in text

    @pytest.mark.parametrize("check_id", CHECK_IDS)
    def test_each_page_is_reachable_from_the_index(self, check_id: str) -> None:
        index = (CHECKS / "index.md").read_text(encoding="utf-8")
        assert f"({check_id.lower()}.md)" in index


class TestNav:
    def test_every_nav_target_exists(self) -> None:
        config = yaml.safe_load(
            (REPO / "mkdocs.yml").read_text(encoding="utf-8").replace("!!python/name:", "")
        )

        def targets(node: object) -> list[str]:
            if isinstance(node, str):
                return [node]
            if isinstance(node, dict):
                return [t for v in node.values() for t in targets(v)]
            if isinstance(node, list):
                return [t for v in node for t in targets(v)]
            return []

        for target in targets(config["nav"]):
            assert (DOCS / target).exists(), target

    def test_every_docs_page_is_in_the_nav(self) -> None:
        """An unreachable page is one nobody will find or maintain."""
        config = yaml.safe_load((REPO / "mkdocs.yml").read_text(encoding="utf-8"))

        def targets(node: object) -> set[str]:
            if isinstance(node, str):
                return {node}
            if isinstance(node, dict):
                return {t for v in node.values() for t in targets(v)}
            if isinstance(node, list):
                return {t for v in node for t in targets(v)}
            return set()

        in_nav = targets(config["nav"])
        on_disk = {str(p.relative_to(DOCS)) for p in DOCS.rglob("*.md")}
        assert on_disk - in_nav == set(), f"not in the nav: {sorted(on_disk - in_nav)}"


class TestHandWrittenClaims:
    def test_no_page_references_an_unknown_check_id(self) -> None:
        from ehrlint.checks.base import check_ids

        known = set(check_ids()) | {"LK099", "LK098", "LK097", "LK096", "LK101", "LK102", "LK103"}
        for path in DOCS.rglob("*.md"):
            for match in set(re.findall(r"\bLK\d{3}\b", path.read_text(encoding="utf-8"))):
                assert match in known, f"{path.name} references {match}"

    def test_the_cli_flags_the_docs_promise_exist(self) -> None:
        from typer.testing import CliRunner

        from ehrlint.cli import app

        runner = CliRunner()
        for command in ("audit", "synth", "validate-task", "list-checks"):
            result = runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0, command

        audit_help = runner.invoke(app, ["audit", "--help"]).output
        for flag in (
            "--data",
            "--task",
            "--out",
            "--format",
            "--splits",
            "--labels",
            "--features",
            "--checks",
            "--fail-on",
        ):
            assert flag in audit_help, flag

    def test_the_readme_check_table_lists_every_check(self) -> None:
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        for check_id in CHECK_IDS:
            assert check_id in readme, check_id

    def test_the_decisions_page_records_the_required_verifications(self) -> None:
        """SPEC §0 requires the MEDS schema and the MIMIC access pattern be recorded."""
        text = (DOCS / "decisions.md").read_text(encoding="utf-8")
        assert "meds` 0.4.1" in text or "meds 0.4.1" in text
        assert "MIMIC-IV Clinical Database Demo" in text
        assert "ODC Open Database License" in text

    def test_the_decision_ids_are_sequential(self) -> None:
        text = (DOCS / "decisions.md").read_text(encoding="utf-8")
        found = [int(m) for m in re.findall(r"### D-(\d{3})", text)]
        assert found == list(range(1, len(found) + 1)), found
