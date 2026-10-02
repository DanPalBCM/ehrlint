"""The check registry and the custom-check extension point."""

from __future__ import annotations

import polars as pl
import pytest

from ehrlint.checks.base import (
    BaseCheck,
    Check,
    all_checks,
    check_ids,
    clear_registry,
    get_check,
    register,
    register_check,
    resolve_checks,
)
from ehrlint.context import AuditContext
from ehrlint.exceptions import RegistryError
from ehrlint.schemas.findings import Finding, Severity, make_finding

BUILTIN_IDS = [f"LK{n:03d}" for n in range(1, 11)]


class TestBuiltins:
    def test_all_ten_leakage_checks_are_registered(self) -> None:
        assert check_ids() == BUILTIN_IDS

    def test_every_check_carries_a_rationale(self) -> None:
        """A finding without a reason is a number the user cannot act on."""
        for check in all_checks():
            assert check.metadata()["rationale"].strip(), check.id  # type: ignore[attr-defined]

    def test_every_check_carries_a_description(self) -> None:
        for check in all_checks():
            assert check.description.strip(), check.id

    def test_every_check_satisfies_the_protocol(self) -> None:
        for check in all_checks():
            assert isinstance(check, Check)

    def test_the_registry_is_ordered_by_id_so_runs_are_reproducible(self) -> None:
        assert [c.id for c in all_checks()] == sorted(c.id for c in all_checks())


class TestGetCheck:
    def test_is_case_insensitive(self) -> None:
        assert get_check("lk001").id == "LK001"

    def test_tolerates_surrounding_space(self) -> None:
        assert get_check(" LK001 ").id == "LK001"

    def test_an_unknown_id_lists_the_registered_ones(self) -> None:
        """A typo in --checks must not silently run nothing."""
        with pytest.raises(RegistryError, match="LK001"):
            get_check("LK999")


class TestResolveChecks:
    def test_none_means_everything(self) -> None:
        assert len(resolve_checks(None)) == len(BUILTIN_IDS)

    def test_an_empty_list_means_everything(self) -> None:
        assert len(resolve_checks([])) == len(BUILTIN_IDS)

    def test_a_selection_is_deduplicated_and_sorted(self) -> None:
        resolved = resolve_checks(["LK003", "lk001", "LK003"])
        assert [c.id for c in resolved] == ["LK001", "LK003"]

    def test_an_unknown_id_in_a_selection_raises(self) -> None:
        with pytest.raises(RegistryError):
            resolve_checks(["LK001", "LK404"])


class TestRegister:
    def test_a_duplicate_id_is_refused_by_default(self) -> None:
        """Two checks under one id would make a report ambiguous."""

        class Dup(BaseCheck):
            id = "LK001"
            name = "duplicate"

        with pytest.raises(RegistryError):
            register(Dup())

    def test_replace_is_explicit(self) -> None:
        class Dup(BaseCheck):
            id = "LK001"
            name = "replacement"

        register(Dup(), replace=True)
        assert get_check("LK001").name == "replacement"

    def test_the_registry_can_be_cleared_for_a_test(self) -> None:
        clear_registry()
        assert check_ids() == []


class TestRegisterCheckDecorator:
    def test_registers_a_plain_function_as_a_check(self, clean_context: AuditContext) -> None:
        @register_check(id="LK099", name="my_check", severity="warning")
        def my_check(ctx: AuditContext) -> list[Finding]:
            """A custom check."""
            return [
                make_finding(
                    "LK099",
                    name="my_check",
                    severity=Severity.WARNING,
                    message="nothing wrong",
                    offending=pl.DataFrame(),
                    n_total_subjects=ctx.n_subjects,
                    query="select 1",
                )
            ]

        assert "LK099" in check_ids()
        registered = get_check("LK099")
        assert registered.severity_default is Severity.WARNING
        assert registered.run(clean_context)[0].check_id == "LK099"  # type: ignore[index]

    def test_returns_the_undecorated_function_so_it_stays_testable(self) -> None:
        @register_check(id="LK098", name="x")
        def my_check(ctx: AuditContext) -> list[Finding]:
            return []

        assert my_check.__name__ == "my_check"
        assert callable(my_check)

    def test_the_docstring_becomes_the_description_when_none_is_given(self) -> None:
        @register_check(id="LK097", name="x")
        def my_check(ctx: AuditContext) -> list[Finding]:
            """First line is the description."""
            return []

        assert get_check("LK097").description == "First line is the description."

    def test_a_custom_check_joins_the_default_battery(self) -> None:
        @register_check(id="LK096", name="x")
        def my_check(ctx: AuditContext) -> list[Finding]:
            return []

        assert "LK096" in [c.id for c in resolve_checks(None)]


def test_clean_produces_a_pass_not_a_skip(clean_context: AuditContext) -> None:
    """A pass says what was verified; a skip says why nothing was."""

    class Passing(BaseCheck):
        id = "LK095"
        name = "passing"
        severity_default = Severity.ERROR

    findings = Passing().clean(clean_context, "nothing found", "select 1")
    assert findings[0].fired is False
    assert findings[0].n_subjects == 0


def test_skip_carries_the_check_identity(clean_context: AuditContext) -> None:
    class Skipping(BaseCheck):
        id = "LK094"
        name = "skipping"

    skip = Skipping().skip("no encounter ids in the data")
    assert skip.check_id == "LK094"
    assert "encounter" in skip.reason
