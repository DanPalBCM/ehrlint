"""The ten leakage checks, one class each.

Each class asserts three things: the check passes on clean data, it fires on
its own injected leak and names the planted subjects, and it skips with a
stated reason when its inputs are missing. The third is as important as the
others -- a check that silently returns clean when it cannot run is the one
failure mode that makes the whole tool untrustworthy.
"""

from __future__ import annotations

import polars as pl
import pytest

from ehrlint.audit import run_audit
from ehrlint.checks.base import get_check
from ehrlint.context import AuditContext
from ehrlint.inject import inject
from ehrlint.schemas.findings import Severity, SkippedCheck
from ehrlint.synth.generator import SyntheticDataset

from .conftest import make_context

ALL_IDS = [f"LK{n:03d}" for n in range(1, 11)]


def run_one(check_id: str, ctx: AuditContext):
    """Run a single check against a context."""
    return get_check(check_id).run(ctx)


def finding_for(check_id: str, data: SyntheticDataset):
    """Run one check over a dataset and return the finding that fired.

    A check may emit more than one finding -- LK001 describes post-index events
    as an observation *and* flags a leaky feature matrix as an error -- so the
    fired one is what an injector test is about.
    """
    result = run_one(check_id, make_context(data))
    assert not isinstance(result, SkippedCheck), f"{check_id} skipped: {result.reason}"
    assert result, f"{check_id} returned no findings at all"
    fired = [f for f in result if f.fired]
    if fired:
        assert len(fired) == 1, f"{check_id} fired {len(fired)} findings"
        return fired[0]
    return result[0]


@pytest.mark.parametrize("check_id", ALL_IDS)
def test_nothing_fires_on_clean_data(check_id: str, dataset: SyntheticDataset) -> None:
    """The baseline must be clean, or every injector test is meaningless."""
    result = run_one(check_id, make_context(dataset))
    assert not isinstance(result, SkippedCheck), f"{check_id} skipped: {result.reason}"
    for finding in result:
        assert finding.fired is False, f"{check_id}: {finding.message}"


@pytest.mark.parametrize("check_id", ALL_IDS)
def test_each_check_names_the_subjects_its_leak_was_planted_into(
    check_id: str, dataset: SyntheticDataset
) -> None:
    """A check that fires on the wrong subjects is as broken as one that is silent."""
    injection = inject(dataset, check_id, n=5, seed=3)
    finding = finding_for(check_id, injection.dataset)
    assert finding.fired, f"{check_id} did not fire on its own injected leak"
    assert set(injection.subjects) <= set(finding.subjects), (
        f"{check_id} fired but missed planted subjects: "
        f"{sorted(set(injection.subjects) - set(finding.subjects))}"
    )


@pytest.mark.parametrize("check_id", ALL_IDS)
def test_every_finding_carries_the_rows_that_produced_it(
    check_id: str, dataset: SyntheticDataset
) -> None:
    """SPEC §0. A count with no rows is an assertion nobody can check."""
    injection = inject(dataset, check_id, n=5, seed=3)
    finding = finding_for(check_id, injection.dataset)
    assert finding.examples, check_id
    assert finding.query.strip(), check_id


@pytest.mark.parametrize("check_id", ALL_IDS)
def test_every_finding_cites_rows_that_exist_in_the_input(
    check_id: str, dataset: SyntheticDataset
) -> None:
    injection = inject(dataset, check_id, n=5, seed=3)
    ctx = make_context(injection.dataset)
    finding = finding_for(check_id, injection.dataset)
    finding.validate_evidence(ctx.events)  # must not raise


class TestLk001PostIndexFeatures:
    def test_post_index_events_are_an_observation_not_an_error(
        self, dataset: SyntheticDataset
    ) -> None:
        """Outcomes and follow-up live after the index date.

        Flagging their mere presence as an error would make LK001 permanently
        red and teach users to ignore it.
        """
        findings = run_one("LK001", make_context(dataset))
        event_level = [f for f in findings if f.severity is Severity.INFO]
        assert event_level, "LK001 should describe post-index events"
        assert event_level[0].n_subjects > 0
        assert event_level[0].fired is False

    def test_a_feature_row_after_the_index_date_is_an_error(
        self, dataset: SyntheticDataset
    ) -> None:
        injection = inject(dataset, "LK001", n=5, seed=1)
        fired = [f for f in run_one("LK001", make_context(injection.dataset)) if f.fired]
        assert fired and fired[0].severity is Severity.ERROR

    def test_skips_the_matrix_branch_without_a_matrix(self, dataset: SyntheticDataset) -> None:
        assert dataset.features is None
        findings = run_one("LK001", make_context(dataset))
        assert all(f.severity is Severity.INFO for f in findings)


class TestLk002SameEncounter:
    def test_skips_when_no_event_has_an_encounter_id(self, dataset: SyntheticDataset) -> None:
        data = dataset.copy()
        data.events = data.events.with_columns(pl.lit(None, dtype=pl.String).alias("encounter_id"))
        result = run_one("LK002", make_context(data))
        assert isinstance(result, SkippedCheck)
        assert "encounter" in result.reason

    def test_fires_on_an_outcome_code_in_the_index_encounter(
        self, dataset: SyntheticDataset
    ) -> None:
        injection = inject(dataset, "LK002", n=4, seed=2)
        finding = finding_for("LK002", injection.dataset)
        assert finding.severity is Severity.ERROR


class TestLk003OutcomeProxy:
    def test_skips_when_the_task_declares_no_proxy_codes(self, dataset: SyntheticDataset) -> None:
        """Proxies are task-specific; ehrlint must not invent them."""
        data = dataset.copy()
        data.task = data.task.model_copy(
            update={"outcome": data.task.outcome.model_copy(update={"proxy_codes": []})}
        )
        result = run_one("LK003", make_context(data))
        assert isinstance(result, SkippedCheck)
        assert "proxy" in result.reason

    def test_is_a_warning_because_a_proxy_is_a_judgement_call(
        self, dataset: SyntheticDataset
    ) -> None:
        injection = inject(dataset, "LK003", n=4, seed=2)
        assert finding_for("LK003", injection.dataset).severity is Severity.WARNING


class TestLk004ImmortalTime:
    def test_fires_on_a_negative_with_truncated_follow_up(self, dataset: SyntheticDataset) -> None:
        injection = inject(dataset, "LK004", n=4, seed=2)
        finding = finding_for("LK004", injection.dataset)
        assert finding.severity is Severity.ERROR
        assert "horizon" in finding.message

    def test_skips_when_the_task_does_not_require_full_follow_up(
        self, dataset: SyntheticDataset
    ) -> None:
        data = dataset.copy()
        data.task = data.task.model_copy(
            update={
                "followup": data.task.followup.model_copy(
                    update={"require_full_horizon_for_negatives": False}
                )
            }
        )
        result = run_one("LK004", make_context(data))
        assert isinstance(result, SkippedCheck)


class TestLk005SplitOverlap:
    def test_skips_without_splits(self, dataset: SyntheticDataset) -> None:
        ctx = AuditContext.from_frames(
            dataset.events, task_spec=dataset.task, labels=dataset.labels, splits=None
        )
        result = run_one("LK005", ctx)
        assert isinstance(result, SkippedCheck)
        assert "split" in result.reason

    def test_fires_on_a_subject_in_two_splits(self, dataset: SyntheticDataset) -> None:
        injection = inject(dataset, "LK005", n=4, seed=2)
        finding = finding_for("LK005", injection.dataset)
        assert finding.severity is Severity.ERROR

    def test_the_example_rows_name_both_splits(self, dataset: SyntheticDataset) -> None:
        injection = inject(dataset, "LK005", n=4, seed=2)
        finding = finding_for("LK005", injection.dataset)
        assert any("split" in column for column in finding.example_columns)


class TestLk006TemporalSplit:
    def test_is_an_error_when_the_spec_declares_a_temporal_split(
        self, dataset: SyntheticDataset
    ) -> None:
        """A declared property that does not hold is a broken claim, not a hint."""
        assert dataset.task.splits.temporal is True
        injection = inject(dataset, "LK006", n=4, seed=2)
        assert finding_for("LK006", injection.dataset).severity is Severity.ERROR

    def test_is_only_a_warning_when_no_temporal_split_was_claimed(
        self, dataset: SyntheticDataset
    ) -> None:
        injection = inject(dataset, "LK006", n=4, seed=2)
        data = injection.dataset
        data.task = data.task.model_copy(
            update={"splits": data.task.splits.model_copy(update={"temporal": False})}
        )
        assert finding_for("LK006", data).severity is Severity.WARNING


class TestLk007LabelDefinition:
    def test_fires_on_a_positive_label_with_no_outcome_in_the_window(
        self, dataset: SyntheticDataset
    ) -> None:
        injection = inject(dataset, "LK007", n=4, seed=2)
        finding = finding_for("LK007", injection.dataset)
        assert finding.severity is Severity.ERROR

    def test_skips_without_labels(self, dataset: SyntheticDataset) -> None:
        ctx = AuditContext.from_frames(
            dataset.events,
            task_spec=dataset.task,
            labels=None,
            splits=dataset.splits_mapping(),
        )
        result = run_one("LK007", ctx)
        assert isinstance(result, SkippedCheck)


class TestLk008EncounterGroup:
    def test_fires_when_one_encounter_spans_two_splits(self, dataset: SyntheticDataset) -> None:
        injection = inject(dataset, "LK008", n=4, seed=2)
        finding = finding_for("LK008", injection.dataset)
        assert finding.severity is Severity.ERROR

    def test_names_every_subject_in_the_straddling_group(self, dataset: SyntheticDataset) -> None:
        """A group has at least two subjects; naming one understates by half."""
        injection = inject(dataset, "LK008", n=4, seed=2)
        finding = finding_for("LK008", injection.dataset)
        assert set(injection.subjects) <= set(finding.subjects)
        assert finding.detail["n_groups"] >= 1

    def test_skips_without_a_group_column(self, dataset: SyntheticDataset) -> None:
        data = dataset.copy()
        data.events = data.events.with_columns(pl.lit(None, dtype=pl.String).alias("encounter_id"))
        data.task = data.task.model_copy(
            update={"splits": data.task.splits.model_copy(update={"group_column": None})}
        )
        result = run_one("LK008", make_context(data))
        assert isinstance(result, SkippedCheck)


class TestLk009DuplicateSubjects:
    def test_fires_on_a_subject_duplicated_under_a_second_id(
        self, dataset: SyntheticDataset
    ) -> None:
        injection = inject(dataset, "LK009", n=4, seed=2)
        finding = finding_for("LK009", injection.dataset)
        assert finding.severity is Severity.WARNING

    def test_skips_when_the_demographic_signature_is_not_discriminating(
        self, dataset: SyntheticDataset
    ) -> None:
        """A coarse signature collides by chance, and a chance collision is not
        a duplicate patient. Reporting one would be a confident false finding."""
        from ehrlint.synth.generator import BIRTHDATE_CODE_PREFIX

        data = dataset.copy()
        data.events = data.events.filter(
            ~pl.col("code").str.starts_with("POSTAL_")
            & ~pl.col("code").str.starts_with(BIRTHDATE_CODE_PREFIX)
        )
        result = run_one("LK009", make_context(data))
        assert isinstance(result, SkippedCheck)
        assert "arithmetic rather than evidence" in result.reason

    def test_names_both_ids_of_a_duplicate_pair(self, dataset: SyntheticDataset) -> None:
        """ "These two ids are one person" is not actionable with one id."""
        injection = inject(dataset, "LK009", n=4, seed=2)
        finding = finding_for("LK009", injection.dataset)
        assert set(injection.subjects) <= set(finding.subjects)


class TestLk010HorizonViolation:
    def test_fires_on_an_outcome_beyond_the_declared_horizon(
        self, dataset: SyntheticDataset
    ) -> None:
        injection = inject(dataset, "LK010", n=4, seed=2)
        finding = finding_for("LK010", injection.dataset)
        assert finding.severity is Severity.ERROR

    def test_skips_without_a_horizon(self, dataset: SyntheticDataset) -> None:
        data = dataset.copy()
        data.task = data.task.model_copy(
            update={
                "horizon": None,
                "followup": data.task.followup.model_copy(
                    update={"require_full_horizon_for_negatives": False}
                ),
            }
        )
        result = run_one("LK010", make_context(data))
        assert isinstance(result, SkippedCheck)


class TestSkipsAreNotPasses:
    def test_an_audit_of_an_empty_table_skips_rather_than_passing(self) -> None:
        """The most dangerous input: nothing to find, and nothing found."""
        from ehrlint.schemas.events import empty_events
        from ehrlint.synth.generator import default_task

        ctx = AuditContext.from_frames(empty_events(), task_spec=default_task())
        report = run_audit(ctx)
        assert report.summary.n_checks_skipped > 0
        assert report.summary.n_checks_skipped + report.summary.n_checks_run == len(ALL_IDS)

    def test_every_skip_states_a_reason(self) -> None:
        from ehrlint.schemas.events import empty_events
        from ehrlint.synth.generator import default_task

        ctx = AuditContext.from_frames(empty_events(), task_spec=default_task())
        report = run_audit(ctx)
        for skip in report.skipped:
            assert len(skip.reason.strip()) > 10, skip.check_id
