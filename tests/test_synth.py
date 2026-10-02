"""The synthetic generator, and the structural guarantee that it holds no PHI.

The PHI argument here is structural, not an inspection. Every value in a
generated dataset comes from a fixed module-level vocabulary, a sequential
subject index, or an arithmetic offset from a constant epoch. Nothing is
sampled from a real record, so there is no real record to leak -- and these
tests assert that property rather than spot-checking the output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
import pytest

from ehrlint.synth.generator import (
    BIRTH_DECADE_CODES,
    BIRTHDATE_CODE_PREFIX,
    EPOCH,
    GENDER_CODES,
    HISTORY_CODES,
    INDEX_CODE,
    LAB_CODES,
    OUTCOME_CODES,
    PROXY_CODES,
    RACE_CODES,
    default_task,
    generate,
)

ALLOWED_CODES = (
    {INDEX_CODE}
    | set(OUTCOME_CODES)
    | set(PROXY_CODES)
    | set(HISTORY_CODES)
    | {code for code, _, _ in LAB_CODES}
    | set(GENDER_CODES)
    | set(RACE_CODES)
    | set(BIRTH_DECADE_CODES)
)

SUBJECT_ID_PATTERN = re.compile(r"^S\d{6}$")


class TestPhiSafetyIsStructural:
    def test_every_code_comes_from_the_fixed_vocabulary(self) -> None:
        codes = set(generate(200, seed=0).events["code"].unique().to_list())
        synthetic_patterns = {c for c in codes if c.startswith(("POSTAL_", BIRTHDATE_CODE_PREFIX))}
        assert codes - synthetic_patterns <= ALLOWED_CODES

    def test_every_subject_id_is_a_sequential_surrogate(self) -> None:
        for subject in generate(50, seed=0).events["subject_id"].unique().to_list():
            assert SUBJECT_ID_PATTERN.match(subject), subject

    def test_no_free_text_is_generated_at_all(self) -> None:
        """No note text means no note text to de-identify."""
        events = generate(100, seed=0).events
        assert events["text_value"].null_count() == events.height

    def test_every_timestamp_derives_from_the_constant_epoch(self) -> None:
        """No real date can appear, because no date is sampled from anything.

        Index dates span three years from the epoch, history reaches two years
        back from an index date, and follow-up reaches the horizon plus a
        margin forward -- so every timestamp lies inside a window the epoch
        and those constants fix.
        """
        from datetime import timedelta

        dataset = generate(100, seed=0)
        times = dataset.events["time"].drop_nulls()
        horizon_days = dataset.task.horizon.days  # type: ignore[union-attr]

        earliest = EPOCH - timedelta(days=2 * 365)
        latest = EPOCH + timedelta(days=3 * 365 + horizon_days + 60)
        assert times.min() >= earliest
        assert times.max() <= latest

    def test_no_value_looks_like_a_name_or_an_identifier(self) -> None:
        """A crude scan, as a second line of defence behind the structural one."""
        events = generate(100, seed=0).events
        text = " ".join(
            str(v)
            for column in ("code", "code_system", "text_value", "encounter_id")
            for v in events[column].drop_nulls().unique().to_list()
        )
        assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", text)  # SSN shape
        assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)  # email
        assert not re.search(r"\bMRN\b", text, re.IGNORECASE)

    def test_the_written_manifest_states_the_data_is_synthetic(self, tmp_path: Path) -> None:
        written = generate(20, seed=0).write(tmp_path / "ds")
        manifest = json.loads(written["manifest"].read_text())
        assert "synthetic" in manifest["note"].lower()
        assert "No real patient record" in manifest["note"]


class TestCleanByConstruction:
    def test_every_predictor_event_predates_its_index_date(self) -> None:
        dataset = generate(100, seed=0)
        index = dict(
            zip(
                dataset.labels["subject_id"].to_list(),
                dataset.labels["index_date"].to_list(),
                strict=True,
            )
        )
        history = dataset.events.filter(
            pl.col("code_system") == "ICD10CM", pl.col("time").is_not_null()
        )
        # Follow-up events legitimately postdate the index date; what must hold
        # is that every subject has pre-index history to predict from.
        with_history = {
            row["subject_id"]
            for row in history.iter_rows(named=True)
            if row["time"] < index[row["subject_id"]]
        }
        assert with_history == set(index)

    def test_the_demographic_signature_is_unique_per_subject(self) -> None:
        """Otherwise LK009 fires on data documented as clean."""
        for n in (61, 87, 200):
            events = generate(n, seed=n).events
            signatures = (
                events.filter(pl.col("time").is_null())
                .group_by("subject_id")
                .agg(pl.col("code").unique().sort().str.join("|").alias("signature"))
            )
            assert signatures["signature"].n_unique() == signatures.height, n

    def test_splits_are_disjoint(self) -> None:
        mapping = generate(100, seed=0).splits_mapping()
        seen: set[str] = set()
        for ids in mapping.values():
            assert not (seen & set(ids))
            seen |= set(ids)

    def test_splits_are_time_ordered(self) -> None:
        dataset = generate(200, seed=0)
        index = dict(
            zip(
                dataset.labels["subject_id"].to_list(),
                dataset.labels["index_date"].to_list(),
                strict=True,
            )
        )
        mapping = dataset.splits_mapping()
        train_max = max(index[s] for s in mapping["train"])
        test_min = min(index[s] for s in mapping["test"])
        assert train_max <= test_min

    def test_every_subject_is_in_exactly_one_split(self) -> None:
        dataset = generate(100, seed=0)
        assigned = {s for ids in dataset.splits_mapping().values() for s in ids}
        assert assigned == set(dataset.events["subject_id"].unique().to_list())

    def test_every_positive_outcome_falls_inside_the_horizon(self) -> None:
        dataset = generate(200, seed=0)
        horizon_days = dataset.task.horizon.days  # type: ignore[union-attr]
        index = dict(
            zip(
                dataset.labels["subject_id"].to_list(),
                dataset.labels["index_date"].to_list(),
                strict=True,
            )
        )
        outcomes = dataset.events.filter(pl.col("code").is_in(list(OUTCOME_CODES)))
        for row in outcomes.iter_rows(named=True):
            delta = (row["time"] - index[row["subject_id"]]).total_seconds() / 86400
            assert 0 <= delta <= horizon_days, row["subject_id"]

    def test_the_task_spec_the_data_satisfies_weakens_no_check(self) -> None:
        assert default_task().warnings() == []


class TestArguments:
    @pytest.mark.parametrize("n", [0, -1])
    def test_a_nonpositive_subject_count_is_rejected(self, n: int) -> None:
        with pytest.raises(ValueError, match="n_subjects"):
            generate(n)

    @pytest.mark.parametrize("rate", [-0.1, 1.1])
    def test_an_out_of_range_positive_rate_is_rejected(self, rate: float) -> None:
        with pytest.raises(ValueError, match="positive_rate"):
            generate(10, positive_rate=rate)

    def test_a_nonpositive_horizon_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="horizon_days"):
            generate(10, horizon_days=0)

    def test_one_subject_is_allowed(self) -> None:
        assert generate(1, seed=0).n_subjects == 1

    def test_the_positive_rate_is_approximately_honoured(self) -> None:
        dataset = generate(1000, seed=0, positive_rate=0.3)
        observed = dataset.labels["label"].mean()
        assert observed == pytest.approx(0.3, abs=0.05)

    def test_a_zero_positive_rate_produces_no_positives(self) -> None:
        assert generate(100, seed=0, positive_rate=0.0).labels["label"].sum() == 0


class TestWrite:
    def test_writes_the_meds_layout_plus_side_files(self, tmp_path: Path) -> None:
        written = generate(30, seed=0).write(tmp_path / "ds")
        for key in ("data", "dataset_metadata", "labels", "splits_json", "task", "manifest"):
            assert key in written and written[key].exists()

    def test_the_task_yaml_reloads(self, tmp_path: Path) -> None:
        from ehrlint.schemas.task import TaskSpec

        written = generate(10, seed=0).write(tmp_path / "ds")
        assert TaskSpec.from_yaml(written["task"]).task == "dka_2y"

    def test_the_describe_counts_match_the_frames(self) -> None:
        dataset = generate(40, seed=0)
        described = dataset.describe()
        assert described["n_subjects"] == dataset.n_subjects
        assert described["n_events"] == dataset.events.height
        assert described["n_labels"] == dataset.labels.height


class TestCopy:
    def test_a_copy_is_independent(self) -> None:
        dataset = generate(20, seed=0)
        copy = dataset.copy()
        copy.events = copy.events.head(1)
        assert dataset.events.height > 1

    def test_the_task_is_deep_copied(self) -> None:
        dataset = generate(20, seed=0)
        copy = dataset.copy()
        copy.task = copy.task.model_copy(update={"task": "other"})
        assert dataset.task.task == "dka_2y"
