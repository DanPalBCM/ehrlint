"""The check battery: LK001 through LK010.

Each check documents its rationale, runs one stated query, and returns findings
that carry the rows that triggered them. The `query` string on every finding is
SQL-ish pseudocode of the exact logic, so a reader can reproduce the result by
hand rather than trusting the implementation.

The checks live in one module rather than ten files. SPEC §4.1 sketches
`lk001_post_index.py`, `lk002_same_encounter.py`, and so on, but they share the
index-date join, the code-set matcher, and the skip vocabulary — splitting them
would mean ten files importing the same six helpers and a reader hopping
between them to compare two checks that differ by one predicate. See
``docs/decisions.md`` D-006.
"""

from __future__ import annotations

import polars as pl

from ehrlint.checks.base import BaseCheck, CheckResult, register
from ehrlint.context import AuditContext
from ehrlint.schemas.events import (
    CODE,
    CODE_SYSTEM,
    ENCOUNTER_ID,
    MEDS_DEATH_CODE,
    SUBJECT_ID,
    TIME,
)
from ehrlint.schemas.findings import Finding, Severity, make_finding

#: Columns carried in example rows for event-level findings. Enough to find
#: the row in the source data, no more.
_EVENT_EXAMPLE_COLUMNS = [SUBJECT_ID, TIME, CODE, CODE_SYSTEM, "index_date"]

#: LK009 refuses to report duplicates when distinct demographic signatures
#: number fewer than this fraction of subjects. Below it, a collision is the
#: birthday paradox rather than a duplicate record.
MIN_SIGNATURE_DISTINCTNESS = 0.9

#: How the outcome window treats its lower bound: **open at the index date**.
#:
#: An occurrence stamped at exactly the index date is not predictable from the
#: index date, so it does not count as an outcome. This matters more than it
#: sounds on real data: most EHR extracts date a visit's diagnoses by the
#: admission timestamp, so an entire encounter's codes -- including the ones
#: entered at discharge -- land exactly on the index moment. Counting them
#: would make every such task look positive and well predicted. LK002 is the
#: check that reports them for what they are.
#:
#: Stated as a constant because a task spec written `[index_date, index_date +
#: 2y]` reads closed, and a user whose own label query used `>=` has to be able
#: to explain the difference. Every message that depends on this boundary says
#: "strictly after" rather than "inside the declared window".
OUTCOME_WINDOW_EXCLUDES_INDEX_MOMENT = True


class PostIndexFeatures(BaseCheck):
    """LK001 — events used as predictors that happen after the index date."""

    id = "LK001"
    name = "post_index_features"
    description = "Events or features timestamped after the index date."
    rationale = (
        "A model that sees the future cannot be deployed. This is the most common and "
        "most damaging leak: a feature computed over a window that extends past the "
        "prediction time will often be the model's strongest signal, and it will be "
        "entirely unavailable at inference."
    )
    severity_default = Severity.ERROR

    def run(self, ctx: AuditContext) -> CheckResult:
        if ctx.index_dates().height == 0:
            return self.skip(
                "no index date could be resolved for any subject, so 'after the index "
                "date' is undefined"
            )

        findings: list[Finding] = []

        # The feature matrix is the stronger evidence when it exists: it shows
        # what the model actually saw rather than what the event stream allows.
        if ctx.features is not None and ctx.features.time_column is not None:
            findings.append(self._check_matrix(ctx))

        joined = ctx.events_with_index()
        query = (
            "SELECT * FROM events e JOIN index_dates i USING (subject_id)\n"
            "WHERE e.time > i.index_date"
        )
        offending = joined.filter(pl.col(TIME) > pl.col("index_date"))

        # Reported as `info`, not `error`. Post-index events are not a bug --
        # outcomes and follow-up live there, so every real dataset has them,
        # and flagging their mere presence as an error would make LK001
        # permanently red and teach users to ignore it. What matters is whether
        # the *feature pipeline* consumed them, which ehrlint cannot see from
        # the event stream alone. Supply --features for the real check.
        findings.append(
            make_finding(
                self.id,
                name=f"{self.name}_events",
                severity=Severity.INFO,
                message=(
                    f"{offending[SUBJECT_ID].n_unique()} of {ctx.n_subjects} subject(s) "
                    f"have {offending.height} event(s) after their index date. That is "
                    "expected -- outcomes live there. ehrlint cannot tell which events "
                    "your feature pipeline consumed; supply --features to check the "
                    "matrix directly."
                    if offending.height
                    else "No event falls after its subject's index date."
                ),
                offending=offending,
                n_total_subjects=ctx.n_subjects,
                query=query,
                example_columns=_EVENT_EXAMPLE_COLUMNS,
                detail={
                    "n_post_index_events": offending.height,
                    "severity_note": (
                        "info rather than error: the presence of post-index events is "
                        "normal. Only a feature matrix can show that they leaked."
                    ),
                },
            )
        )
        return findings

    def _check_matrix(self, ctx: AuditContext) -> Finding:
        """The direct check: feature rows stamped after the index date."""
        features = ctx.features
        assert features is not None and features.time_column is not None
        index = ctx.index_dates()
        joined = features.df.join(index, on="subject_id", how="inner")
        offending = joined.filter(pl.col(features.time_column) > pl.col("index_date"))
        return make_finding(
            f"{self.id}",
            name=f"{self.name}_matrix",
            severity=Severity.ERROR,
            message=(
                f"{offending['subject_id'].n_unique()} feature row(s) are stamped after "
                "the index date, so the matrix itself contains future information."
                if offending.height
                else "Every feature row is stamped at or before its index date."
            ),
            offending=offending,
            n_total_subjects=ctx.n_subjects,
            query=(
                f"SELECT * FROM features f JOIN index_dates i USING (subject_id)\n"
                f"WHERE f.{features.time_column} > i.index_date"
            ),
            example_columns=["subject_id", features.time_column, "index_date"],
            detail={"source": "feature_matrix", "time_column": features.time_column},
        )


class SameEncounter(BaseCheck):
    """LK002 — outcome codes recorded in the same encounter as the index event."""

    id = "LK002"
    name = "same_encounter"
    description = "Outcome codes from the index encounter itself."
    rationale = (
        "Codes are often entered at the end of an encounter, after the outcome is "
        "known. A diagnosis coded during the index visit may postdate the prediction "
        "moment even though its timestamp does not, because the timestamp is the "
        "visit's, not the moment of entry. Same-encounter outcome codes are therefore "
        "suspect regardless of their recorded time."
    )
    severity_default = Severity.ERROR

    def run(self, ctx: AuditContext) -> CheckResult:
        if not ctx.has_encounters:
            return self.skip(
                "no event carries an encounter_id. MEDS has no encounter column; supply "
                "one via --encounter-column, or use OMOP input where "
                "visit_occurrence_id is available"
            )
        if ctx.index_dates().height == 0:
            return self.skip("no index date could be resolved for any subject")
        outcome_codes = ctx.task.outcome.codes
        if not outcome_codes:
            return self.skip(
                "the task spec declares no outcome codes, so there is nothing to look "
                "for in the index encounter"
            )

        joined = ctx.events_with_index()
        if joined.height == 0:
            return self.skip("no timed event could be joined to an index date")

        # The encounter that contains the index moment, per subject.
        index_encounters = (
            joined.filter(pl.col(TIME) == pl.col("index_date"))
            .select([SUBJECT_ID, ENCOUNTER_ID])
            .drop_nulls()
            .unique()
            .rename({ENCOUNTER_ID: "index_encounter"})
        )
        if index_encounters.height == 0:
            return self.skip(
                "no event sits exactly at an index date, so the index encounter could "
                "not be identified"
            )

        outcomes = ctx.match_code_sets(joined, outcome_codes)
        if outcomes.height == 0:
            return self.clean(
                ctx,
                "No outcome code appears anywhere in the event data.",
                "SELECT * FROM events WHERE code IN outcome_codes",
            )

        offending = outcomes.join(index_encounters, on=SUBJECT_ID, how="inner").filter(
            pl.col(ENCOUNTER_ID) == pl.col("index_encounter")
        )

        return [
            make_finding(
                self.id,
                name=self.name,
                severity=self.severity_default,
                message=(
                    f"{offending[SUBJECT_ID].n_unique()} subject(s) have an outcome code "
                    "recorded in the same encounter as the index event. Coding often "
                    "happens after the outcome is known."
                    if offending.height
                    else "No outcome code shares an encounter with the index event."
                ),
                offending=offending,
                n_total_subjects=ctx.n_subjects,
                query=(
                    "WITH idx AS (\n"
                    "  SELECT subject_id, encounter_id AS index_encounter FROM events e\n"
                    "  JOIN index_dates i USING (subject_id) WHERE e.time = i.index_date\n"
                    ")\n"
                    "SELECT * FROM events e JOIN idx USING (subject_id)\n"
                    "WHERE e.code IN outcome_codes AND e.encounter_id = idx.index_encounter"
                ),
                example_columns=[SUBJECT_ID, TIME, CODE, ENCOUNTER_ID, "index_encounter"],
            )
        ]


class OutcomeProxy(BaseCheck):
    """LK003 — proxy codes for the outcome present before the index date."""

    id = "LK003"
    name = "outcome_proxy"
    description = "User-listed proxy codes occurring pre-index."
    rationale = (
        "A treatment given only to cases is a label in disguise. If insulin infusion "
        "appears pre-index in a DKA prediction task, the model learns the treatment, "
        "not the risk, and will fail on patients who have not yet been treated. "
        "Proxies are task-specific and cannot be inferred, so they must be declared in "
        "the task spec."
    )
    severity_default = Severity.WARNING

    def run(self, ctx: AuditContext) -> CheckResult:
        proxies = ctx.task.outcome.proxy_codes
        if not proxies:
            return self.skip(
                "the task spec declares no outcome.proxy_codes. ehrlint cannot infer "
                "which codes imply your outcome -- list the treatments or findings that "
                "only cases receive"
            )
        if ctx.index_dates().height == 0:
            return self.skip("no index date could be resolved for any subject")

        joined = ctx.events_with_index()
        matched = ctx.match_code_sets(joined, proxies)
        offending = matched.filter(pl.col(TIME) <= pl.col("index_date"))

        return [
            make_finding(
                self.id,
                name=self.name,
                severity=self.severity_default,
                message=(
                    f"{offending[SUBJECT_ID].n_unique()} subject(s) have a declared "
                    "outcome proxy before the index date. A model may be learning the "
                    "treatment rather than the risk."
                    if offending.height
                    else "No declared outcome proxy appears before the index date."
                ),
                offending=offending,
                n_total_subjects=ctx.n_subjects,
                query=(
                    "SELECT * FROM events e JOIN index_dates i USING (subject_id)\n"
                    "WHERE e.code IN proxy_codes AND e.time <= i.index_date"
                ),
                example_columns=_EVENT_EXAMPLE_COLUMNS,
                detail={"n_proxy_codes": len(ctx.task.outcome.all_proxies())},
            )
        ]


class ImmortalTime(BaseCheck):
    """LK004 — negatives who were not observed for the full horizon."""

    id = "LK004"
    name = "immortal_time"
    description = "Subjects who could not have had the outcome during follow-up."
    rationale = (
        "A subject labelled negative who left the data after one month is not a "
        "two-year negative -- they are censored. Counting them as negatives means the "
        "negative class is partly made of people who had no opportunity to have the "
        "outcome, which inflates every metric and is the classic immortal-time error."
    )
    severity_default = Severity.ERROR

    def run(self, ctx: AuditContext) -> CheckResult:
        if ctx.task.horizon is None:
            return self.skip("the task spec declares no horizon, so follow-up cannot be checked")
        if not ctx.task.followup.require_full_horizon_for_negatives:
            return self.skip(
                "followup.require_full_horizon_for_negatives is false, so the task "
                "accepts partial follow-up for negatives by design"
            )
        index = ctx.index_dates()
        if index.height == 0:
            return self.skip("no index date could be resolved for any subject")

        horizon_days = ctx.task.horizon.days
        timed = ctx.timed()

        # Last observed event per subject: the end of their follow-up.
        last_seen = (
            timed.group_by(SUBJECT_ID)
            .agg(pl.col(TIME).max().alias("last_event_time"))
            .rename({SUBJECT_ID: "subject_id"})
        )

        # Subjects who had the outcome do not need full follow-up -- the event
        # happened, so there is no immortal time to worry about.
        outcome_subjects: set[str] = set()
        if ctx.task.outcome.codes:
            matched = ctx.match_code_sets(timed, ctx.task.outcome.codes)
            outcome_subjects = set(matched[SUBJECT_ID].drop_nulls().unique().to_list())

        # Death (or another competing event) legitimately ends follow-up.
        censored: set[str] = set()
        if ctx.task.followup.allow_death_as_censoring:
            competing = ctx.task.outcome.all_competing() or {MEDS_DEATH_CODE}
            died = timed.filter(pl.col(CODE).is_in(list(competing)))
            censored = set(died[SUBJECT_ID].drop_nulls().unique().to_list())

        frame = index.join(last_seen, on="subject_id", how="left").with_columns(
            ((pl.col("last_event_time") - pl.col("index_date")).dt.total_seconds() / 86400.0).alias(
                "observed_days"
            )
        )

        exempt = outcome_subjects | censored
        offending = frame.filter(
            (pl.col("observed_days").is_null() | (pl.col("observed_days") < horizon_days))
            & ~pl.col("subject_id").is_in(list(exempt) or [""])
        ).with_columns(
            pl.lit(horizon_days).alias("required_days"),
            pl.lit(str(ctx.task.horizon)).alias("horizon"),
        )

        return [
            make_finding(
                self.id,
                name=self.name,
                severity=self.severity_default,
                message=(
                    f"{offending['subject_id'].n_unique()} subject(s) are observed for "
                    f"less than the {ctx.task.horizon} horizon without having the "
                    "outcome or a competing event. Labelling them negative counts "
                    "people who had no opportunity to have the outcome."
                    if offending.height
                    else f"Every negative subject is observed for the full "
                    f"{ctx.task.horizon} horizon."
                ),
                offending=offending,
                n_total_subjects=ctx.n_subjects,
                subject_column="subject_id",
                query=(
                    "WITH followup AS (\n"
                    "  SELECT subject_id, MAX(time) AS last_event_time FROM events\n"
                    "  WHERE time IS NOT NULL GROUP BY subject_id\n"
                    ")\n"
                    "SELECT * FROM index_dates i LEFT JOIN followup f USING (subject_id)\n"
                    f"WHERE DATEDIFF('day', i.index_date, f.last_event_time) < {horizon_days:g}\n"
                    "  AND subject_id NOT IN (outcome_subjects UNION competing_event_subjects)"
                ),
                example_columns=[
                    "subject_id",
                    "index_date",
                    "last_event_time",
                    "observed_days",
                    "required_days",
                    "horizon",
                ],
                detail={
                    "horizon_days": horizon_days,
                    "n_outcome_subjects_exempt": len(outcome_subjects),
                    "n_censored_exempt": len(censored),
                    "allow_death_as_censoring": ctx.task.followup.allow_death_as_censoring,
                },
            )
        ]


class SplitOverlap(BaseCheck):
    """LK005 — a subject appearing in more than one split."""

    id = "LK005"
    name = "split_overlap"
    description = "Subject ids present in more than one split."
    rationale = (
        "The same patient in train and test means the test metric is partly measuring "
        "memorization. In EHR data this happens easily: splitting on encounters, "
        "admissions, or rows rather than on people puts one patient's records on both "
        "sides."
    )
    severity_default = Severity.ERROR

    def run(self, ctx: AuditContext) -> CheckResult:
        if ctx.splits is None:
            return self.skip("no splits were supplied")
        if len(ctx.splits.names) < 2:
            return self.skip(
                f"only {len(ctx.splits.names)} split is defined, so there is nothing to "
                "overlap with"
            )

        rows: list[dict[str, object]] = []
        for subject, names in sorted(ctx.splits.split_of().items()):
            if len(names) > 1:
                rows.append({"subject_id": subject, "splits": ",".join(sorted(names))})

        offending = (
            pl.DataFrame(rows)
            if rows
            else pl.DataFrame(schema={"subject_id": pl.String, "splits": pl.String})
        )
        n_assigned = len(ctx.splits.all_subjects())

        return [
            make_finding(
                self.id,
                name=self.name,
                severity=self.severity_default,
                message=(
                    f"{len(rows)} subject(s) appear in more than one split, so the test "
                    "metric is partly measuring memorization."
                    if rows
                    else f"The {len(ctx.splits.names)} splits are disjoint by subject."
                ),
                offending=offending,
                n_total_subjects=n_assigned or ctx.n_subjects,
                subject_column="subject_id",
                query=(
                    "SELECT subject_id, COUNT(DISTINCT split) AS n_splits\n"
                    "FROM subject_splits GROUP BY subject_id HAVING n_splits > 1"
                ),
                example_columns=["subject_id", "splits"],
                detail={
                    "split_sizes": ctx.splits.sizes(),
                    "overlapping_pairs": {
                        f"{a}|{b}": len(ids) for (a, b), ids in ctx.splits.overlaps().items()
                    },
                },
            )
        ]


class TemporalSplit(BaseCheck):
    """LK006 — a temporal split that does not actually respect time."""

    id = "LK006"
    name = "temporal_split"
    description = "Test index dates that precede the training period's end."
    rationale = (
        "A split described as temporal but not ordered in time reports an optimistic "
        "number: the model is evaluated on a period it was partly trained on, so it "
        "benefits from knowing the era's coding practices, formulary, and case mix."
    )
    severity_default = Severity.WARNING

    def run(self, ctx: AuditContext) -> CheckResult:
        if ctx.splits is None:
            return self.skip("no splits were supplied")
        from ehrlint.schemas.splits import TEST, TRAIN

        if TRAIN not in ctx.splits.assignments or TEST not in ctx.splits.assignments:
            return self.skip(f"a train and a test split are both needed; found {ctx.splits.names}")
        index = ctx.index_dates()
        if index.height == 0:
            return self.skip("no index date could be resolved for any subject")

        # A split declared temporal is an error when violated; one that never
        # claimed to be is only a warning.
        severity = Severity.ERROR if ctx.task.splits.temporal else Severity.WARNING

        train_ids = ctx.splits.get(TRAIN)
        test_ids = ctx.splits.get(TEST)
        train_index = index.filter(pl.col("subject_id").is_in(train_ids))
        if train_index.height == 0:
            return self.skip("no training subject has a resolvable index date")

        train_max = train_index["index_date"].max()
        # Rendered once: a Polars `.max()` is typed loosely enough that
        # interpolating it directly could stringify bytes as `b'...'`.
        train_max_text = str(train_max)
        offending = (
            index.filter(pl.col("subject_id").is_in(test_ids))
            .filter(pl.col("index_date") < train_max)
            .with_columns(pl.lit(train_max).alias("train_max_index_date"))
        )

        return [
            make_finding(
                self.id,
                name=self.name,
                severity=severity,
                message=(
                    f"{offending['subject_id'].n_unique()} test subject(s) have an index "
                    f"date before the training period ends ({train_max_text}). "
                    + (
                        "The task spec declares this split temporal, so this invalidates it."
                        if ctx.task.splits.temporal
                        else "The split does not claim to be temporal; this is reported for review."
                    )
                    if offending.height
                    else f"Every test index date falls at or after the training maximum "
                    f"({train_max_text})."
                ),
                offending=offending,
                n_total_subjects=len(test_ids) or ctx.n_subjects,
                subject_column="subject_id",
                query=(
                    "WITH train_max AS (\n"
                    "  SELECT MAX(index_date) AS t FROM index_dates\n"
                    "  WHERE subject_id IN train_split\n"
                    ")\n"
                    "SELECT * FROM index_dates WHERE subject_id IN test_split\n"
                    "  AND index_date < (SELECT t FROM train_max)"
                ),
                example_columns=["subject_id", "index_date", "train_max_index_date"],
                detail={
                    "train_max_index_date": train_max_text,
                    "declared_temporal": ctx.task.splits.temporal,
                    "n_test_subjects": len(test_ids),
                },
            )
        ]


class LabelDefinition(BaseCheck):
    """LK007 — the outcome ascertained with information from after the window."""

    id = "LK007"
    name = "label_definition"
    description = "Labels that depend on information outside the outcome window."
    rationale = (
        "If a subject is labelled positive on the strength of an outcome code that "
        "falls outside the declared window, the label is not the label the task "
        "defines. The model is then trained toward a target nobody specified, and the "
        "reported task definition does not describe what was learned."
    )
    severity_default = Severity.ERROR

    def run(self, ctx: AuditContext) -> CheckResult:
        if ctx.labels is None or "label" not in ctx.labels.columns:
            return self.skip(
                "no labels table was supplied, so ehrlint cannot compare the given "
                "labels against the outcome definition"
            )
        if not ctx.task.outcome.codes:
            return self.skip("the task spec declares no outcome codes")
        if ctx.task.horizon is None:
            return self.skip("the task spec declares no horizon")
        if ctx.labels["label"].null_count() == ctx.labels.height:
            return self.skip("the labels table carries no label values")

        index = ctx.index_dates()
        horizon_days = ctx.task.horizon.days
        timed = ctx.timed()
        outcomes = ctx.match_code_sets(timed, ctx.task.outcome.codes)

        # Outcome occurrences inside the declared window, per subject.
        in_window = (
            outcomes.join(index, left_on=SUBJECT_ID, right_on="subject_id", how="inner")
            .filter(
                (pl.col(TIME) > pl.col("index_date"))
                & (
                    (pl.col(TIME) - pl.col("index_date")).dt.total_seconds() / 86400.0
                    <= horizon_days
                )
            )
            .select(SUBJECT_ID)
            .unique()
            .rename({SUBJECT_ID: "subject_id"})
            .with_columns(pl.lit(True).alias("outcome_in_window"))
        )

        labelled = (
            ctx.labels.select(["subject_id", "label"])
            .drop_nulls("label")
            .join(in_window, on="subject_id", how="left")
            .with_columns(pl.col("outcome_in_window").fill_null(False))
        )

        # A positive label with no in-window outcome: the label came from
        # somewhere the task definition does not reach.
        offending = labelled.filter(pl.col("label") & ~pl.col("outcome_in_window")).with_columns(
            pl.lit(str(ctx.task.horizon)).alias("horizon"),
            pl.lit(
                "positive label with no outcome code strictly after the index date "
                "and within the horizon"
            ).alias("why"),
        )

        return [
            make_finding(
                self.id,
                name=self.name,
                severity=self.severity_default,
                message=(
                    f"{offending['subject_id'].n_unique()} subject(s) are labelled "
                    f"positive with no outcome code strictly after the index date and "
                    f"within {ctx.task.horizon} of it, so the label was ascertained "
                    "from information the task definition does not describe. An "
                    "occurrence stamped at exactly the index date does not count here "
                    "-- it is not predictable from that moment, and LK002 reports it."
                    if offending.height
                    else f"Every positive label is supported by an outcome code "
                    f"strictly after the index date and within {ctx.task.horizon}."
                ),
                offending=offending,
                n_total_subjects=labelled.height or ctx.n_subjects,
                subject_column="subject_id",
                query=(
                    "WITH in_window AS (\n"
                    "  SELECT DISTINCT e.subject_id FROM events e\n"
                    "  JOIN index_dates i USING (subject_id)\n"
                    "  WHERE e.code IN outcome_codes AND e.time > i.index_date\n"
                    f"    AND DATEDIFF('day', i.index_date, e.time) <= {horizon_days:g}\n"
                    ")\n"
                    "SELECT * FROM labels l WHERE l.label = TRUE\n"
                    "  AND l.subject_id NOT IN (SELECT subject_id FROM in_window)"
                ),
                example_columns=["subject_id", "label", "outcome_in_window", "horizon", "why"],
                detail={"horizon_days": horizon_days, "n_labelled": labelled.height},
            )
        ]


class EncounterGroup(BaseCheck):
    """LK008 — one encounter or provider contributing to two splits."""

    id = "LK008"
    name = "encounter_group"
    description = "A group (encounter, provider) straddling two splits."
    rationale = (
        "Records from one encounter share a clinician, a note template, and a coding "
        "pattern. When the same group appears in train and test, the model can "
        "recognize the group rather than the clinical signal -- the same failure as "
        "subject overlap, one level up."
    )
    severity_default = Severity.ERROR

    def run(self, ctx: AuditContext) -> CheckResult:
        if ctx.splits is None:
            return self.skip("no splits were supplied")
        if len(ctx.splits.names) < 2:
            return self.skip("fewer than two splits are defined")

        group_column = ctx.task.splits.group_column or ENCOUNTER_ID
        if group_column not in ctx.events.columns:
            return self.skip(
                f"the group column {group_column!r} is not in the event data; set "
                "splits.group_column in the task spec or supply encounter ids"
            )
        if ctx.events[group_column].null_count() == ctx.events.height:
            return self.skip(f"every {group_column} is null")

        assignment = ctx.subject_split()
        grouped = (
            ctx.events.select([SUBJECT_ID, group_column])
            .drop_nulls()
            .unique()
            .join(assignment, left_on=SUBJECT_ID, right_on="subject_id", how="inner")
        )
        if grouped.height == 0:
            return self.skip(
                "no event could be joined to a split; the subject ids in the splits may "
                "not match those in the event data"
            )

        straddling = (
            grouped.group_by(group_column)
            .agg(
                pl.col("split").n_unique().alias("n_splits"),
                pl.col("split").unique().sort().str.join(",").alias("splits"),
                pl.col(SUBJECT_ID).n_unique().alias("n_subjects_in_group"),
            )
            .filter(pl.col("n_splits") > 1)
            .sort(group_column)
        )

        # One row per (group, subject), not one per group. A straddling group
        # has at least two subjects by definition, and naming one of them
        # leaves the reader to re-derive the rest -- while `n_subjects` and the
        # machine-readable subject list would both understate the damage by
        # half. The group count stays in the message and in `detail`.
        members = (
            grouped.join(
                straddling.select(group_column, "splits", "n_splits"),
                on=group_column,
                how="inner",
            )
            .unique()
            .sort([group_column, SUBJECT_ID])
        )

        return [
            make_finding(
                self.id,
                name=self.name,
                severity=self.severity_default,
                message=(
                    f"{straddling.height} {group_column} value(s) spanning "
                    f"{members[SUBJECT_ID].n_unique()} subject(s) contribute to more "
                    "than one split, so the model can learn the group rather than the "
                    "clinical signal."
                    if straddling.height
                    else f"No {group_column} value spans two splits."
                ),
                offending=members,
                n_total_subjects=ctx.n_subjects,
                subject_column=SUBJECT_ID,
                query=(
                    f"SELECT {group_column}, COUNT(DISTINCT split) AS n_splits\n"
                    f"FROM events e JOIN subject_splits s USING (subject_id)\n"
                    f"WHERE {group_column} IS NOT NULL\n"
                    f"GROUP BY {group_column} HAVING n_splits > 1"
                ),
                example_columns=[group_column, SUBJECT_ID, "split", "splits", "n_splits"],
                detail={"group_column": group_column, "n_groups": straddling.height},
            )
        ]


class DuplicateSubjects(BaseCheck):
    """LK009 — one person under more than one subject id."""

    id = "LK009"
    name = "duplicate_subjects"
    description = "Subjects that look like the same person under two ids."
    rationale = (
        "Patients get duplicate records across registrations, sites, and merges. Two "
        "ids for one person defeat a subject-level split in exactly the way LK005 "
        "guards against, but invisibly -- the split is disjoint by id while not being "
        "disjoint by person."
    )
    severity_default = Severity.WARNING

    def run(self, ctx: AuditContext) -> CheckResult:
        static = ctx.events.filter(pl.col(TIME).is_null())
        if static.height == 0:
            return self.skip(
                "no static (null-time) events are present, so there are no demographics "
                "to match on. Deterministic matching needs demographic facts"
            )

        # A deterministic signature: the sorted multiset of a subject's static
        # codes. Deliberately not probabilistic -- a fuzzy match would produce
        # findings nobody can verify, and §0 forbids unverifiable statistics.
        signatures = (
            static.select([SUBJECT_ID, CODE])
            .drop_nulls()
            .unique()
            .group_by(SUBJECT_ID)
            .agg(pl.col(CODE).unique().sort().str.join("|").alias("signature"))
        )
        if signatures.height == 0:
            return self.skip("no subject has any static code to build a signature from")

        # Only signatures with enough facts to be discriminating. One shared
        # demographic code is not evidence of anything.
        signatures = signatures.with_columns(
            pl.col("signature").str.count_matches(r"\|").add(1).alias("n_facts")
        ).filter(pl.col("n_facts") >= 2)

        collisions = (
            signatures.group_by("signature")
            .agg(
                pl.col(SUBJECT_ID).n_unique().alias("n_subject_ids"),
                pl.col(SUBJECT_ID).unique().sort().str.join(",").alias("subject_ids"),
                pl.col("n_facts").first().alias("n_facts"),
            )
            .filter(pl.col("n_subject_ids") > 1)
            .sort("signature")
        )

        # Guard against a signature too coarse to mean anything. If the
        # demographics in this dataset only distinguish a handful of profiles,
        # collisions are arithmetic rather than evidence -- and listing them as
        # candidate duplicates would be exactly the fabricated statistic §0
        # forbids. Refuse to conclude instead.
        n_signatures = signatures["signature"].n_unique()
        n_signed_subjects = signatures[SUBJECT_ID].n_unique()
        if n_signed_subjects and n_signatures / n_signed_subjects < MIN_SIGNATURE_DISTINCTNESS:
            return self.skip(
                f"the static demographics distinguish only {n_signatures} profile(s) "
                f"across {n_signed_subjects} subject(s), so collisions are expected by "
                "arithmetic rather than evidence of duplication. Deterministic matching "
                "needs higher-cardinality facts (a birth date, a postal code) than this "
                "dataset carries"
            )

        # One row per colliding id, not one per group. A duplicate finding whose
        # subject list names one of the two ids is not actionable: the whole
        # claim is that *these two ids* are one person.
        members = signatures.join(
            collisions.select("signature", "n_subject_ids", "subject_ids"),
            on="signature",
            how="inner",
        ).sort(["signature", SUBJECT_ID])

        return [
            make_finding(
                self.id,
                name=self.name,
                severity=self.severity_default,
                message=(
                    f"{collisions.height} group(s) covering "
                    f"{members[SUBJECT_ID].n_unique()} subject id(s) share an identical "
                    "static demographic signature and may be the same person under "
                    "multiple ids."
                    if collisions.height
                    else "No two subject ids share an identical demographic signature."
                ),
                offending=members,
                n_total_subjects=ctx.n_subjects,
                subject_column=SUBJECT_ID,
                query=(
                    "WITH sig AS (\n"
                    "  SELECT subject_id, STRING_AGG(DISTINCT code, '|' ORDER BY code) AS signature\n"
                    "  FROM events WHERE time IS NULL GROUP BY subject_id\n"
                    ")\n"
                    "SELECT signature, COUNT(DISTINCT subject_id) AS n_subject_ids\n"
                    "FROM sig GROUP BY signature HAVING n_subject_ids > 1"
                ),
                example_columns=[SUBJECT_ID, "signature", "subject_ids", "n_subject_ids"],
                detail={
                    "method": "deterministic match on the sorted set of static codes",
                    "min_facts": 2,
                    "n_groups": collisions.height,
                    "note": (
                        "Deterministic, not probabilistic. A shared signature is a "
                        "candidate for review, not a confirmed duplicate -- twins and "
                        "sparse demographics both produce collisions."
                    ),
                },
            )
        ]


class HorizonViolation(BaseCheck):
    """LK010 — outcomes outside the declared horizon counted as positives."""

    id = "LK010"
    name = "horizon_violation"
    description = "Outcome occurrences beyond the declared horizon."
    rationale = (
        "A two-year task whose outcome window actually reaches five years is a "
        "five-year task reported as two. Every metric is then describing a different "
        "question from the one the paper states, and the model will underperform when "
        "deployed against the stated horizon."
    )
    severity_default = Severity.ERROR

    def run(self, ctx: AuditContext) -> CheckResult:
        if ctx.task.horizon is None:
            return self.skip("the task spec declares no horizon")
        if not ctx.task.outcome.codes:
            return self.skip("the task spec declares no outcome codes")
        index = ctx.index_dates()
        if index.height == 0:
            return self.skip("no index date could be resolved for any subject")

        horizon_days = ctx.task.horizon.days
        outcomes = ctx.match_code_sets(ctx.timed(), ctx.task.outcome.codes)
        if outcomes.height == 0:
            return self.clean(
                ctx,
                "No outcome code appears in the event data.",
                "SELECT * FROM events WHERE code IN outcome_codes",
            )

        joined = outcomes.join(
            index, left_on=SUBJECT_ID, right_on="subject_id", how="inner"
        ).with_columns(
            ((pl.col(TIME) - pl.col("index_date")).dt.total_seconds() / 86400.0).alias(
                "days_after_index"
            )
        )

        # Subjects whose *only* outcome occurrence is beyond the horizon. A
        # subject with an in-window occurrence is a legitimate positive, and
        # their later recurrences are not a violation.
        in_window = set(
            joined.filter(
                (pl.col("days_after_index") > 0) & (pl.col("days_after_index") <= horizon_days)
            )[SUBJECT_ID]
            .unique()
            .to_list()
        )
        offending = joined.filter(
            (pl.col("days_after_index") > horizon_days)
            & ~pl.col(SUBJECT_ID).is_in(list(in_window) or [""])
        ).with_columns(
            pl.lit(horizon_days).alias("horizon_days"),
            pl.lit(str(ctx.task.horizon)).alias("horizon"),
        )

        return [
            make_finding(
                self.id,
                name=self.name,
                severity=self.severity_default,
                message=(
                    f"{offending[SUBJECT_ID].n_unique()} subject(s) have their only "
                    f"outcome occurrence beyond the declared {ctx.task.horizon} horizon. "
                    "If they are labelled positive, the task's real horizon is longer "
                    "than stated."
                    if offending.height
                    else f"Every outcome occurrence falls within the declared "
                    f"{ctx.task.horizon} horizon."
                ),
                offending=offending,
                n_total_subjects=ctx.n_subjects,
                query=(
                    "SELECT * FROM events e JOIN index_dates i USING (subject_id)\n"
                    "WHERE e.code IN outcome_codes\n"
                    f"  AND DATEDIFF('day', i.index_date, e.time) > {horizon_days:g}\n"
                    "  AND e.subject_id NOT IN (subjects with an in-window occurrence)"
                ),
                example_columns=[
                    SUBJECT_ID,
                    TIME,
                    CODE,
                    "index_date",
                    "days_after_index",
                    "horizon_days",
                    "horizon",
                ],
                detail={
                    "horizon_days": horizon_days,
                    "n_in_window_subjects": len(in_window),
                },
            )
        ]


#: The built-in battery, in id order.
BUILTIN_CHECKS: tuple[type[BaseCheck], ...] = (
    PostIndexFeatures,
    SameEncounter,
    OutcomeProxy,
    ImmortalTime,
    SplitOverlap,
    TemporalSplit,
    LabelDefinition,
    EncounterGroup,
    DuplicateSubjects,
    HorizonViolation,
)


def register_builtins(*, replace: bool = True) -> None:
    """Register every built-in check.

    Called at import time of :mod:`ehrlint.checks`. Idempotent by default, so
    re-importing does not raise on duplicate ids.
    """
    for cls in BUILTIN_CHECKS:
        register(cls(), replace=replace)
