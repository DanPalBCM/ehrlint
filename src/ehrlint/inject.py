"""Leak injectors: one per check.

The testing strategy this module exists for. Generate clean data, assert no
check fires. Inject one specific leak, assert **that** check fires and names
the subjects it was planted into. That is ground truth no amount of staring at
a real dataset can give you.

Each injector returns the mutated dataset **and the subject ids it touched**,
so a test can assert the check found the right people rather than merely
finding someone. A check that fires on the wrong subjects is as broken as one
that does not fire.

SPEC §11 requires injector/check duality: every check has an injector, and an
injected leak is always caught. A property test asserts it over seeds and
subject counts.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import polars as pl

from ehrlint.exceptions import EhrlintError
from ehrlint.schemas.events import coerce_events
from ehrlint.schemas.splits import TEST, TRAIN
from ehrlint.synth.generator import (
    HISTORY_SYSTEM,
    OUTCOME_CODES,
    OUTCOME_SYSTEM,
    PROXY_CODES,
    PROXY_SYSTEM,
    SyntheticDataset,
)


@dataclass
class Injection:
    """The result of planting a leak.

    Attributes:
        dataset: The mutated dataset.
        check_id: The check this leak should trigger.
        subjects: Subject ids the leak was planted into. The ground truth a
            test compares the finding against.
        description: What was done, for the test's failure message.
    """

    dataset: SyntheticDataset
    check_id: str
    subjects: list[str]
    description: str


def _sample(
    candidates: list[str],
    rng: random.Random,
    n: int,
    exclude: set[str] | None = None,
) -> list[str]:
    """Choose up to ``n`` subjects deterministically, skipping ``exclude``.

    ``exclude`` is what makes several injectors composable. Every injector is
    seeded the same way, so without it they all choose the *same* subjects --
    and then one leak can undo another: LK010 gives a negative an outcome long
    after the horizon, which extends the follow-up LK004 had just truncated,
    and LK004 stops firing. A multi-leak dataset exists to be ground truth, so
    the leaks have to land on disjoint people.
    """
    pool = [s for s in candidates if not exclude or s not in exclude]
    if not pool:
        raise EhrlintError(
            f"no subject is left to inject into: {len(candidates)} candidate(s), "
            f"{len(exclude or ())} already carrying a planted leak. Generate more "
            "subjects or plant fewer leaks"
        )
    return sorted(rng.sample(pool, k=min(n, len(pool))))


def _pick(
    dataset: SyntheticDataset,
    rng: random.Random,
    n: int,
    exclude: set[str] | None = None,
) -> list[str]:
    """Choose ``n`` subjects from the whole cohort."""
    candidates = sorted(dataset.events["subject_id"].unique().to_list())
    if not candidates:
        raise EhrlintError("cannot inject a leak into a dataset with no subjects")
    return _sample(candidates, rng, n, exclude)


def _append_events(dataset: SyntheticDataset, rows: list[dict[str, Any]]) -> None:
    """Append event rows in place, coercing to the canonical schema."""
    if not rows:
        return
    extra = coerce_events(pl.DataFrame(rows), source="inject")
    dataset.events = pl.concat([dataset.events, extra], how="vertical")


def _index_dates(dataset: SyntheticDataset) -> dict[str, Any]:
    """Subject id to index date, from the labels table."""
    return {str(r["subject_id"]): r["index_date"] for r in dataset.labels.iter_rows(named=True)}


# ---------------------------------------------------------------------------
# the injectors
# ---------------------------------------------------------------------------


def inject_lk001(
    dataset: SyntheticDataset,
    *,
    n: int = 5,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> Injection:
    """LK001 — add a feature matrix whose rows postdate the index date.

    The clean baseline already contains post-index *events* (outcomes live
    there), so planting more would not distinguish a leak from normal data.
    The unambiguous leak is a **feature matrix** stamped after the index date,
    which is what LK001's matrix branch checks.
    """
    rng = random.Random(seed)
    out = dataset.copy()
    chosen = _pick(out, rng, n, exclude)
    index = _index_dates(out)

    rows = []
    for subject, when in index.items():
        leaked = subject in chosen
        as_of = when + timedelta(days=90) if leaked else when
        rows.append(
            {
                "subject_id": subject,
                "prediction_time": as_of,
                "n_prior_visits": rng.randint(0, 9),
                "last_a1c": round(rng.uniform(5.0, 12.0), 2),
            }
        )

    out.features = pl.DataFrame(rows)
    out.injected.append("LK001")
    return Injection(
        dataset=out,
        check_id="LK001",
        subjects=chosen,
        description=(
            f"attached a feature matrix whose rows for {len(chosen)} subject(s) are "
            "stamped 90 days after their index date"
        ),
    )


def inject_lk002(
    dataset: SyntheticDataset,
    *,
    n: int = 5,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> Injection:
    """LK002 — put an outcome code inside the index encounter."""
    rng = random.Random(seed)
    out = dataset.copy()
    chosen = _pick(out, rng, n, exclude)
    index = _index_dates(out)

    rows = []
    for subject in chosen:
        when = index.get(subject)
        if when is None:
            continue
        rows.append(
            {
                "subject_id": subject,
                "time": when,
                "code": rng.choice(OUTCOME_CODES),
                "code_system": OUTCOME_SYSTEM,
                "numeric_value": None,
                "text_value": None,
                # The index encounter, which is what makes this the leak.
                "encounter_id": f"{subject}-IDX",
            }
        )
    _append_events(out, rows)
    out.injected.append("LK002")
    return Injection(
        dataset=out,
        check_id="LK002",
        subjects=sorted({r["subject_id"] for r in rows}),
        description=f"added an outcome code to the index encounter of {len(rows)} subject(s)",
    )


def inject_lk003(
    dataset: SyntheticDataset,
    *,
    n: int = 5,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> Injection:
    """LK003 — put the outcome proxy before the index date."""
    rng = random.Random(seed)
    out = dataset.copy()
    chosen = _pick(out, rng, n, exclude)
    index = _index_dates(out)

    rows = []
    for subject in chosen:
        when = index.get(subject)
        if when is None:
            continue
        rows.append(
            {
                "subject_id": subject,
                "time": when - timedelta(days=rng.uniform(1.0, 30.0)),
                "code": rng.choice(PROXY_CODES),
                "code_system": PROXY_SYSTEM,
                "numeric_value": None,
                "text_value": None,
                "encounter_id": f"{subject}-PROXY",
            }
        )
    _append_events(out, rows)
    out.injected.append("LK003")
    return Injection(
        dataset=out,
        check_id="LK003",
        subjects=sorted({r["subject_id"] for r in rows}),
        description=f"added a pre-index outcome proxy for {len(rows)} subject(s)",
    )


def inject_lk004(
    dataset: SyntheticDataset,
    *,
    n: int = 5,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> Injection:
    """LK004 — truncate a negative's follow-up well inside the horizon.

    Every event after index + 30 days is removed for the chosen subjects, and
    they keep their negative label. They are now censored patients counted as
    two-year negatives, which is the immortal-time error exactly.
    """
    rng = random.Random(seed)
    out = dataset.copy()
    negatives = sorted(out.labels.filter(~pl.col("label"))["subject_id"].to_list())
    if not negatives:
        raise EhrlintError("no negative subjects available to truncate")
    chosen = _sample(negatives, rng, n, exclude)
    index = _index_dates(out)

    cutoffs = pl.DataFrame(
        [{"subject_id": s, "cutoff": index[s] + timedelta(days=30)} for s in chosen if s in index]
    )
    if cutoffs.height == 0:
        raise EhrlintError("no chosen negative subject has an index date")

    joined = out.events.join(cutoffs, on="subject_id", how="left")
    out.events = (
        joined.filter(
            pl.col("cutoff").is_null()
            | pl.col("time").is_null()
            | (pl.col("time") <= pl.col("cutoff"))
        )
        .drop("cutoff")
        .select(out.events.columns)
    )
    out.injected.append("LK004")
    return Injection(
        dataset=out,
        check_id="LK004",
        subjects=sorted(cutoffs["subject_id"].to_list()),
        description=(
            f"truncated follow-up to 30 days for {cutoffs.height} negative subject(s) "
            "while keeping their negative label"
        ),
    )


def inject_lk005(
    dataset: SyntheticDataset,
    *,
    n: int = 5,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> Injection:
    """LK005 — copy test subjects into the train split."""
    rng = random.Random(seed)
    out = dataset.copy()
    test_subjects = sorted(out.splits.filter(pl.col("split") == TEST)["subject_id"].to_list())
    if not test_subjects:
        raise EhrlintError("no test subjects available to duplicate")
    chosen = _sample(test_subjects, rng, n, exclude)

    extra = pl.DataFrame({"subject_id": chosen, "split": [TRAIN] * len(chosen)})
    out.splits = pl.concat([out.splits, extra], how="vertical")
    out.injected.append("LK005")
    return Injection(
        dataset=out,
        check_id="LK005",
        subjects=chosen,
        description=f"added {len(chosen)} test subject(s) to the train split as well",
    )


def inject_lk006(
    dataset: SyntheticDataset,
    *,
    n: int = 5,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> Injection:
    """LK006 — move the earliest train subjects into the test split.

    The split stays disjoint — this is not LK005 — but the test set now
    contains the oldest index dates, so a split declared temporal is not.
    """
    rng = random.Random(seed)
    _ = rng
    out = dataset.copy()
    index = out.labels.select(["subject_id", "index_date"])
    train_subjects = (
        out.splits.filter(pl.col("split") == TRAIN)
        .join(index, on="subject_id", how="inner")
        .sort("index_date")
    )
    if train_subjects.height == 0:
        raise EhrlintError("no train subjects available to move")
    # The earliest ones specifically -- not a sample -- because that is what
    # breaks a temporal split. `exclude` is honoured by skipping them in order
    # rather than by sampling, so the choice stays deterministic.
    eligible = [
        s for s in train_subjects["subject_id"].to_list() if not exclude or s not in exclude
    ]
    if not eligible:
        raise EhrlintError(
            "every train subject already carries a planted leak, so LK006 has none left to move"
        )
    chosen = sorted(eligible[:n])

    out.splits = out.splits.with_columns(
        pl.when(pl.col("subject_id").is_in(chosen))
        .then(pl.lit(TEST))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    out.injected.append("LK006")
    return Injection(
        dataset=out,
        check_id="LK006",
        subjects=chosen,
        description=(
            f"moved the {len(chosen)} earliest train subject(s) into the test split, so "
            "test index dates now precede the training maximum"
        ),
    )


def inject_lk007(
    dataset: SyntheticDataset,
    *,
    n: int = 5,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> Injection:
    """LK007 — flip negatives to positive without any in-window outcome.

    The label now asserts an outcome the event data does not support inside
    the declared window, which is what "ascertained with post-index
    information" looks like from the outside.
    """
    rng = random.Random(seed)
    out = dataset.copy()
    negatives = sorted(out.labels.filter(~pl.col("label"))["subject_id"].to_list())
    if not negatives:
        raise EhrlintError("no negative subjects available to relabel")
    chosen = _sample(negatives, rng, n, exclude)

    out.labels = out.labels.with_columns(
        pl.when(pl.col("subject_id").is_in(chosen))
        .then(pl.lit(True))
        .otherwise(pl.col("label"))
        .alias("label")
    )
    out.injected.append("LK007")
    return Injection(
        dataset=out,
        check_id="LK007",
        subjects=chosen,
        description=(
            f"relabelled {len(chosen)} subject(s) positive with no outcome code inside "
            "the declared window"
        ),
    )


def inject_lk008(
    dataset: SyntheticDataset,
    *,
    n: int = 3,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> Injection:
    """LK008 — give a train and a test subject the same encounter id.

    Subject-level splits stay disjoint, so LK005 is clean; the encounter is
    what straddles.
    """
    rng = random.Random(seed)
    out = dataset.copy()
    train_subjects = sorted(out.splits.filter(pl.col("split") == TRAIN)["subject_id"].to_list())
    test_subjects = sorted(out.splits.filter(pl.col("split") == TEST)["subject_id"].to_list())
    if not train_subjects or not test_subjects:
        raise EhrlintError("both a train and a test split are needed")

    pairs = list(
        zip(
            _sample(train_subjects, rng, n, exclude),
            _sample(test_subjects, rng, n, exclude),
            strict=False,
        )
    )
    index = _index_dates(out)

    rows = []
    touched: list[str] = []
    for i, (train_subject, test_subject) in enumerate(pairs):
        shared = f"SHARED-ENC-{i}"
        for subject in (train_subject, test_subject):
            when = index.get(subject)
            if when is None:
                continue
            rows.append(
                {
                    "subject_id": subject,
                    "time": when - timedelta(days=5),
                    "code": "I10",
                    "code_system": HISTORY_SYSTEM,
                    "numeric_value": None,
                    "text_value": None,
                    "encounter_id": shared,
                }
            )
            touched.append(subject)
    _append_events(out, rows)
    out.injected.append("LK008")
    return Injection(
        dataset=out,
        check_id="LK008",
        subjects=sorted(set(touched)),
        description=(
            f"gave {len(pairs)} train/test subject pair(s) a shared encounter id while "
            "keeping the subject split disjoint"
        ),
    )


def inject_lk009(
    dataset: SyntheticDataset,
    *,
    n: int = 3,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> Injection:
    """LK009 — duplicate a subject under a second id with identical demographics."""
    rng = random.Random(seed)
    out = dataset.copy()
    chosen = _pick(out, rng, n, exclude)

    rows: list[dict[str, Any]] = []
    touched: list[str] = []
    for subject in chosen:
        twin = f"{subject}-DUP"
        own = out.events.filter(pl.col("subject_id") == subject)
        for row in own.iter_rows(named=True):
            copy = dict(row)
            copy["subject_id"] = twin
            if copy.get("encounter_id"):
                copy["encounter_id"] = str(copy["encounter_id"]).replace(subject, twin)
            rows.append(copy)
        touched.extend([subject, twin])

    _append_events(out, rows)

    # The duplicate needs a label and a split, or the duplicate id would be
    # invisible to every other check and the dataset would be inconsistent.
    index = _index_dates(out)
    new_labels = [
        {
            "subject_id": f"{s}-DUP",
            "index_date": index.get(s),
            "label": bool(out.labels.filter(pl.col("subject_id") == s)["label"].first()),
        }
        for s in chosen
        if index.get(s) is not None
    ]
    if new_labels:
        out.labels = pl.concat(
            [out.labels, pl.DataFrame(new_labels).select(out.labels.columns)],
            how="vertical",
        )
    existing_split = {
        str(r["subject_id"]): str(r["split"]) for r in out.splits.iter_rows(named=True)
    }
    new_splits = [{"subject_id": f"{s}-DUP", "split": existing_split.get(s, TRAIN)} for s in chosen]
    out.splits = pl.concat([out.splits, pl.DataFrame(new_splits)], how="vertical")

    out.injected.append("LK009")
    return Injection(
        dataset=out,
        check_id="LK009",
        subjects=sorted(set(touched)),
        description=(
            f"duplicated {len(chosen)} subject(s) under a second id with identical "
            "static demographics"
        ),
    )


def inject_lk010(
    dataset: SyntheticDataset,
    *,
    n: int = 5,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> Injection:
    """LK010 — give negatives an outcome occurrence beyond the horizon.

    They have no in-window occurrence, so if they are counted as positives the
    task's real horizon is longer than declared.
    """
    rng = random.Random(seed)
    out = dataset.copy()
    negatives = sorted(out.labels.filter(~pl.col("label"))["subject_id"].to_list())
    if not negatives:
        raise EhrlintError("no negative subjects available")
    chosen = _sample(negatives, rng, n, exclude)
    index = _index_dates(out)
    horizon_days = out.task.horizon.days if out.task.horizon else 730.5

    rows = []
    for subject in chosen:
        when = index.get(subject)
        if when is None:
            continue
        rows.append(
            {
                "subject_id": subject,
                "time": when + timedelta(days=horizon_days + rng.uniform(30.0, 400.0)),
                "code": rng.choice(OUTCOME_CODES),
                "code_system": OUTCOME_SYSTEM,
                "numeric_value": None,
                "text_value": None,
                "encounter_id": f"{subject}-LATE",
            }
        )
    _append_events(out, rows)
    out.injected.append("LK010")
    return Injection(
        dataset=out,
        check_id="LK010",
        subjects=sorted({r["subject_id"] for r in rows}),
        description=(
            f"added an outcome occurrence beyond the horizon for {len(rows)} subject(s) "
            "with no in-window occurrence"
        ),
    )


#: Check id to injector. SPEC §11's duality requirement is that this covers
#: every registered check; a test asserts it.
INJECTORS: dict[str, Callable[..., Injection]] = {
    "LK001": inject_lk001,
    "LK002": inject_lk002,
    "LK003": inject_lk003,
    "LK004": inject_lk004,
    "LK005": inject_lk005,
    "LK006": inject_lk006,
    "LK007": inject_lk007,
    "LK008": inject_lk008,
    "LK009": inject_lk009,
    "LK010": inject_lk010,
}


def inject(
    dataset: SyntheticDataset,
    check_id: str,
    *,
    n: int = 5,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> Injection:
    """Plant the leak that should trigger ``check_id``.

    Args:
        dataset: The dataset to copy and mutate.
        check_id: Which leak to plant.
        n: How many subjects to affect.
        seed: Seed; the same arguments plant the same leak.
        exclude: Subjects to leave alone, normally those already carrying a
            planted leak.

    Raises:
        EhrlintError: If no injector exists for the check, listing those that
            do.
    """
    key = check_id.strip().upper()
    if key not in INJECTORS:
        raise EhrlintError(
            f"no injector for {check_id!r}; available: {', '.join(sorted(INJECTORS))}"
        )
    return INJECTORS[key](dataset, n=n, seed=seed, exclude=exclude)


def inject_many(
    dataset: SyntheticDataset, check_ids: list[str], *, n: int = 5, seed: int = 0
) -> tuple[SyntheticDataset, dict[str, list[str]]]:
    """Plant several leaks in sequence, on disjoint subjects.

    Disjoint because every injector is seeded identically and would otherwise
    choose the same people -- and then one leak masks another. Giving a
    negative an outcome long after the horizon (LK010) extends the follow-up
    that LK004 had just truncated, and LK004 goes quiet. A dataset built to be
    ground truth cannot have that in it.

    Returns:
        ``(dataset, {check_id: [subject_id]})``. Injectors are applied in
        sorted id order so the result is deterministic regardless of the order
        the caller listed them.
    """
    current = dataset
    planted: dict[str, list[str]] = {}
    touched: set[str] = set()
    for check_id in sorted({c.strip().upper() for c in check_ids if c.strip()}):
        result = inject(current, check_id, n=n, seed=seed, exclude=touched)
        touched.update(result.subjects)
        current = result.dataset
        planted[check_id] = result.subjects
    return current, planted


def injector_ids() -> list[str]:
    """Check ids that have an injector, sorted."""
    return sorted(INJECTORS)
