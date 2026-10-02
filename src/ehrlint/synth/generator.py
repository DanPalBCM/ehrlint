"""Clean synthetic EHR data, deterministic given a seed.

This is the only data ehrlint ships or tests against. Nothing here derives from
a real record: subjects are sequential, codes come from a fixed vocabulary, and
every timestamp is computed from a constant epoch.

The baseline is **clean by construction** — no post-index events used as
features, full follow-up for negatives, disjoint splits, outcomes inside the
horizon. That matters because the test strategy is differential: generate
clean data, assert no check fires, inject one specific leak, assert exactly
that check fires. A baseline that already leaked would make every injector test
meaningless.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from ehrlint.schemas.events import coerce_events
from ehrlint.schemas.splits import TEST, TRAIN, VALID
from ehrlint.schemas.task import TaskSpec

#: Fixed epoch for every generated timestamp.
EPOCH = datetime(2018, 1, 1, 8, 0, 0)

#: The index event code, matching the default task spec.
INDEX_CODE = "ed_visit"
INDEX_SYSTEM = "VISIT"

#: Outcome codes. Two, so a code set with several values is exercised.
OUTCOME_CODES: tuple[str, ...] = ("E10.10", "E11.65")
OUTCOME_SYSTEM = "ICD10CM"

#: A treatment only cases receive -- the proxy LK003 looks for.
PROXY_CODES: tuple[str, ...] = ("insulin_drip",)
PROXY_SYSTEM = "PROCEDURE"

#: Ordinary pre-index history, the legitimate predictors.
HISTORY_CODES: tuple[str, ...] = (
    "I10",
    "E78.5",
    "J45.909",
    "N18.3",
    "F32.9",
    "E66.9",
    "K21.9",
    "M19.90",
)
HISTORY_SYSTEM = "ICD10CM"

#: Lab codes with numeric values.
LAB_CODES: tuple[tuple[str, float, float], ...] = (
    ("4548-4", 5.0, 12.0),
    ("2345-7", 70.0, 260.0),
    ("2160-0", 0.5, 2.5),
    ("2823-3", 3.2, 5.6),
)
LAB_SYSTEM = "LOINC"

#: Static demographic codes, used by LK009's deterministic signature.
GENDER_CODES: tuple[str, ...] = ("GENDER_F", "GENDER_M")
RACE_CODES: tuple[str, ...] = ("RACE_1", "RACE_2", "RACE_3", "RACE_4", "RACE_5")
BIRTH_DECADE_CODES: tuple[str, ...] = (
    "BORN_1940s",
    "BORN_1950s",
    "BORN_1960s",
    "BORN_1970s",
    "BORN_1980s",
    "BORN_1990s",
    "BORN_2000s",
)
#: A repeating mid-cardinality static fact, for realism: real cohorts share
#: postal codes, and a signature built only from unique facts would never
#: collide even when a dataset really does hold duplicates.
N_POSTAL_CODES = 4000

#: A per-subject **unique** static fact, standing in for the exact birth date
#: that deterministic patient matching actually keys on.
#:
#: It is derived from the subject index rather than sampled, and that is the
#: point. Gender x race x decade x postal code gives ~280,000 profiles, which
#: sounds ample until the birthday paradox is applied: at 87 subjects a chance
#: collision happens in roughly one dataset in seventy, and a chance collision
#: makes LK009 fire on data documented as clean by construction. A property
#: test over seeds found exactly that. "Clean by construction" has to hold for
#: every seed, not for most of them.
BIRTHDATE_CODE_PREFIX = "BORN_ON_"
STATIC_SYSTEM = "DEMOGRAPHIC"


@dataclass
class SyntheticDataset:
    """Generated events, labels, splits, and the task they describe.

    Attributes:
        events: The canonical event table.
        labels: ``subject_id``, ``index_date``, ``label``.
        splits: ``subject_id``, ``split``.
        task: The task spec these data satisfy.
        features: An optional feature matrix. The clean generator produces
            none; `inject.inject_lk001` attaches one, because a leak in the
            matrix is the unambiguous LK001 case.
        seed: The seed used.
        injected: Check ids whose leak has been planted, in injection order.
    """

    events: pl.DataFrame
    labels: pl.DataFrame
    splits: pl.DataFrame
    task: TaskSpec
    features: pl.DataFrame | None = None
    seed: int = 0
    injected: list[str] = field(default_factory=list)

    @property
    def n_subjects(self) -> int:
        """Distinct subjects."""
        return self.events["subject_id"].n_unique()

    def splits_mapping(self) -> dict[str, list[str]]:
        """Splits as the ``{name: [subject_id]}`` mapping ehrlint's JSON uses."""
        out: dict[str, list[str]] = {}
        for row in self.splits.iter_rows(named=True):
            out.setdefault(str(row["split"]), []).append(str(row["subject_id"]))
        return {k: sorted(v) for k, v in out.items()}

    def copy(self) -> SyntheticDataset:
        """A deep-enough copy for an injector to mutate."""
        return SyntheticDataset(
            events=self.events.clone(),
            labels=self.labels.clone(),
            splits=self.splits.clone(),
            task=self.task.model_copy(deep=True),
            features=None if self.features is None else self.features.clone(),
            seed=self.seed,
            injected=list(self.injected),
        )

    def describe(self) -> dict[str, Any]:
        """Summary for the manifest."""
        return {
            "n_subjects": self.n_subjects,
            "n_events": self.events.height,
            "n_labels": self.labels.height,
            "n_positive": int(self.labels["label"].sum() or 0),
            "splits": {k: len(v) for k, v in self.splits_mapping().items()},
            "n_feature_rows": 0 if self.features is None else self.features.height,
            "seed": self.seed,
            "injected": self.injected,
            "task": self.task.task,
        }

    def write(self, outdir: str | Path) -> dict[str, Path]:
        """Write the dataset in MEDS layout plus ehrlint's own side files."""
        import json

        from ehrlint.io.meds import write_meds_dataset

        root = Path(outdir)
        root.mkdir(parents=True, exist_ok=True)
        written = dict(
            write_meds_dataset(
                self.events,
                root,
                dataset_name="ehrlint-synthetic",
                splits=self.splits,
            )
        )

        labels_path = root / "labels.parquet"
        self.labels.write_parquet(labels_path)
        written["labels"] = labels_path

        splits_path = root / "splits.json"
        splits_path.write_text(json.dumps(self.splits_mapping(), indent=2))
        written["splits_json"] = splits_path

        task_path = root / "task.yaml"
        self.task.to_yaml(task_path)
        written["task"] = task_path

        if self.features is not None:
            features_path = root / "features.parquet"
            self.features.write_parquet(features_path)
            written["features"] = features_path

        manifest = {
            "generator": "ehrlint.synth",
            **self.describe(),
            "note": (
                "Every record here is synthetic. Subjects are sequential, codes come "
                "from a fixed vocabulary, and timestamps derive from a constant epoch. "
                "No real patient record was involved."
            ),
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        written["manifest"] = manifest_path
        return written


def default_task() -> TaskSpec:
    """The task spec the generated data satisfies.

    A two-year DKA-style task: index on an ED visit, outcome from two ICD-10
    codes, full follow-up required for negatives, temporal subject split.
    """
    return TaskSpec.from_dict(
        {
            "task": "dka_2y",
            "description": "Synthetic two-year outcome task, for exercising ehrlint.",
            "index": {"event": INDEX_CODE, "code_system": INDEX_SYSTEM},
            "outcome": {
                "codes": [{"system": OUTCOME_SYSTEM, "values": list(OUTCOME_CODES)}],
                "window": ["index_date", "index_date + 2y"],
                "proxy_codes": [{"system": PROXY_SYSTEM, "values": list(PROXY_CODES)}],
                "competing_codes": [{"system": None, "values": ["MEDS_DEATH"]}],
            },
            "horizon": "2y",
            "followup": {
                "require_full_horizon_for_negatives": True,
                "allow_death_as_censoring": True,
            },
            "splits": {
                "strategy": "subject",
                "temporal": True,
                "group_column": "encounter_id",
            },
        }
    )


def generate(
    n_subjects: int = 500,
    *,
    seed: int = 0,
    positive_rate: float = 0.2,
    horizon_days: float = 730.5,
    history_years: float = 2.0,
) -> SyntheticDataset:
    """Generate a clean synthetic dataset.

    Clean means: every predictor event predates the index date, every negative
    is observed for the full horizon, splits are disjoint and time-ordered, and
    every positive's outcome falls inside the horizon.

    Args:
        n_subjects: How many subjects.
        seed: Seed for all randomness; identical arguments give identical output.
        positive_rate: Share of subjects with the outcome.
        horizon_days: The task horizon, in days.
        history_years: How much pre-index history to generate.

    Returns:
        A :class:`SyntheticDataset`.

    Raises:
        ValueError: On arguments that cannot produce a usable dataset.
    """
    if n_subjects < 1:
        raise ValueError(f"n_subjects must be at least 1, got {n_subjects}")
    if not 0.0 <= positive_rate <= 1.0:
        raise ValueError(f"positive_rate must be in [0, 1], got {positive_rate}")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")

    rng = random.Random(seed)
    events: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    # Index dates spread over three years so a temporal split is meaningful.
    index_span_days = 365 * 3

    for i in range(n_subjects):
        subject = f"S{i + 1:06d}"
        index_offset = rng.uniform(0, index_span_days)
        index_date = EPOCH + timedelta(days=index_offset)
        is_positive = rng.random() < positive_rate

        # --- static demographics (null time), the LK009 signature ------
        for code in (
            rng.choice(GENDER_CODES),
            rng.choice(RACE_CODES),
            rng.choice(BIRTH_DECADE_CODES),
            f"POSTAL_{rng.randrange(N_POSTAL_CODES):05d}",
            f"{BIRTHDATE_CODE_PREFIX}{i + 1:08d}",
        ):
            events.append(
                {
                    "subject_id": subject,
                    "time": None,
                    "code": code,
                    "code_system": STATIC_SYSTEM,
                    "numeric_value": None,
                    "text_value": None,
                    "encounter_id": None,
                }
            )

        # --- pre-index history, in its own encounters ------------------
        n_history_encounters = rng.randint(1, 4)
        for e in range(n_history_encounters):
            encounter = f"{subject}-E{e}"
            offset = rng.uniform(1.0, history_years * 365.0)
            when = index_date - timedelta(days=offset)
            for code in rng.sample(HISTORY_CODES, k=rng.randint(1, 3)):
                events.append(
                    {
                        "subject_id": subject,
                        "time": when,
                        "code": code,
                        "code_system": HISTORY_SYSTEM,
                        "numeric_value": None,
                        "text_value": None,
                        "encounter_id": encounter,
                    }
                )
            lab, low, high = rng.choice(LAB_CODES)
            events.append(
                {
                    "subject_id": subject,
                    "time": when + timedelta(hours=1),
                    "code": lab,
                    "code_system": LAB_SYSTEM,
                    "numeric_value": round(rng.uniform(low, high), 2),
                    "text_value": None,
                    "encounter_id": encounter,
                }
            )

        # --- the index event, in its own encounter ---------------------
        index_encounter = f"{subject}-IDX"
        events.append(
            {
                "subject_id": subject,
                "time": index_date,
                "code": INDEX_CODE,
                "code_system": INDEX_SYSTEM,
                "numeric_value": None,
                "text_value": None,
                "encounter_id": index_encounter,
            }
        )

        # --- follow-up -------------------------------------------------
        if is_positive:
            # The outcome lands strictly inside the horizon.
            outcome_offset = rng.uniform(1.0, horizon_days - 1.0)
            outcome_time = index_date + timedelta(days=outcome_offset)
            events.append(
                {
                    "subject_id": subject,
                    "time": outcome_time,
                    "code": rng.choice(OUTCOME_CODES),
                    "code_system": OUTCOME_SYSTEM,
                    "numeric_value": None,
                    "text_value": None,
                    "encounter_id": f"{subject}-OUT",
                }
            )
            last_time = outcome_time
        else:
            # A negative is observed for the whole horizon, so LK004 is clean.
            last_time = index_date + timedelta(days=horizon_days + rng.uniform(1.0, 60.0))
            n_followup = rng.randint(1, 3)
            for f in range(n_followup):
                when = index_date + timedelta(days=rng.uniform(1.0, horizon_days))
                events.append(
                    {
                        "subject_id": subject,
                        "time": when,
                        "code": rng.choice(HISTORY_CODES),
                        "code_system": HISTORY_SYSTEM,
                        "numeric_value": None,
                        "text_value": None,
                        "encounter_id": f"{subject}-F{f}",
                    }
                )
            # A final visit at the end of follow-up, so last_event_time clears
            # the horizon and the negative is a genuine negative.
            events.append(
                {
                    "subject_id": subject,
                    "time": last_time,
                    "code": INDEX_CODE,
                    "code_system": INDEX_SYSTEM,
                    "numeric_value": None,
                    "text_value": None,
                    "encounter_id": f"{subject}-END",
                }
            )

        labels.append({"subject_id": subject, "index_date": index_date, "label": is_positive})
        split_rows.append({"subject_id": subject, "index_offset": index_offset})

    # --- temporal, disjoint splits by subject --------------------------
    split_rows.sort(key=lambda r: r["index_offset"])
    n = len(split_rows)
    n_train, n_valid = int(n * 0.6), int(n * 0.2)
    assignments: list[dict[str, Any]] = []
    for i, row in enumerate(split_rows):
        if i < n_train:
            name = TRAIN
        elif i < n_train + n_valid:
            name = VALID
        else:
            name = TEST
        assignments.append({"subject_id": row["subject_id"], "split": name})

    events_df = coerce_events(pl.DataFrame(events), source="synth")
    labels_df = pl.DataFrame(labels).with_columns(
        pl.col("subject_id").cast(pl.String),
        pl.col("index_date").cast(pl.Datetime("us")),
        pl.col("label").cast(pl.Boolean),
    )
    splits_df = pl.DataFrame(assignments).with_columns(
        pl.col("subject_id").cast(pl.String), pl.col("split").cast(pl.String)
    )

    return SyntheticDataset(
        events=events_df,
        labels=labels_df,
        splits=splits_df,
        task=default_task(),
        seed=seed,
    )
