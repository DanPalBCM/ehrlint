"""Train/validation/test splits.

Two input shapes, one model:

* ehrlint's own JSON — ``{"train": ["s1", ...], "valid": [...], "test": [...]}``
  (SPEC §6.2);
* a MEDS ``subject_splits.parquet`` with ``subject_id`` and ``split``, whose
  split names are **`train`, `tuning`, `held_out`** rather than
  train/valid/test.

Those two vocabularies are reconciled here, with an alias map, because
`tuning` and `valid` mean the same thing and a user should not have to rename
their splits to run an audit. See ``docs/decisions.md`` D-003.

Subject ids are normalized to **string** for the same reason as in the event
table: an int/string mismatch between a splits file and an event table
produces an empty intersection, which looks exactly like "no overlap found".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ehrlint.exceptions import SplitsError

#: Canonical split names ehrlint reports under.
TRAIN = "train"
VALID = "valid"
TEST = "test"

#: MEDS split names, mapped to the canonical ones. MEDS calls them `tuning`
#: and `held_out`; nothing is lost by treating them as valid and test.
SPLIT_ALIASES: dict[str, str] = {
    "train": TRAIN,
    "training": TRAIN,
    "tuning": VALID,
    "valid": VALID,
    "validation": VALID,
    "dev": VALID,
    "held_out": TEST,
    "heldout": TEST,
    "test": TEST,
    "testing": TEST,
    "eval": TEST,
}

#: The order splits are reported in.
SPLIT_ORDER: tuple[str, ...] = (TRAIN, VALID, TEST)


def canonical_split_name(name: str) -> str:
    """Map a split name to its canonical form.

    An unrecognized name is kept as-is rather than guessed at: a project may
    legitimately have a fourth split, and renaming it would hide that.
    """
    return SPLIT_ALIASES.get(str(name).strip().lower(), str(name).strip())


class Splits(BaseModel):
    """Subject ids per split."""

    model_config = ConfigDict(extra="forbid")

    assignments: dict[str, list[str]] = Field(default_factory=dict)
    source: str | None = None
    original_names: dict[str, str] = Field(
        default_factory=dict,
        description="Canonical name -> the name in the source file, so the report can "
        "echo the user's own vocabulary.",
    )

    @field_validator("assignments")
    @classmethod
    def _normalize(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        return {str(k): sorted({str(s) for s in ids if s is not None}) for k, ids in v.items()}

    def __len__(self) -> int:
        return len(self.assignments)

    def __bool__(self) -> bool:
        return bool(self.assignments)

    @property
    def names(self) -> list[str]:
        """Split names, canonical ones first in a stable order."""
        known = [n for n in SPLIT_ORDER if n in self.assignments]
        extra = sorted(set(self.assignments) - set(SPLIT_ORDER))
        return known + extra

    def get(self, name: str) -> list[str]:
        """Subject ids in one split, or an empty list."""
        return self.assignments.get(name, [])

    def all_subjects(self) -> set[str]:
        """Every subject id mentioned in any split."""
        return {s for ids in self.assignments.values() for s in ids}

    def sizes(self) -> dict[str, int]:
        """Subject count per split."""
        return {name: len(self.assignments[name]) for name in self.names}

    def overlaps(self) -> dict[tuple[str, str], list[str]]:
        """Subject ids appearing in more than one split.

        Returns:
            ``{(split_a, split_b): [subject_id, ...]}`` for every pair that
            intersects, sorted. Empty when the splits are disjoint.
        """
        out: dict[tuple[str, str], list[str]] = {}
        names = self.names
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                shared = sorted(set(self.assignments[a]) & set(self.assignments[b]))
                if shared:
                    out[(a, b)] = shared
        return out

    def split_of(self) -> dict[str, list[str]]:
        """Subject id -> the splits it appears in.

        A subject with more than one entry is an LK005 finding.
        """
        out: dict[str, list[str]] = {}
        for name in self.names:
            for subject in self.assignments[name]:
                out.setdefault(subject, []).append(name)
        return out

    def as_frame(self) -> pl.DataFrame:
        """Long-format frame with ``subject_id`` and ``split``."""
        rows = [
            {"subject_id": subject, "split": name}
            for name in self.names
            for subject in self.assignments[name]
        ]
        if not rows:
            return pl.DataFrame(schema={"subject_id": pl.String, "split": pl.String})
        return pl.DataFrame(rows)

    def describe(self) -> dict[str, Any]:
        """Summary for the report header."""
        return {
            "source": self.source,
            "names": self.names,
            "sizes": self.sizes(),
            "n_subjects": len(self.all_subjects()),
            "n_overlapping_pairs": len(self.overlaps()),
            "original_names": self.original_names,
        }

    # -- loading --------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> Splits:
        """Load ehrlint's own splits JSON (SPEC §6.2)."""
        source = Path(path)
        if not source.exists():
            raise SplitsError(f"splits file not found: {source}")
        try:
            raw = json.loads(source.read_text())
        except json.JSONDecodeError as exc:
            raise SplitsError(f"{source}: malformed JSON: {exc.msg}") from exc
        if not isinstance(raw, dict):
            raise SplitsError(
                f"{source}: expected an object mapping split name to subject id list, "
                f"got {type(raw).__name__}"
            )
        return cls.from_mapping(raw, source=str(source))

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, source: str | None = None) -> Splits:
        """Build from a plain mapping of split name to subject ids."""
        assignments: dict[str, list[str]] = {}
        original: dict[str, str] = {}
        for name, ids in raw.items():
            if not isinstance(ids, (list, tuple, set)):
                raise SplitsError(
                    f"split {name!r} must map to a list of subject ids, got {type(ids).__name__}"
                )
            canonical = canonical_split_name(name)
            original[canonical] = str(name)
            assignments.setdefault(canonical, [])
            assignments[canonical].extend(str(s) for s in ids if s is not None)
        if not assignments:
            raise SplitsError(f"{source or 'splits'} defines no splits")
        return cls(assignments=assignments, source=source, original_names=original)

    @classmethod
    def from_meds(cls, path: str | Path, *, id_map: dict[str, str] | None = None) -> Splits:
        """Load a MEDS ``subject_splits.parquet``.

        Expects ``subject_id`` and ``split``. MEDS split names (`train`,
        `tuning`, `held_out`) are mapped to ehrlint's canonical ones.

        Args:
            path: The splits parquet.
            id_map: Surrogate-to-original subject id mapping, when the dataset
                carries one. The splits file has to use the same int64 ids as
                the event shards; without the same mapping applied here, the
                splits would name subjects the event table does not contain and
                LK005 and LK008 would both find nothing to compare.
        """
        source = Path(path)
        if not source.exists():
            raise SplitsError(f"MEDS subject splits not found: {source}")
        try:
            df = pl.read_parquet(source)
        except Exception as exc:
            raise SplitsError(f"{source}: could not read parquet: {exc}") from exc

        missing = [c for c in ("subject_id", "split") if c not in df.columns]
        if missing:
            raise SplitsError(
                f"{source}: MEDS subject splits need column(s) {missing}; found {df.columns}"
            )

        df = df.with_columns(
            pl.col("subject_id").cast(pl.String, strict=False),
            pl.col("split").cast(pl.String, strict=False),
        )
        raw: dict[str, list[str]] = {}
        for row in df.drop_nulls("split").iter_rows(named=True):
            subject = str(row["subject_id"])
            if id_map is not None:
                if subject not in id_map:
                    raise SplitsError(
                        f"{source}: subject id {subject!r} is not in the dataset's "
                        "subject id map. Mixing mapped and unmapped ids would make "
                        "the split comparisons silently vacuous"
                    )
                subject = id_map[subject]
            raw.setdefault(str(row["split"]), []).append(subject)
        return cls.from_mapping(raw, source=str(source))

    @classmethod
    def from_frame(cls, df: pl.DataFrame, *, source: str | None = None) -> Splits:
        """Build from a long-format frame with ``subject_id`` and ``split``."""
        missing = [c for c in ("subject_id", "split") if c not in df.columns]
        if missing:
            raise SplitsError(f"splits frame needs column(s) {missing}; found {df.columns}")
        raw: dict[str, list[str]] = {}
        for row in df.drop_nulls("split").iter_rows(named=True):
            raw.setdefault(str(row["split"]), []).append(str(row["subject_id"]))
        return cls.from_mapping(raw, source=source)

    def write_json(self, path: str | Path) -> Path:
        """Write in ehrlint's splits JSON format."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({n: self.assignments[n] for n in self.names}, indent=2))
        return target

    def warnings(self, known_subjects: set[str] | None = None) -> list[str]:
        """Problems worth reporting that are not themselves findings."""
        out: list[str] = []
        empty = [n for n in self.names if not self.assignments[n]]
        if empty:
            out.append(f"split(s) {empty} contain no subjects")
        if len(self.names) < 2:
            out.append(
                f"only {len(self.names)} split is defined, so the split checks "
                "(LK005, LK006, LK008) have nothing to compare"
            )
        if known_subjects is not None:
            assigned = self.all_subjects()
            unknown = sorted(assigned - known_subjects)
            if unknown:
                out.append(
                    f"{len(unknown)} subject(s) in the splits do not appear in the event "
                    f"data, e.g. {unknown[:5]}. This usually means a subject-id type or "
                    "format mismatch between the two files"
                )
            unassigned = sorted(known_subjects - assigned)
            if unassigned:
                out.append(
                    f"{len(unassigned)} subject(s) in the event data are in no split, "
                    f"e.g. {unassigned[:5]}"
                )
        return out
