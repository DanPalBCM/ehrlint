"""The task specification: index date, outcome, horizon, follow-up.

This is the object every check reasons against. A leak is only definable
relative to a task: "post-index features" needs an index date, and
"immortal-time bias" needs a follow-up rule.

The spec is deliberately strict. A task definition that is ambiguous about its
own horizon cannot be audited meaningfully, so ehrlint refuses it at load time
rather than producing findings whose meaning depends on a guess.
"""

from __future__ import annotations

import re
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ehrlint.exceptions import TaskSpecError

#: `30d`, `12w`, `6m`, `2y`. Months and years are calendar-approximate here --
#: see `Duration.to_timedelta` for why that is stated rather than hidden.
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([dwmy])\s*$", re.IGNORECASE)

#: Average days per unit. Months and years are approximations.
_DAYS_PER_UNIT = {"d": 1.0, "w": 7.0, "m": 30.4375, "y": 365.25}


class SplitStrategy(StrEnum):
    """How the dataset was split."""

    SUBJECT = "subject"
    ENCOUNTER = "encounter"
    TEMPORAL = "temporal"
    RANDOM = "random"


class Duration(BaseModel):
    """A horizon or window length, e.g. ``2y``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n: float = Field(gt=0)
    unit: str = Field(pattern="^[dwmy]$")

    @classmethod
    def parse(cls, value: str | Duration | None) -> Duration | None:
        """Parse ``"2y"`` into ``Duration(n=2, unit="y")``.

        Raises:
            TaskSpecError: On anything unparseable, with the accepted forms
                named. A horizon is load-bearing, so a typo must not become a
                silently different number.
        """
        if value is None:
            return None
        if isinstance(value, Duration):
            return value
        match = _DURATION_RE.match(str(value))
        if not match:
            raise TaskSpecError(
                f"cannot parse duration {value!r}; expected forms like '30d', '12w', '6m', '2y'"
            )
        n = float(match.group(1))
        if n <= 0:
            # Caught here rather than left to the field constraint, which would
            # surface a raw pydantic error from a user-facing YAML file.
            raise TaskSpecError(
                f"duration {value!r} is not positive. A zero-length horizon has no "
                "window to audit; expected forms like '30d', '12w', '6m', '2y'"
            )
        return cls(n=n, unit=match.group(2).lower())

    def __str__(self) -> str:
        n = int(self.n) if float(self.n).is_integer() else self.n
        return f"{n}{self.unit}"

    @property
    def days(self) -> float:
        """Length in days.

        Months and years use 30.4375 and 365.25 days. That is an
        approximation, and it is the right one for a *horizon*: "within two
        years of the index date" is a study design choice with a tolerance of
        days, not a calendar-exact boundary. Where exactness matters, express
        the horizon in days.
        """
        return self.n * _DAYS_PER_UNIT[self.unit]

    def to_timedelta(self) -> timedelta:
        """Length as a ``timedelta``."""
        return timedelta(days=self.days)


class CodeSet(BaseModel):
    """A set of codes in one vocabulary."""

    model_config = ConfigDict(extra="forbid")

    system: str | None = Field(
        default=None,
        description="Vocabulary name. Null matches the bare code in any system, which "
        "can cross-match -- ehrlint warns when it has to do that.",
    )
    values: list[str] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def _strip(cls, v: list[str]) -> list[str]:
        out = [s.strip() for s in v if s and s.strip()]
        if not out:
            raise ValueError("a code set needs at least one non-blank value")
        return out

    def matches_any(self) -> set[str]:
        """The code values, as a set."""
        return set(self.values)


class IndexDef(BaseModel):
    """How each subject's index date is determined."""

    model_config = ConfigDict(extra="forbid")

    event: str | None = Field(
        default=None,
        description="Code marking the index event. Null when index dates are supplied "
        "in a labels table instead.",
    )
    code_system: str | None = None
    occurrence: str = Field(
        default="first",
        pattern="^(first|last)$",
        description="Which occurrence of the index event to use when a subject has "
        "several. Defaults to 'first'.",
    )

    @property
    def from_labels(self) -> bool:
        """Whether index dates come from a labels table rather than an event code."""
        return self.event is None


class OutcomeDef(BaseModel):
    """How the outcome is ascertained."""

    model_config = ConfigDict(extra="forbid")

    codes: list[CodeSet] = Field(default_factory=list)
    window: list[str] | None = Field(
        default=None,
        description="Two expressions, e.g. ['index_date', 'index_date + 2y']. Used to "
        "detect a window that extends past the declared horizon.",
    )
    proxy_codes: list[CodeSet] = Field(
        default_factory=list,
        description="Codes that imply the outcome without being it -- a treatment only "
        "given to cases, say. Pre-index occurrences are flagged by LK003.",
    )
    competing_codes: list[CodeSet] = Field(
        default_factory=list,
        description="Competing events, typically death, that end follow-up without the "
        "outcome. Used by LK004.",
    )

    def all_codes(self) -> set[str]:
        """Every outcome code value."""
        return {v for cs in self.codes for v in cs.values}

    def all_proxies(self) -> set[str]:
        """Every proxy code value."""
        return {v for cs in self.proxy_codes for v in cs.values}

    def all_competing(self) -> set[str]:
        """Every competing-event code value."""
        return {v for cs in self.competing_codes for v in cs.values}


class FollowUp(BaseModel):
    """Follow-up requirements, which is where immortal-time bias lives."""

    model_config = ConfigDict(extra="forbid")

    require_full_horizon_for_negatives: bool = Field(
        default=True,
        description="A subject labelled negative must have been observed for the whole "
        "horizon. Without this, a subject who left the data after one month counts as "
        "a two-year negative -- the classic immortal-time error.",
    )
    min_days: float | None = Field(
        default=None,
        ge=0,
        description="Minimum observed follow-up for any subject, in days.",
    )
    allow_death_as_censoring: bool = Field(
        default=True,
        description="Treat death as censoring rather than as a failed negative. When "
        "false, a subject who dies mid-horizon without the outcome is flagged.",
    )


class SplitsSpec(BaseModel):
    """How the splits were constructed, so the split checks know what to expect."""

    model_config = ConfigDict(extra="forbid")

    strategy: SplitStrategy = SplitStrategy.SUBJECT
    temporal: bool = Field(
        default=False,
        description="Whether the split is meant to respect time. When true, LK006 "
        "becomes an error rather than a warning.",
    )
    group_column: str | None = Field(
        default=None,
        description="Column defining a group that must not straddle splits, e.g. "
        "encounter_id or provider_id. Used by LK008.",
    )


class TaskSpec(BaseModel):
    """A complete task definition."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1)
    index: IndexDef = Field(default_factory=IndexDef)
    outcome: OutcomeDef = Field(default_factory=OutcomeDef)
    horizon: Duration | None = None
    followup: FollowUp = Field(default_factory=FollowUp)
    exclusion: list[CodeSet] = Field(default_factory=list)
    splits: SplitsSpec = Field(default_factory=SplitsSpec)
    description: str | None = None

    @field_validator("horizon", mode="before")
    @classmethod
    def _coerce_horizon(cls, v: object) -> object:
        return Duration.parse(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check_consistency(self) -> TaskSpec:
        """Reject combinations that cannot be audited meaningfully."""
        if self.followup.require_full_horizon_for_negatives and self.horizon is None:
            raise TaskSpecError(
                "followup.require_full_horizon_for_negatives is true but no horizon is "
                "declared, so there is no window to require. Set `horizon` or turn the "
                "requirement off."
            )
        if self.outcome.window is not None and len(self.outcome.window) != 2:
            raise TaskSpecError(
                f"outcome.window needs exactly two expressions (start and end), got "
                f"{len(self.outcome.window)}"
            )
        return self

    # -- loading --------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskSpec:
        """Load a task spec from YAML."""
        source = Path(path)
        if not source.exists():
            raise TaskSpecError(f"task spec not found: {source}")
        try:
            raw = yaml.safe_load(source.read_text()) or {}
        except yaml.YAMLError as exc:
            raise TaskSpecError(f"invalid YAML in {source}: {exc}") from exc
        if not isinstance(raw, dict):
            raise TaskSpecError(
                f"{source}: the task spec root must be a mapping, got {type(raw).__name__}"
            )
        return cls.from_dict(raw, source=str(source))

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, source: str = "<dict>") -> TaskSpec:
        """Build a task spec from a plain dict, accepting documented shorthands."""
        data = dict(raw)

        # §6.1 writes outcome.codes as a list of {system, values} mappings, and
        # allows a bare list of code strings as a shorthand.
        outcome = data.get("outcome")
        if isinstance(outcome, dict):
            outcome = dict(outcome)
            for key in ("codes", "proxy_codes", "competing_codes"):
                outcome[key] = _normalize_code_sets(outcome.get(key))
            data["outcome"] = outcome
        data["exclusion"] = _normalize_code_sets(data.get("exclusion"))

        try:
            return cls.model_validate(data)
        except TaskSpecError:
            raise
        except Exception as exc:
            raise TaskSpecError(f"{source}: invalid task spec: {_detail(exc)}") from exc

    def to_yaml(self, path: str | Path) -> Path:
        """Write the task spec to YAML, round-trippable through `from_yaml`."""
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(self.model_dump_json(exclude_none=True))
        # Durations round-trip as their shorthand rather than as a nested dict.
        if self.horizon is not None:
            payload["horizon"] = str(self.horizon)
        target.write_text(yaml.safe_dump(payload, sort_keys=True))
        return target

    def describe(self) -> dict[str, Any]:
        """Short summary for the report header."""
        return {
            "task": self.task,
            "index_event": self.index.event or "(from labels)",
            "horizon": str(self.horizon) if self.horizon else None,
            "n_outcome_codes": len(self.outcome.all_codes()),
            "n_proxy_codes": len(self.outcome.all_proxies()),
            "n_competing_codes": len(self.outcome.all_competing()),
            "split_strategy": self.splits.strategy.value,
            "temporal_split": self.splits.temporal,
            "require_full_horizon_for_negatives": (
                self.followup.require_full_horizon_for_negatives
            ),
        }

    def warnings(self) -> list[str]:
        """Things about this spec that will limit what can be audited.

        Returned rather than raised: an under-specified task is still worth
        auditing for what it does declare, but the report should say which
        checks were weakened.
        """
        out: list[str] = []
        if not self.outcome.codes:
            out.append(
                "no outcome codes are declared, so the label and horizon checks "
                "(LK007, LK010) cannot verify the outcome and will skip"
            )
        if not self.outcome.proxy_codes:
            out.append(
                "no proxy codes are declared, so LK003 has nothing to look for. Proxies "
                "are task-specific and ehrlint cannot infer them -- list the treatments "
                "or codes that imply your outcome"
            )
        if self.horizon is None:
            out.append("no horizon is declared, so LK010 will skip")
        if self.splits.group_column is None:
            out.append(
                "no splits.group_column is declared, so LK008 falls back to encounter_id "
                "if present and skips otherwise"
            )
        if any(cs.system is None for cs in self.outcome.codes):
            out.append(
                "some outcome code sets declare no system, so matching falls back to the "
                "bare code and may cross-match between vocabularies"
            )
        return out


def _normalize_code_sets(value: Any) -> list[dict[str, Any]]:
    """Accept a list of code sets, or the bare-string-list shorthand."""
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise TaskSpecError(f"expected a list of code sets, got {type(value).__name__}")

    out: list[dict[str, Any]] = []
    bare: list[str] = []
    for item in value:
        if isinstance(item, str):
            bare.append(item)
        elif isinstance(item, dict):
            out.append(item)
        else:
            raise TaskSpecError(
                f"a code set must be a mapping or a code string, got {type(item).__name__}"
            )
    if bare:
        out.append({"system": None, "values": bare})
    return out


def _detail(exc: Exception) -> str:
    """Render a pydantic error compactly."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            return "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in errors()
            )
        except Exception:  # pragma: no cover
            pass
    return str(exc)
