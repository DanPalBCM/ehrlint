# Writing a custom check

Leakage patterns are partly local. A check that matters in your health system
may be meaningless in another, so custom checks register into the same battery
as the built-in ten and appear in the same report.

## The short version

```python
import ehrlint as el
import polars as pl


@el.register_check(
    id="LK101",
    name="pre_index_discharge_summary",
    severity="error",
    rationale=(
        "Our ETL stamps discharge summaries with the admission date, so a summary "
        "written after the index visit looks pre-index. Any summary within an hour "
        "of the index date is from the index encounter."
    ),
)
def pre_index_discharge_summary(ctx: el.AuditContext) -> list[el.Finding]:
    """Discharge summaries mis-stamped to the admission date."""
    joined = ctx.events_with_index()
    offending = joined.filter(
        (pl.col("code") == "DISCHARGE_SUMMARY")
        & ((pl.col("index_date") - pl.col("time")).dt.total_seconds().abs() < 3600)
    )
    return [
        el.make_finding(
            "LK101",
            name="pre_index_discharge_summary",
            severity=el.Severity.ERROR,
            message=(
                f"{offending['subject_id'].n_unique()} subject(s) have a discharge "
                "summary within an hour of the index date, which our ETL stamps to "
                "the admission date."
                if offending.height
                else "No discharge summary is stamped at the index date."
            ),
            offending=offending,
            n_total_subjects=ctx.n_subjects,
            query=(
                "SELECT * FROM events e JOIN index_dates i USING (subject_id)\n"
                "WHERE e.code = 'DISCHARGE_SUMMARY'\n"
                "  AND ABS(DATEDIFF('second', i.index_date, e.time)) < 3600"
            ),
            example_columns=["subject_id", "time", "code", "index_date", "encounter_id"],
        )
    ]
```

Then it runs with everything else:

```bash
ehrlint audit --data meds/ --task task.yaml   # LK101 included
```

## Use `make_finding`

It is the single constructor every built-in check uses, and it does four things
you would otherwise have to remember:

- Counts come from the offending frame, not from anything you pass in — so a
  count cannot disagree with the rows behind it.
- An **empty frame means the check did not fire**, and you get a zero-count
  finding that appears in the report as a pass. You do not need a separate
  branch for the clean case beyond the message.
- Example rows are sorted on a total key before truncation, so the evidence is
  the same rows regardless of input order.
- The subject list is derived from the frame rather than passed in.

Its `subject_column` argument matters when your evidence is not one row per
subject. If a finding is about pairs — two subjects sharing an encounter, two
ids for one person — emit **one row per subject**, not one per pair. Otherwise
`n_subjects` understates by half and `findings.jsonl` names only one side.
That was a real bug in LK008 and LK009.

## Report a skip rather than a pass

If your check's inputs are missing, say so:

```python
@el.register_check(id="LK102", name="my_check")
def my_check(ctx: el.AuditContext) -> el.CheckResult:
    if ctx.features is None:
        return el.SkippedCheck(
            check_id="LK102",
            name="my_check",
            reason="no feature matrix was supplied, so there is nothing to inspect",
        )
    ...
```

A skip is counted separately from a pass, and the report prints it above the
findings. Returning an empty list instead would make your check look like it
ran and found nothing, which is the one thing this project is built to prevent.

## Or subclass `BaseCheck`

The decorator is a convenience over the protocol. For a check with state or
helpers, subclass:

```python
class MyCheck(el.BaseCheck):
    id = "LK103"
    name = "my_check"
    description = "One line, shown in `ehrlint list-checks`."
    rationale = "Why a reader should care, printed next to every finding."
    severity_default = el.Severity.WARNING

    def run(self, ctx: el.AuditContext) -> el.CheckResult:
        if not ctx.has_encounters:
            return self.skip("no event carries an encounter_id")
        ...
        return self.clean(ctx, "Nothing found.", query="SELECT 1")


el.register(MyCheck())
```

`self.skip(reason)` and `self.clean(ctx, message, query)` are the two
conveniences: a skip with this check's identity attached, and a zero-count
finding for the genuine-pass case.

## What the context gives you

```python
ctx.events              # the canonical table
ctx.timed()             # events with a timestamp (static facts excluded)
ctx.index_dates()       # subject_id, index_date — cached
ctx.events_with_index() # events joined to the index date
ctx.subject_split()     # subject_id, split
ctx.match_code_sets(frame, code_sets)   # system-aware code matching
ctx.n_subjects          # the denominator
ctx.task                # the TaskSpec
ctx.features            # the FeatureMatrix, or None
ctx.labels              # the labels frame, or None
ctx.splits              # the Splits, or None
ctx.has_encounters      # whether any event carries one
ctx.data_hash           # order-invariant content hash
```

Use `ctx.match_code_sets` rather than a bare `is_in` on `code`. It matches on
system *and* value where a system is declared, so an ICD code cannot match a
LOINC set that happens to share a string.

Use `ctx.timed()` whenever you compare times. Static facts carry a null
timestamp, and treating null as the epoch makes every demographic look like a
pre-index event.

## Testing your check

Follow the pattern the built-ins use: write an injector that plants exactly
your leak, assert nothing fires on clean data, then assert your check fires on
the injected data **and names the subjects the injector touched**.

```python
from ehrlint.synth.generator import generate


def test_my_check_fires_on_its_own_leak():
    dataset = generate(120, seed=0)
    planted = ["S000003", "S000007"]

    leaky = dataset.copy()
    # ... plant the leak into `leaky.events` for `planted` ...

    ctx = el.AuditContext.from_frames(
        leaky.events, task_spec=leaky.task, labels=leaky.labels
    )
    finding = el.run_audit(ctx, ["LK101"]).finding("LK101")
    assert finding is not None and finding.fired
    assert set(planted) <= set(finding.subjects)
```

The second assertion is the one that matters. A check that fires on the wrong
subjects is as broken as one that stays silent, and only ground truth catches
it. See [synthetic data](synthetic-data.md).

## Severity

`error` means the task is invalid as defined. `warning` means a human needs to
look. `info` means a descriptive statistic — and info findings never fire, so
they cannot fail a run.

Choose `error` only when the finding is unambiguous. LK001's event-level branch
is `info` precisely because post-index events are normal, and an error that is
always present is an error everyone learns to ignore.
