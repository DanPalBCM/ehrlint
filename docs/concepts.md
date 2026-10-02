# Concepts

Four things: the event table, the task spec, the findings, and the context that
joins them.

## The canonical event table

Every input format becomes one table:

| column | type | nullable | meaning |
| --- | --- | --- | --- |
| `subject_id` | string | no | the patient |
| `time` | datetime | **yes** | when; null means a static fact |
| `code` | string | no | the clinical code |
| `code_system` | string | yes | the vocabulary |
| `numeric_value` | float | yes | a lab value, a quantity |
| `text_value` | string | yes | a free-text value |
| `encounter_id` | string | yes | the visit |

Three things about this schema are load-bearing.

**`subject_id` is always a string.** An int64 subject id in the events and a
string one in the labels join to nothing — and a join that returns nothing
makes every check report clean. Normalizing at ingestion is the cheapest place
to prevent a false-clean audit, so the coercion is unconditional and tested.

**A null `time` means a static fact, not an event at the epoch.** Demographics
and birth records carry no timestamp. Treating null as the epoch would make
every static fact look like a pre-index event, which is both wrong and the kind
of wrong that produces confident false findings. Static events are excluded
from every time-ordering check.

**`code_system` and `encounter_id` may be absent, and that has consequences.**
MEDS has neither column; it namespaces the vocabulary inside the code string as
`ICD10CM//E11.9`, which ehrlint splits on the first separator. A code with no
separator keeps a **null** system rather than a guessed one. Where no event
carries an encounter id, LK002 and LK008 skip and say so.

## The task spec

A task spec says what the prediction problem is: when the clock starts, what
counts as the outcome, and how long you look. See [task specs](task-spec.md).

The spec is also what makes checks possible or not, which is why
`ehrlint validate-task` reports the checks a given spec weakens. ehrlint will
not infer a missing piece — a guessed proxy code or an assumed horizon would
produce findings nobody could verify.

## Findings

A finding is one check's answer about one dataset:

```python
Finding(
    check_id="LK002",
    name="same_encounter",
    severity=Severity.ERROR,
    message="5 subject(s) have an outcome code recorded in the same encounter ...",
    n_subjects=5,
    n_total_subjects=120,
    n_rows=5,
    examples=[{...}, ...],      # the rows that triggered it
    example_columns=[...],
    query="SELECT ... FROM events e JOIN ...",
    subjects=["S000012", ...],
    detail={...},
)
```

A few properties worth knowing:

**`fired`** is true when the finding reports a problem: a non-zero subject count
at `error` or `warning`. A check that ran and found nothing returns a finding
with `fired == False`, which appears in the report as a pass.

**`subject_rate`** is `None`, not `0.0`, when the denominator is zero. Zero
would read as "no subjects affected", which is a different claim from "there
were no subjects to affect".

**Info findings never fire.** `info` is for descriptive statistics — "120 of 120
subjects have events after their index date" is true of every dataset with
follow-up. They are reported as observations and cannot fail a run, which is
why `--fail-on info` is refused rather than silently treated as
`--fail-on warning`.

**Findings are deterministic.** Example rows are sorted on a *total* key before
truncation, so the same data produces the same evidence regardless of shard
order. This matters more than it sounds: an earlier version sorted by subject
and time only, and a check whose rows tied on both keys produced different
example rows between runs.

### Skips

A check that cannot run returns a `SkippedCheck` instead:

```python
SkippedCheck(
    check_id="LK002",
    name="same_encounter",
    reason="no event carries an encounter_id, so the index encounter cannot be "
           "identified",
)
```

Skips are not findings and are counted separately everywhere. A check that
crashes is also recorded as a skip — with the exception type in the reason, and
labelled a bug in ehrlint rather than a finding about your data — so one broken
check cannot take the other nine down with it.

## The audit context

`AuditContext` holds the resolved inputs and the few derived views every check
needs: the index date per subject, the events joined to it, the split
assignment, and the code-set matcher.

```python
ctx = el.AuditContext.from_meds("data/", task_spec="task.yaml")
ctx = el.AuditContext.from_omop("omop/", task_spec="task.yaml")
ctx = el.AuditContext.from_frames(events_df, task_spec=spec)
```

Index dates resolve in a fixed order: a `prediction_time` in the labels table
wins over the index event code. An explicit label time is the task author's own
statement about when the clock starts, and re-deriving it from an event code
would silently audit a different task than the one that was trained.

`ctx.all_warnings()` collects everything worth saying about the inputs before
any check runs — null subject ids, absent encounter ids, splits naming subjects
the event table does not contain, and the checks the task spec weakens. These
appear at the top of the report, above the findings, for the same reason skips
do.
