# Task specs

A task spec is a YAML file describing the prediction problem: when the clock
starts, what counts as the outcome, how long you look, and how the data was
split. It is what turns a pile of events into something auditable.

## A complete example

```yaml
task: dka_2y
description: Two-year DKA risk after an emergency visit.

index:
  event: ed_visit
  code_system: VISIT

outcome:
  codes:
    - system: ICD10CM
      values: [E10.10, E11.65]
  window: [index_date, index_date + 2y]
  proxy_codes:
    - system: PROCEDURE
      values: [insulin_drip]
  competing_codes:
    - system: null
      values: [MEDS_DEATH]

horizon: 2y

followup:
  require_full_horizon_for_negatives: true
  allow_death_as_censoring: true

splits:
  strategy: subject
  temporal: true
  group_column: encounter_id
```

Validate it before you have data:

```bash
ehrlint validate-task task.yaml
```

Unknown keys are **rejected**, not ignored. A typo in a spec key would
otherwise silently change the task being audited — `horzion: 2y` would leave
the horizon unset and take LK010 with it.

## Fields

### `task`, `description`

A short identifier, printed in every report, and an optional sentence.

### `index`

How each subject's index date is determined.

```yaml
index:
  event: ed_visit        # the code marking the index event
  code_system: VISIT     # null matches the bare code in any vocabulary
  occurrence: first      # first | last
```

If your labels table carries a `prediction_time`, that wins. An explicit label
time is the task author's own statement about when the clock starts; deriving
it again from an event code would audit a different task than the one trained.

### `outcome`

```yaml
outcome:
  codes:
    - system: ICD10CM
      values: [E10.10, E11.65]
  window: [index_date, index_date + 2y]
  proxy_codes: [...]
  competing_codes: [...]
```

`codes` is one or more code sets. A code set with `system: null` matches the
bare code in any vocabulary, which can cross-match between vocabularies —
ehrlint warns when it has to do that rather than doing it quietly.

`window` is the interval in which an occurrence counts as the outcome. Two
expressions exactly; one is an error, because a half-specified window is
ambiguous rather than permissive.

**The lower bound is open**, whatever the brackets suggest. An occurrence
stamped at exactly the index date is not predictable from the index date, so it
does not count as an outcome — it is reported by LK002 instead. If your own
label query used `>=`, LK007 will flag the difference. See
[label auditing](labels.md#the-lower-bound-is-open).

`proxy_codes` are treatments or codes that *imply* the outcome. LK003 looks for
these before the index date. They are yours to declare: ehrlint cannot know
that an insulin drip implies DKA in your cohort, and guessing would produce
findings nobody could verify. Omitting them makes LK003 skip.

`competing_codes` are events that end follow-up without being the outcome —
death, most often. LK004 uses them so that a patient who died before the
horizon is not counted as an unobserved negative.

### `horizon`

```yaml
horizon: 2y
```

Accepted forms: `30d`, `12w`, `6m`, `2y`. Months are 30.4375 days and years are
365.25 — an approximation, and the right one for a horizon: "within two years
of the index date" is a design choice with a tolerance of days, not a
calendar-exact boundary. Where exactness matters, write the horizon in days.

Anything unparseable is an error that names the accepted forms, including `0d`:
a zero-length horizon has no window to audit.

Omitting the horizon makes LK010 skip.

### `followup`

```yaml
followup:
  require_full_horizon_for_negatives: true
  allow_death_as_censoring: true
  min_days: 30
```

`require_full_horizon_for_negatives` is what LK004 checks. With it true, a
negative must be observed for the whole horizon; without it, LK004 skips,
because there is no requirement to violate.

Setting it true with no horizon declared is rejected — there would be no window
to require.

### `splits`

```yaml
splits:
  strategy: subject      # subject | encounter | time
  temporal: true
  group_column: encounter_id
```

`temporal: true` is a claim, and LK006 checks it. When it is declared and does
not hold, LK006 is an `error`; when it is not declared, the same pattern is a
`warning`. A declared property that fails is a broken claim; an undeclared one
is a design question.

`group_column` is what LK008 groups by. Omitting it falls back to
`encounter_id` if the events carry one, and skips otherwise.

### `exclusion`

A list of exclusion criteria, recorded in the report for provenance. ehrlint
does not apply them — v1 audits a finished dataset rather than building one.

## What a spec weakens

`ehrlint validate-task` lists the checks a given spec cannot support:

```text
valid  dka_1y

This spec weakens some checks
  ! no proxy codes are declared, so LK003 has nothing to look for. Proxies are
    task-specific and ehrlint cannot infer them -- list the treatments or codes
    that imply your outcome
  ! no splits.group_column is declared, so LK008 falls back to encounter_id if
    present and skips otherwise
```

This is deliberately a warning rather than an error. An under-specified task is
still worth auditing for what it does declare — but the report has to say which
checks were weakened, or a thin spec produces a clean report that reads as a
clean dataset.

## The splits file

JSON, as subject id lists:

```json
{
  "train": ["S000001", "S000002"],
  "valid": ["S000003"],
  "test":  ["S000004"]
}
```

Also accepted: a parquet with `subject_id` and `split`, or the MEDS dataset's
own `metadata/subject_splits.parquet`, which is used automatically when
`--splits` is omitted. MEDS split names (`train`, `tuning`, `held_out`) map to
ehrlint's `train`, `valid`, `test`.

If the splits name subjects the event table does not contain, the report says
so and calls it what it usually is — a subject-id type or format mismatch
between two files. That warning exists because the alternative is every split
check quietly comparing empty sets.
