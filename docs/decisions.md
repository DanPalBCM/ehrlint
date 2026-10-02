# Decisions and deviations

What was verified before it was relied on, where the implementation departs
from the specification, and the bugs that changed a design. SPEC §0 requires
this file; it is also the page to read before trusting anything else here.

## Verification before implementation

### D-001 — The name is free

`ehrlint` was unclaimed on PyPI and on GitHub when this project started. No
alternative was needed, so no approval was sought.

The project is released on GitHub only. There is no PyPI upload and no deployed
documentation site, by decision — `pip install ehrlint` in the README describes
what the package metadata supports, not a published artifact.

### D-002 — The MEDS schema, read from the installed package

SPEC §0 requires verifying the MEDS schema rather than assuming it. Verified
against **`meds` 0.4.1** by installing and introspecting it. The data schema:

| column | type | required |
| --- | --- | --- |
| `subject_id` | int64 | **yes** |
| `time` | timestamp[us] | **yes** (nullable for static events) |
| `code` | string | **yes** |
| `numeric_value` | float32 | no |
| `text_value` | large_string | no |

Layout constants, from the package's own values:

```python
meds.data_subdirectory          == "data"
meds.dataset_metadata_filepath  == "metadata/dataset.json"
meds.code_metadata_filepath     == "metadata/codes.parquet"
meds.subject_splits_filepath    == "metadata/subject_splits.parquet"
```

Split names: `train`, `tuning`, `held_out`.

Three differences from SPEC §2's sketch, all handled rather than treated as
incompatible, so no approval stop was triggered:

1. **No `code_system` column.** MEDS namespaces the vocabulary inside the code
   string as `ICD10CM//E11.9`. ehrlint splits on the first separator and leaves
   the system **null** when there is none, rather than guessing one.
2. **No `encounter_id` column.** LK002 and LK008 therefore skip on plain MEDS
   input, with a reason. MEDS sets `allow_extra_columns = True`, so a dataset
   may carry one; ehrlint looks for `encounter_id`, `visit_id`, `hadm_id`, and
   `visit_occurrence_id`.
3. **`subject_id` is int64**, while ehrlint compares subject ids as strings.
   Cast at ingestion; see D-012 for the reverse direction.

These constants are asserted equal to the installed package's own by a test, so
a MEDS upgrade that moves them fails CI rather than failing silently. The
`meds` extra is optional: without it ehrlint reads MEDS using these constants,
and with it an audit additionally warns when the installed version requires
columns ehrlint was not written against.

### D-003 — MIMIC-IV demo access, verified against the live server

SPEC §0 requires verifying the access pattern before writing the walkthrough.
The **MIMIC-IV Clinical Database Demo v2.2** is openly downloadable from
PhysioNet under the ODC Open Database License, with **no credentials** —
confirmed by fetching the file index and the license directly.

So the walkthrough uses it rather than falling back to Synthea only. It
downloads at runtime into a gitignored `mimic_demo/`, commits nothing, and
**no test depends on it**; the test suite makes no network calls. See
[the walkthrough](walkthroughs/mimic-iv.md).

This resolves the tension between SPEC §0 ("never use real patient data") and
SPEC §8.4 (a MIMIC-IV walkthrough): the demo subset is public, the script is
opt-in, and nothing it touches enters the repository or the test suite.

## Deviations from the specification

### D-004 — One `checks/leakage.py` instead of ten files

SPEC §4.1 lays out `checks/lk001_post_index.py` through `lk010_*.py`. All ten
live in one `checks/leakage.py` instead.

The ten checks share a substantial amount: the index-date join, code-set
matching, the denominator, the skip vocabulary. Split across ten files, that
either duplicates or moves into a `_shared.py` that the files then all import —
and reading any one check would mean reading two files. At ~240 statements
total the module is navigable, and the registry means the file layout is not
part of the public API.

### D-005 — No separate `splits.py` and `labels.py`

SPEC §4.1 lists top-level `splits.py` (split auditing) and `labels.py` (label
auditing). The auditing itself is in the checks (LK005–LK008 for splits,
LK004/LK007/LK010 for labels); the shared machinery is in `context.py`.

Two modules whose only content is helpers for three checks each would be
indirection without a reader. `schemas/splits.py` does exist, holding the
`Splits` model.

### D-006 — `Finding.examples` is a list, with `examples_frame()` beside it

SPEC §5.2 shows `finding.examples.head()`, implying a DataFrame. `examples` is
a `list[dict]` because findings serialize to JSON and JSONL (SPEC §6.3/§6.4)
and a frame does not. `examples_frame()` returns a Polars frame, so the
documented ergonomics work:

```python
finding.examples_frame().head()
```

### D-007 — `--fail-on info` is refused, not accepted

SPEC §5.1 says "exit 0 if no finding at or above `--fail-on`" without
enumerating the accepted values, and SPEC §4.4 defines `info` as descriptive
statistics.

Those two together make `--fail-on info` meaningless: info findings are
descriptive and never fire, so the flag would behave exactly like
`--fail-on warning` while reading as if it were stricter. It is rejected with
exit code 2 and that explanation. A threshold that does not do what it says is
worse than one fewer option.

### D-008 — `info` findings never fire

Related, and worth stating separately. `Finding.fired` is false for every
`info` finding regardless of its subject count. "120 of 120 subjects have
events after their index date" is true of every dataset with follow-up; making
it capable of failing a run would make the tool unusable.

### D-009 — LK001's event-level branch is `info`, not `error`

SPEC §4.3 describes LK001 as flagging "events or features with
`time > index_date` used as predictors". The phrase that carries the weight is
*used as predictors*, and ehrlint cannot see that from the event stream.

Post-index events exist in every dataset with follow-up — outcomes live there.
Flagging their presence as an error would make LK001 permanently red and teach
users to ignore it. So LK001 emits two findings: an `info` observation about
post-index events, and an `error` when a supplied feature matrix has rows
stamped after the index date. The second is the real check, and
`--features` is what enables it.

### D-010 — The outcome window is open at the index date

A task spec written `window: [index_date, index_date + 2y]` reads closed at the
lower bound. LK007 and LK010 treat it as **open**: an occurrence stamped at
exactly the index date is not predictable from the index date.

This is not a corner case. Most EHR extracts date a visit's diagnoses by the
admission timestamp, so an entire encounter's codes — including those entered
at discharge — land exactly on the index moment. Counting them as outcomes
would make such a task look positive and well predicted; LK002 reports them for
what they are.

Found while writing the MIMIC-IV walkthrough, where LK007 fired on 18 subjects
whose label query had used `>=`. The behaviour was right and the *message* was
wrong — it said "inside the declared window", which a user comparing their own
query could not reconcile. Every message that depends on this boundary now says
"strictly after the index date".

### D-011 — `exclusion` is recorded, not applied

SPEC §6.1 includes `exclusion: []` in the task spec. It is parsed and carried
into the report for provenance, but ehrlint does not apply it. v1 audits a
finished dataset (SPEC §3) rather than constructing one.

## Design decisions

### D-012 — A surrogate subject-id map when writing MEDS

MEDS requires an int64 `subject_id`; ehrlint's canonical ids are strings. A
string like `S000042` has no integer form, and
`cast(Int64, strict=False)` turns it into **null**.

That was a real bug, caught by the first end-to-end CLI run: the synthetic
writer nulled every subject id, and the audit then failed on evidence
validation. Had it not, a dataset with no subject ids would have audited
perfectly clean — the single worst failure available to this tool.

So: integer-valued ids are written directly; otherwise a deterministic
surrogate is assigned and the original written to
`metadata/subject_id_map.parquet`, an ehrlint extension MEDS permits. The
reader restores from it, and the splits file uses the same surrogates — ids
that disagreed between the two files would make LK005 and LK008 silently
vacuous. A non-null id that would become null raises, with the message saying
it is a bug in ehrlint rather than a problem with your data.

A MEDS round trip is consequently lossless except `numeric_value`, which the
standard stores as float32. That narrowing is asserted explicitly by the
round-trip test rather than hidden behind a tolerance.

### D-013 — `subject_id` is always coerced to a string

In the canonical table, unconditionally. An int64 subject id in the events and
a string one in the labels join to nothing, and a join that returns nothing
makes every check report clean. Normalizing at ingestion is the cheapest place
to prevent a false-clean audit.

### D-014 — A null `time` means a static fact, never the epoch

Demographics and birth records carry no timestamp. Treating null as the epoch
would make every static fact look like a pre-index event — wrong, and the kind
of wrong that produces confident false findings. Static events are excluded
from every time-ordering check, and the report says how many there were.

### D-015 — A crashing check is recorded as a skip, not a failure of the run

An exception in one check would otherwise lose the other nine results. It is
recorded as a skip whose reason names the exception type and states that it is
a bug in ehrlint rather than a finding about your data.

### D-016 — Evidence validation is on by default

After each check, the subjects a finding names are verified to exist in the
input. SPEC §0 forbids fabricated statistics; this is the mechanical guard
rather than a convention. It can be turned off with
`run_audit(..., validate_evidence=False)`, which exists for testing a check
that deliberately reports synthetic rows.

### D-017 — LK009 refuses to conclude on coarse demographics

Deterministic matching on the set of static codes needs the codes to be
discriminating. When distinct signatures number fewer than 90% of subjects, a
collision is the birthday paradox rather than a duplicate record, and LK009
skips with that arithmetic in the reason.

This fires in practice: on the MIMIC-IV demo, gender × anchor age × anchor year
group gives 82 profiles across 100 subjects, and the check declines. Listing
coincidences as candidate duplicates would be exactly the fabricated statistic
SPEC §0 forbids.

### D-018 — Findings about pairs name both subjects

LK008 (an encounter straddling two splits) and LK009 (two ids for one person)
are both about *pairs*. Both originally reported one representative subject per
group, which made `n_subjects` understate by half and left `findings.jsonl`
naming only one side of each pair.

Both now emit one row per affected subject, with the group count in `detail`.
"These two ids are one person" is not actionable with one id.

### D-019 — Example rows are sorted on a total key

`make_finding` sorts the offending frame before truncating, so the example rows
are the same rows regardless of input order. The sort key must be **total**:
sorting by subject and time alone left tied rows in whatever order the frame
held them, and truncating a tie picked different rows on different runs. Found
by the determinism test after D-018 introduced multi-row-per-subject evidence.

Columns with nested dtypes are excluded from the key, since a check is free to
put one in its evidence frame.

### D-020 — `inject_many` plants on disjoint subjects

Every injector is seeded identically and so chose the same subjects — and then
one leak masked another: LK010 gives a negative an outcome long after the
horizon, which extended the follow-up LK004 had just truncated, and LK004 went
quiet. Found by a CLI run that planted LK002, LK004, and LK010 and saw only
two findings.

Every injector now takes an `exclude` argument and `inject_many` accumulates
the subjects already used. An exhausted pool is an error: a leak that was never
planted would make a duality test pass vacuously.

### D-021 — The synthetic generator gives each subject a unique high-cardinality fact

"Clean by construction" has to hold for every seed. Gender × race × birth
decade × postal code gives ~280,000 profiles, which sounds ample until the
birthday paradox is applied: at 87 subjects a chance collision happened in
roughly one dataset in seventy, and a chance collision made LK009 fire on data
documented as clean. Found by the property test over seeds.

Each subject now carries a unique fact derived from its index — standing in for
the exact birth date deterministic matching keys on — alongside a repeating
postal code for realism.

### D-022 — The HTML report escapes; the Markdown report does not

The report prints codes, free-text values, and example rows straight from the
input. A single `<` in a text value would truncate the page from that point on.

The subtlety that made this a bug twice: the templates are named
`report.html.j2` and `report.md.j2`, and Jinja's `select_autoescape` looks at
the *final* suffix, finds `.j2`, and leaves escaping off for both. A test using
a bare `.html` name passed while the shipped template was unescaped. The policy
now strips the Jinja marker before checking the suffix, and a test asserts
`_should_autoescape("report.html.j2") is True` specifically.

### D-023 — `AuditReport.finding()` returns the finding that fired

A check may emit more than one finding (D-009). Returning the first emitted
handed callers LK001's reassuring `info` observation while an `error` sat
behind it. `finding()` now returns the most serious one, preferring any that
fired, and `findings_for()` returns them all.

### D-024 — `--run-slow` is implemented, not just declared

The `slow` marker was documented in `pyproject.toml` as "run with
`--run-slow`", but nothing implemented the option, so the full-scale
performance measurements ran in every default suite. A marker documented as
gated but not gated is how a slow test ends up deleted instead of fixed.

### D-025 — OMOP start dates, and the unmapped-concept label

Every OMOP table is read at its **start** timestamp. A condition's onset is
what a prediction task indexes on; an end date would systematically shift
events later and mask exactly the post-index leaks this tool looks for. A test
asserts no table spec references a column with `end` in its name.

`concept_id = 0` — OMOP's "did not map" sentinel — falls back to the source
value, and the system label becomes `OMOP_CONDITION_SOURCE` rather than
`OMOP_CONDITION`. Keeping concept 0 would collapse every unmapped row onto one
meaningless code; reusing the standard label would let a task matching standard
concepts silently also match unmapped source strings.

### D-026 — The check documentation is generated

SPEC §4.3 asks each check to document its rationale, its exact query, its
default severity, and a worked synthetic example. All four already exist in the
code, so `docs/checks/*.md` is generated by
`scripts/generate_check_docs.py` and a test asserts the committed pages match
what the generator produces. Four hand-maintained copies of the same facts
would drift, and prose drifts from code silently.

## Measured, not asserted

### D-027 — The §4.5 performance target

SPEC §4.5: the core checks run on 100,000 subjects / 10,000,000 events in under
five minutes on a 16 GB laptop.

| cohort | events | audit time | peak RSS |
| --- | --- | --- | --- |
| 100,000 subjects | 1,608,344 | 9.3 s | — |
| 640,000 subjects | 10,302,048 | 67.5 s | 8.3 GiB |

Both halves are measured, because 100,000 subjects alone only reaches ~1.6 M
events — which would let a tool that is slow per *event* pass a test named
after the target. Reproduce with `uv run pytest --run-slow`.

The default suite runs a smaller version that compares the per-subject cost at
two cohort sizes, so a regression that makes a check quadratic is caught on
every push rather than only in a scheduled job.
