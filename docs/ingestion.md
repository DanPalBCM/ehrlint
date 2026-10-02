# MEDS and OMOP ingestion

Both formats become [the canonical event table](concepts.md). What follows is
what each conversion assumes, because an audit is only as good as the
assumption underneath it.

## MEDS

```bash
ehrlint audit --data path/to/meds/ --task task.yaml
```

Expected layout, taken from the `meds` package's own constants:

```text
<root>/data/**/*.parquet                 the event shards
<root>/metadata/dataset.json             dataset metadata
<root>/metadata/codes.parquet            code metadata
<root>/metadata/subject_splits.parquet   subject_id, split
```

A flat directory of parquet files with no `data/` subdirectory is also
accepted — hand-made extracts often look like that, and insisting on the full
layout would be pedantry.

### Verified against the installed package

The MEDS schema is versioned and evolving, so it was read from the installed
package rather than from memory. Against `meds` 0.4.1 the data schema is:

| column | type | required |
| --- | --- | --- |
| `subject_id` | int64 | **yes** |
| `time` | timestamp[us] | **yes** (nullable for static events) |
| `code` | string | **yes** |
| `numeric_value` | float32 | no |
| `text_value` | large_string | no |

Three differences from a MEDS-like sketch matter, and all three are handled
here rather than assumed away.

**There is no `code_system` column.** MEDS namespaces the vocabulary inside the
code string, as `ICD10CM//E11.9`. ehrlint splits on the first separator to
recover a system and leaves it null when there is no separator — rather than
guessing one, which would cross-match between vocabularies later.

**There is no `encounter_id` column.** The same-encounter checks (LK002, LK008)
therefore skip on plain MEDS input and say so. MEDS sets
`allow_extra_columns = True`, so a dataset may carry one; ehrlint looks for
`encounter_id`, `visit_id`, `hadm_id`, and `visit_occurrence_id`, or you can
name it explicitly.

**`subject_id` is int64**, while ehrlint compares subject ids as strings
everywhere. It is cast at ingestion. The reverse direction is the interesting
one — see below.

With the `meds` extra installed, an audit also warns when the installed version
requires columns ehrlint was not written against, instead of failing on a
schema change it cannot know about.

### Writing MEDS, and the subject-id surrogate

`ehrlint synth` writes datasets in real MEDS layout, so the demo exercises the
real reader rather than a shortcut. That runs into the int64 requirement: a
string id like `S000042` has no integer form.

A plain `cast(Int64, strict=False)` turns it into **null** — and a dataset with
no subject ids audits perfectly clean, which is the single worst failure
available to this tool. So:

- When every id is already integer-valued, they are written directly.
- Otherwise a deterministic surrogate integer is assigned and the original is
  written to `metadata/subject_id_map.parquet`, an ehrlint extension that MEDS
  permits. The reader restores from it, and the splits file uses the same
  surrogates as the shards — different ids in the two files would make LK005
  and LK008 silently vacuous.
- A non-null id that *would* become null raises, with the message saying it is
  a bug in ehrlint rather than a problem with your data.

A MEDS round trip is lossless except for `numeric_value`, which the standard
stores as float32. That narrowing is a property of MEDS, not a bug, and the
round-trip test asserts it explicitly so nobody comparing two hashes is
surprised.

## OMOP CDM

```bash
ehrlint audit --data path/to/omop/ --format omop --task task.yaml
```

Tables read, as parquet or CSV, case-insensitively:

| table | time column | code column | system label |
| --- | --- | --- | --- |
| `condition_occurrence` | `condition_start_datetime` | `condition_concept_id` | `OMOP_CONDITION` |
| `drug_exposure` | `drug_exposure_start_datetime` | `drug_concept_id` | `OMOP_DRUG` |
| `measurement` | `measurement_datetime` | `measurement_concept_id` | `OMOP_MEASUREMENT` |
| `procedure_occurrence` | `procedure_datetime` | `procedure_concept_id` | `OMOP_PROCEDURE` |
| `observation` | `observation_datetime` | `observation_concept_id` | `OMOP_OBSERVATION` |
| `visit_occurrence` | `visit_start_datetime` | `visit_concept_id` | `OMOP_VISIT` |
| `person` | — (static) | demographics | `OMOP_GENDER`, … |

The `*_date` variant is used when the `*_datetime` one is absent.

### Three choices worth stating

**Start dates throughout.** A condition's *onset* is when it became true of the
patient, which is what a prediction task indexes on. Using an end date would
systematically shift events later and mask exactly the post-index leaks this
tool exists to find. A test asserts no spec references a column with `end` in
its name.

**`concept_id = 0` falls back to the source value.** Zero is OMOP's sentinel
for "this source code did not map to a standard concept". Keeping it would
collapse every unmapped row onto one meaningless code. The system label becomes
`OMOP_CONDITION_SOURCE` rather than `OMOP_CONDITION`, so a task matching
standard concepts does not silently also match unmapped source strings.

**`person` becomes static, null-time events.** A birth year is a fact about a
person, not an event competing with the index date, so it carries no timestamp
and is excluded from every time-ordering check. `birth_datetime`, where
present, does become a timed `MEDS_BIRTH` event.

`visit_occurrence_id` becomes `encounter_id`, which is what makes LK002 and
LK008 runnable on OMOP input where they would skip on plain MEDS.

## Labels

Either shape works:

```text
ehrlint:  subject_id, index_date, label
MEDS:     subject_id, prediction_time, boolean_value | integer_value | float_value
```

MEDS spreads the label across typed columns; ehrlint normalizes whichever is
populated into a single `label`, and takes `prediction_time` as the index date
when the task spec does not define one from an event code.

## Feature matrices

Parquet or CSV. The subject column is found among `subject_id`, `patient_id`,
`person_id`, and the as-of timestamp among `prediction_time`, `index_date`,
`as_of`. Both can be named explicitly.

If no subject column can be found, that is an error listing the names it looked
for — guessing one would risk auditing the wrong column, which is worse than
stopping. See [feature-matrix auditing](features.md).
