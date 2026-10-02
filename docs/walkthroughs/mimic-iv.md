# MIMIC-IV walkthrough

```bash
uv run python scripts/mimic_demo_walkthrough.py
```

This is the one part of ehrlint that touches real patient records, which is
worth being precise about before anything else.

## Access and handling

The **MIMIC-IV Clinical Database Demo** (v2.2) is a 100-patient subset
published openly under the ODC Open Database License. It needs no PhysioNet
credentials — that was verified against the live server rather than assumed,
which is what makes this walkthrough runnable by anyone reading this page.

It is still de-identified records of real people, so:

- the files download **at runtime** into `mimic_demo/`, which is gitignored
- **nothing from it is committed**, and the repository contains no MIMIC data
- **no test depends on this script**; the test suite makes no network calls
- `uv run python scripts/mimic_demo_walkthrough.py clean` removes the download

The full MIMIC-IV is credentialed and this walkthrough does not apply to it,
beyond the mapping below, which you are free to reuse.

> Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R.
> (2023). MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet.
> <https://doi.org/10.13026/dp1f-ex47>

## Why the mapping lives in a script

MEDS and OMOP are the formats ehrlint supports. MIMIC's own schema is a third,
and the converter stays in `scripts/` rather than in the package so that
distinction is honest: it is an example of writing a converter, not a supported
input format.

It is also short, which is the point — mapping a new source onto the canonical
event table is mostly a matter of deciding which timestamp each table gets.

## What the script does

### 1. Download

Four tables and the license: `patients`, `admissions`, `diagnoses_icd`,
`labevents`. The demo holds far more; this is what the audit needs.

### 2. Map to the canonical event table

```text
112,823 events, 100 subjects, 300 static
  MIMIC_LABITEM              107,727
  ICD10CM                    2,313
  ICD9CM                     2,193
  MIMIC_ADMISSION            275
  MIMIC_ANCHOR_AGE           100
  MIMIC_GENDER               100
  MIMIC_ANCHOR_YEAR_GROUP    100
```

Three decisions in that mapping are worth naming.

**`admittime`, never `dischtime`.** A discharge time postdates everything that
happened in the visit, so indexing on it would push the prediction moment past
the information it is meant to exclude.

**Diagnoses are dated by their admission, because there is nothing else.**
`diagnoses_icd` carries no timestamp of its own. Every diagnosis for a visit
therefore lands at `admittime` — including the codes entered at discharge. This
is not a quirk of the demo; it is how most EHR extracts date their diagnosis
codes, and it is exactly what LK002 exists to find.

**Demographics are static.** MIMIC shifts dates per patient for
de-identification, so `anchor_year` is not a real year. It is still a
per-patient constant, which is all LK009's signature needs, and it carries a
null timestamp so no time-ordering check treats it as an event.

Every column is read as text and cast explicitly. MIMIC's identifier columns
look numeric but are identifiers, and letting schema inference turn
`subject_id` into an integer is the first step towards the silent empty join
that makes an audit look clean.

### 3. Define a task

A type 2 diabetes diagnosis within a year of the first admission.

Not readmission, and the reason is instructive: with a readmission task the
outcome code *is* the index code, so the index admission is itself an outcome
occurrence in the index encounter, and LK002 fires on 100% of subjects. That is
a correct finding about a circular task definition and a useless demonstration,
because it would happen on any dataset.

### 4. Audit

```text
2 check(s) did not run. A skip is not a pass:
   LK003  the task spec declares no outcome.proxy_codes. ehrlint cannot infer
          which codes imply your outcome
   LK009  the static demographics distinguish only 82 profile(s) across 100
          subject(s), so collisions are expected by arithmetic rather than
          evidence of duplication

check                            severity  subjects    rate
LK002 same_encounter             error           22   22.0%
LK004 immortal_time              error           50   50.0%
LK007 label_definition           error           18   18.0%
LK010 horizon_violation          error            6    6.0%
```

Exit code 1, on a real dataset, with a task definition written in good faith.
Each of those four is worth understanding, because they are the four most
common ways a clinical task goes wrong.

**LK002 (22%).** The diabetes code is in the index encounter, dated to
`admittime`. A model given that encounter's codes as features would be reading
its own label.

**LK004 (50%).** Half the cohort is not observed for a full year after their
first admission. The demo is a small slice of a hospital's data, so most
patients simply stop appearing — and labelling them negative counts censoring
as evidence.

**LK007 (18%).** The label query in the script uses `>=` on the index date; the
outcome window is open at the lower bound. Those 18 subjects are positive only
because of a code stamped at exactly the index moment — which is the same
pattern LK002 reported, seen from the label side. See
[the lower bound is open](../labels.md#the-lower-bound-is-open).

**LK010 (6%).** Six subjects have their only diabetes code more than a year
after the index date. They are negatives for a one-year task.

**LK009's skip is the interesting one.** Gender × anchor age × anchor year group
gives 82 distinct profiles across 100 subjects, below the threshold at which a
collision means anything. Rather than listing coincidences as candidate
duplicates, the check refuses to conclude and says why. On the full MIMIC-IV,
with more demographic detail, it would run.

### 5. Report

Four artifacts in `demo_output/mimic/`. Open `report.html`.

## Synthea

If you would rather not download real records at all, `ehrlint synth` produces
cohorts with the same structure and known ground truth — see
[synthetic data](../synthetic-data.md). A Synthea export can be mapped the same
way the script maps MIMIC, since Synthea emits FHIR or CSV with explicit
timestamps on every resource; the only real work is choosing which resource
type supplies the index event.
