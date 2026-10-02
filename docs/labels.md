# Label auditing and immortal time

Three checks look at the labels rather than the features: LK004, LK007, and
LK010. They catch different versions of the same mistake — a label that is
correct about the patient and wrong about the prediction problem.

## LK004: immortal time

A subject labelled negative who left the data a month after the index date is
not a two-year negative. They are censored.

Counting them as negatives means part of the negative class is made of people
who had no *opportunity* to have the outcome. Every metric improves, and none
of the improvement is real. This is the classic immortal-time error, and it is
easy to introduce because the label query looks right: no outcome code was
found, so the label is false.

What LK004 does:

```sql
WITH followup AS (
  SELECT subject_id, MAX(time) AS last_event_time FROM events
  WHERE time IS NOT NULL GROUP BY subject_id
)
SELECT * FROM index_dates i LEFT JOIN followup f USING (subject_id)
WHERE DATEDIFF('day', i.index_date, f.last_event_time) < 730.5
  AND subject_id NOT IN (outcome_subjects UNION competing_event_subjects)
```

Subjects with the outcome are excluded, because they had the event and their
follow-up ending is not a problem. Subjects with a **competing event** —
usually death — are excluded too, if the task declares
`allow_death_as_censoring`. A patient who died at six months in a two-year task
was observed as long as they could be.

That last exclusion is why `competing_codes` is worth filling in. Without it,
every death in the cohort becomes an LK004 finding, and a check with a large
false-positive rate gets switched off.

The check skips entirely when the task sets
`require_full_horizon_for_negatives: false` — there is no requirement to
violate.

## LK007: label definition

The label asserts an outcome the event data does not support inside the
declared window.

This is what "ascertained with post-index information" looks like from the
outside. A label built from a registry flag, or from a chart review that used
the whole record, is correct about the patient — they did develop the
condition — and wrong as a label for a two-year-ahead prediction task.

LK007 compares each positive label against the outcome codes inside the
window:

```sql
WITH in_window AS (
  SELECT DISTINCT e.subject_id FROM events e
  JOIN index_dates i USING (subject_id)
  WHERE e.code IN outcome_codes AND e.time > i.index_date
    AND DATEDIFF('day', i.index_date, e.time) <= 730.5
)
SELECT * FROM labels l WHERE l.label = TRUE
  AND l.subject_id NOT IN (SELECT subject_id FROM in_window)
```

It skips without a labels table, and it skips when the task declares no outcome
codes — there would be nothing to compare against.

### The lower bound is open

Note the `>` in that query, not `>=`. **The outcome window is open at the index
date**: an occurrence stamped at exactly the index moment does not count as an
outcome, because it is not predictable from that moment.

This matters more on real data than it sounds. Most EHR extracts date a visit's
diagnoses by the admission timestamp, so an entire encounter's codes — including
the ones entered at discharge — land exactly on the index date. Counting them
would make every such task look positive and well predicted. LK002 is the check
that reports them for what they are.

A task spec written `window: [index_date, index_date + 2y]` reads closed, so if
your own label query used `>=`, LK007 will flag subjects your query called
positive. That is the boundary, not a bug, and every message that depends on it
says "strictly after the index date" rather than "inside the declared window".

## LK010: horizon violation

An outcome occurrence beyond the declared horizon, counted as a positive.

Usually this is a label query with no upper bound on time: "did this patient
ever get this code", with the index date applied as a lower bound and nothing
applied as an upper one. The prevalence comes out too high, and the model is
being trained on a different question than the one stated.

LK010 differs from LK007 in which direction the mismatch runs. LK007 finds
positives with *no* in-window evidence; LK010 finds the specific case where the
only evidence is *outside* the horizon. A subject flagged by LK010 usually
appears in LK007 too, and the two messages together tell you what to fix.

## Reading these three together

| finding | the label says | the data says |
| --- | --- | --- |
| LK004 | negative | nobody looked long enough to know |
| LK007 | positive | no in-window outcome at all |
| LK010 | positive | the outcome happened, but too late to count |

All three reduce your positive count or your cohort size when fixed. That is
the point: the number was wrong before.

## Competing events

```yaml
outcome:
  competing_codes:
    - system: null
      values: [MEDS_DEATH]
followup:
  allow_death_as_censoring: true
```

A competing event ends follow-up without being the outcome. ehrlint uses them
only to avoid false LK004 findings — it does not estimate competing-risk
models, and a task where competing risks are the scientific question needs more
than a leakage auditor.
