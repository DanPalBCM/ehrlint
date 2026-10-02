# Reading a report

```bash
ehrlint audit --data meds/ --task task.yaml --out report/
```

Four artifacts:

| file | what it is for |
| --- | --- |
| `report.html` | a single self-contained file, for a human |
| `report.md` | the same content, for a pull request or a notebook |
| `report.json` | the whole audit, for a program |
| `findings.jsonl` | one finding per line, for a pipeline |

## The HTML report is one file

Figures are inlined as base64 data URIs and there is no JavaScript. An audit
report gets attached to a review, a ticket, or a submission, and a report that
breaks when its `figures/` directory is lost is a report that will be seen
broken. CI asserts the file contains no `<script`, references no external URL,
and does contain inlined images.

## The order is deliberate

It is the opposite of a dashboard's. From the top:

1. **The headline counts** — subjects, events, checks run, errors, warnings, skips.
2. **Pass or fail**, with the threshold that produced it.
3. **Skipped checks**, with the reason each one could not run.
4. **Warnings about the inputs**.
5. **Figures**.
6. **The data and task description**.
7. **The findings**, worst first.
8. **Observations** (`info`).
9. **Checks that ran and found nothing.**

Skips and input warnings come *before* the findings because "six of ten checks
did not run" changes how every result below should be read, and a reader who
meets a green summary first will stop there.

## What a finding contains

Each fired finding prints, in order:

- the check id, name, and severity
- one actionable sentence
- how many subjects and rows, with the denominator
- **why it matters** — the check's rationale, so the finding is not just a number
- **the query that found it**, in SQL-ish pseudocode you can run by hand
- **example rows** from your data, not aggregates
- the affected subject ids, and any structured detail

The query and the rows are the point. A finding you cannot independently
reproduce is an assertion, and this report is meant to be the kind of thing a
reviewer can check.

## Skips, passes, and observations are three different things

**A skip** means the check could not run. It reports a reason. It is not
evidence of anything about your data.

**A pass** means the check ran and found nothing, and it says what it verified:

```text
LK005  split_overlap  The 3 splits are disjoint by subject.
```

**An observation** (`info`) is a descriptive statistic. It never fails a run.
"120 of 120 subjects have events after their index date" is true of every
dataset with follow-up; it is there for orientation, not as a problem.

The report lists all three separately, and the summary counts them separately,
because collapsing them is how an incomplete audit comes to read as a clean
one.

## Exit codes and gating

| code | meaning |
| --- | --- |
| `0` | nothing fired at or above `--fail-on` |
| `1` | something did |
| `2` | usage or input error |

`--fail-on` accepts `error` (the default) and `warning`. It refuses `info`,
with the reason: info findings are descriptive and never fire, so accepting
`--fail-on info` would give you a flag that behaves exactly like
`--fail-on warning` while reading as if it were stricter.

**One thing to watch when gating.** The exit code answers "did anything fire".
It cannot also answer "was the audit complete" — an audit where every check
skipped exits `0`. If ehrlint is a gate in your pipeline, read
`summary.n_checks_skipped` too:

```bash
ehrlint audit --data meds/ --task task.yaml --out report/
skipped=$(jq '.summary.n_checks_skipped' report/report.json)
test "$skipped" -eq 0 || { echo "incomplete audit: $skipped checks skipped"; exit 1; }
```

## The JSON

```json
{
  "task": "dka_2y",
  "data_hash": "7703496d5aa2947f",
  "findings": [...],
  "skipped": [...],
  "summary": {
    "n_checks_requested": 10,
    "n_checks_run": 10,
    "n_checks_skipped": 0,
    "n_fired": 4,
    "n_errors": 4,
    "n_warnings": 0,
    "n_subjects": 120,
    "worst_severity": "error"
  },
  "warnings": [...],
  "context": {...},
  "check_metadata": [...],
  "fail_on": "error",
  "exit_code": 1,
  "ehrlint_version": "1.0.0"
}
```

`data_hash` is a content hash of the event table, invariant to row order, so
two audits can be compared and a finding can be tied to the exact data that
produced it. The same data loaded by two different readers hashes the same.

## Determinism

Two audits of the same data produce the same findings, in the same order, with
the same example rows. A report is evidence attached to a review; one that
changes between runs is not evidence.

This is a tested property, not an aspiration, and it took one real fix to hold:
example rows are now sorted on a *total* key before truncation. Sorting by
subject and time alone left tied rows in whatever order the frame happened to
hold them, and truncating a tie picked different rows on different runs.
