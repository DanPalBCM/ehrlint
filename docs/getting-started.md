# Getting started

## Install

```bash
uv add ehrlint              # or: pip install ehrlint
uv add 'ehrlint[meds]'      # adds MEDS schema validation
uv add 'ehrlint[all]'       # everything
```

Python 3.11, 3.12, and 3.13 are tested in CI.

The `meds` extra is optional and only adds validation: ehrlint reads MEDS
datasets without it, using layout constants that a test asserts equal to the
installed package's own when it is present. With the extra installed, an audit
also warns if the installed MEDS version requires columns ehrlint was not
written against.

## Run the demo

The repository ships two synthetic cohorts and a demo that audits both:

```bash
git clone https://github.com/DanPalBCM/ehrlint
cd ehrlint
uv sync
uv run python demo.py
```

The demo:

1. Validates the task spec and reports which checks it would weaken.
2. Audits the clean cohort — which must pass. A leakage auditor that fires on
   everything tells you nothing, so the clean pass is what makes the rest mean
   something.
3. Audits a cohort with four planted leaks.
4. Checks every finding against the ground truth in
   `examples/leaky_meds/planted_leaks.json`, and exits non-zero if a check went
   quiet or named the wrong subjects.
5. Shows the evidence behind one finding: the query, then the rows.
6. Writes both reports.

Open `demo_output/leaky/report.html` — a single self-contained file, no
JavaScript, figures inlined as data URIs so it survives being attached to a
ticket without its folder.

## Audit your own data

You need two things: an event dataset and a task spec.

```bash
ehrlint audit \
  --data path/to/meds/ \
  --task task.yaml \
  --splits splits.json \
  --labels labels.parquet \
  --out report/
```

Add `--features features.parquet` if you have the matrix you actually trained
on. Without it, LK001 can only report that post-index events exist — true of
every dataset with follow-up — rather than that they reached your features.

For an OMOP CDM extract, add `--format omop`:

```bash
ehrlint audit --data path/to/omop/ --format omop --task task.yaml --out report/
```

See [ingestion](ingestion.md) for what each format needs.

## Write a task spec

The minimum is an index definition and an outcome:

```yaml
task: dka_2y
index:
  event: ed_visit
  code_system: VISIT
outcome:
  codes:
    - system: ICD10CM
      values: [E10.10, E11.65]
horizon: 2y
followup:
  require_full_horizon_for_negatives: true
```

Check it before you have data:

```bash
ehrlint validate-task task.yaml
```

That tells you which checks the spec weakens. A spec with no `proxy_codes`
cannot support LK003, and the right time to learn that is now rather than from
a skip in the report. See [task specs](task-spec.md) for every field.

## Use it as a CI gate

```yaml
- name: Audit the task
  run: |
    ehrlint audit --data data/meds --task tasks/dka_2y/task.yaml \
      --splits tasks/dka_2y/splits.json --out report/
```

Exit codes:

| code | meaning |
| --- | --- |
| `0` | nothing fired at or above `--fail-on` |
| `1` | something did |
| `2` | usage or input error — a missing file, an unknown check id |

The three are distinct on purpose. A broken invocation must never be
mistakable for a clean audit, because that is how a gate gets silently
disabled.

One thing to know about gating: the exit code answers "did anything fire", and
it cannot also answer "was the audit complete". An audit where every check
skipped exits `0`. If you are gating on ehrlint, check
`summary.n_checks_skipped` in `report.json` as well — or read the report, where
skips are printed first.

## Generate test data

```bash
# A clean cohort: audits to exit code 0 with nothing skipped
ehrlint synth --out clean/ --n-subjects 5000 --seed 0

# The same cohort with two specific leaks planted
ehrlint synth --out leaky/ --n-subjects 5000 --seed 0 --inject LK001,LK004
```

Same seed, same bytes. See [synthetic data](synthetic-data.md).

## From Python

```python
import ehrlint as el

ctx = el.AuditContext.from_meds(
    "data/",
    task_spec="tasks/dka_2y/task.yaml",
    labels_path="tasks/dka_2y/labels.parquet",
    splits_path="tasks/dka_2y/splits.json",
)

report = el.run_audit(ctx)
print(report.summary.to_markdown())

for finding in report.fired:
    print(finding.check_id, finding.severity, finding.subject_rate)
    print(finding.examples_frame().head())

report.write_html("report/")
```

`report.fired` is the findings that report a problem, worst first.
`report.passed` is the checks that ran and found nothing. `report.skipped` is
the ones that could not run, with reasons. The three are deliberately separate.
