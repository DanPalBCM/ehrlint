# ehrlint

**Lint your clinical prediction task before you train on it.**

[![ci](https://github.com/DanPalBCM/ehrlint/actions/workflows/ci.yml/badge.svg)](https://github.com/DanPalBCM/ehrlint/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Most published clinical prediction models have a quiet flaw: the features, the
labels, or the splits leak information from the future. Post-index features,
outcome codes entered during the index encounter, treatment proxies, immortal
time, patient overlap across splits — these are the most common reasons
clinical models look excellent in a paper and fail in deployment.

Nothing widely used audits for them. Cohort-extraction systems help you
*define* a task; ehrlint audits one that already exists.

```bash
ehrlint audit --data meds/ --task task.yaml --splits splits.json --out report/
```

```
ehrlint dka_2y  120 subjects, 1919 events, 10/10 checks run

check                             severity  subjects   rate
LK001 post_index_features_matrix  error            5   4.2%
LK002 same_encounter              error            5   4.2%
LK004 immortal_time               error            5   4.2%
LK010 horizon_violation           error            5   4.2%

FAIL 4 error(s), 0 warning(s) at threshold error
```

Exit code `1`. Your CI just caught a leaky task.

## Two rules the whole tool is built around

**Evidence or nothing.** Every finding references the concrete rows that
triggered it — subject ids, event times, feature names — plus the query that
found them, so you can reproduce any finding by hand. A `Finding` with a
non-zero count and no example rows is rejected by a validator, not merely
discouraged.

**A skipped check is not a passed check.** A check that cannot run says so
with a reason, and the report prints skips *above* the findings. A clean report
that quietly skipped six of ten checks is worse than a dirty one, because it
reads as reassurance.

## Install

```bash
uv add ehrlint              # or: pip install ehrlint
uv add 'ehrlint[meds]'      # adds MEDS schema validation
```

Python 3.11+. The core depends on Polars, DuckDB, Pydantic, Typer, and
matplotlib.

## Try the demo

The repository ships two synthetic cohorts and a demo that audits both:

```bash
git clone https://github.com/DanPalBCM/ehrlint
cd ehrlint
uv sync
uv run python demo.py
```

It audits the clean cohort (which must pass — a leakage auditor that fires on
everything tells you nothing), audits a cohort with four planted leaks, and
then checks every finding against the ground truth in
`examples/leaky_meds/planted_leaks.json`. The demo exits non-zero if a check
goes quiet or names the wrong subjects, so it verifies the tool rather than
just running it.

Open `demo_output/leaky/report.html` — a single self-contained file, no
JavaScript, figures inlined.

## The checks

| ID | Name | What it flags | Default |
|---|---|---|---|
| LK001 | `post_index_features` | Events or feature rows timestamped after the index date | error |
| LK002 | `same_encounter` | Outcome codes from the index encounter itself | error |
| LK003 | `outcome_proxy` | Declared proxy codes occurring pre-index | warning |
| LK004 | `immortal_time` | Negatives who could not have had the outcome during follow-up | error |
| LK005 | `split_overlap` | Subject ids present in more than one split | error |
| LK006 | `temporal_split` | Test index dates preceding the training period's end | error¹ |
| LK007 | `label_definition` | Labels unsupported by any outcome inside the window | error |
| LK008 | `encounter_group` | One encounter or provider straddling two splits | error |
| LK009 | `duplicate_subjects` | One person under more than one subject id | warning |
| LK010 | `horizon_violation` | Outcome occurrences beyond the declared horizon | error |

¹ `error` when the task spec declares `splits.temporal: true` — a declared
property that does not hold is a broken claim, not a hint — and `warning`
otherwise.

Run `ehrlint list-checks` for the installed battery, and see
[the check reference](https://github.com/DanPalBCM/ehrlint/tree/main/docs/checks)
for each check's rationale, exact query, and worked example.

## Usage

### CLI

```bash
# Audit a MEDS dataset
ehrlint audit --data meds/ --task task.yaml --splits splits.json --out report/

# Audit the feature matrix you actually trained on
ehrlint audit --data meds/ --task task.yaml --features features.parquet --out report/

# OMOP CDM input
ehrlint audit --data omop/ --format omop --task task.yaml --out report/

# Narrow the battery, tighten the gate
ehrlint audit --data meds/ --task task.yaml --checks LK001,LK002,LK005 --fail-on warning

# Generate synthetic data, optionally with specific leaks planted
ehrlint synth --n-subjects 5000 --inject LK001,LK004 --seed 0 --out demo/

# Check a task spec before you have data
ehrlint validate-task task.yaml
```

Exit codes: `0` nothing fired at or above `--fail-on`; `1` something did; `2`
usage or input error. The three are distinct on purpose — a broken invocation
must never look like a clean audit.

### Python

```python
import ehrlint as el

ctx = el.AuditContext.from_meds(
    "data/",
    task_spec="tasks/dka_2y/task.yaml",
    labels_path="tasks/dka_2y/labels.parquet",
    splits_path="tasks/dka_2y/splits.json",
)

report = el.run_audit(ctx, checks=["LK001", "LK002", "LK005"])
print(report.summary.to_markdown())

for finding in report.fired:
    print(finding.check_id, finding.severity, finding.subject_rate)
    print(finding.examples_frame().head())   # concrete rows, not aggregates

report.write_html("report/")
```

Custom checks register into the same battery:

```python
@el.register_check(id="LK099", name="my_check", severity="warning")
def my_check(ctx: el.AuditContext) -> list[el.Finding]:
    ...
```

## Inputs

**Events.** A [MEDS](https://github.com/Medical-Event-Data-Standard/meds)
dataset directory, or an OMOP CDM extract as parquet or CSV. Both become one
canonical table: `subject_id`, `time`, `code`, `code_system`, `numeric_value`,
`text_value`, `encounter_id`.

**Task spec.** YAML describing the index date, the outcome, the horizon, and
the follow-up requirements:

```yaml
task: dka_2y
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
horizon: 2y
followup:
  require_full_horizon_for_negatives: true
splits:
  strategy: subject
  temporal: true
  group_column: encounter_id
```

`ehrlint validate-task` tells you which checks a given spec weakens. A spec
with no `proxy_codes` cannot support LK003, and the right time to learn that is
before the audit rather than from a skip in the report.

**Feature matrix (optional).** The matrix you actually trained on. Without it,
ehrlint audits the event stream against the task spec; with it, LK001 can show
that post-index information reached your features rather than merely that it
exists in the data.

**Splits (optional).** JSON, parquet, or the dataset's own
`metadata/subject_splits.parquet`. MEDS names (`train`/`tuning`/`held_out`) are
mapped to `train`/`valid`/`test`.

## Testing approach

Every check has a matching **injector** that plants exactly the leak it looks
for. The test strategy is differential: generate clean data, assert nothing
fires; plant one leak, assert *that* check fires and names the subjects the
injector touched. A check that fires on the wrong people is as broken as one
that stays silent.

A property test runs that duality over a range of seeds and cohort sizes, and
over all ten leaks planted at once on disjoint subjects. Two findings from
that test are worth knowing about:

- Injectors can mask each other. Every injector was seeded identically and so
  chose the same subjects, and LK010's far-future outcome extended the
  follow-up that LK004 had just truncated — LK004 went quiet. `inject_many`
  now plants on disjoint subjects.
- "Clean by construction" has to hold for *every* seed. At 87 subjects, the
  generator's demographic signature collided by chance in roughly one dataset
  in seventy, and a chance collision made LK009 fire on data documented as
  clean. The generator now gives each subject a unique high-cardinality fact.

The §4.5 performance target is measured, not asserted: 100,000 subjects audit
in 9 s and 10.3 M events in 68 s on a laptop, against a five-minute budget.
Run `uv run pytest --run-slow` to reproduce.

## No patient data, structurally

Every value in a generated dataset comes from a fixed module-level vocabulary,
a sequential subject index, or an arithmetic offset from a constant epoch. No
free text is generated at all, so there is no note text to de-identify. Tests
assert that property rather than spot-checking the output, CI fails on any data
file committed outside `examples/`, and every committed dataset carries a
manifest saying what it is.

The MIMIC-IV demo walkthrough is openly downloadable (ODbL, 100 patients, no
credentials) and downloads at runtime into a gitignored directory. Nothing from
it is committed and no test depends on it.

## Limitations

Stated plainly, because a tool that audits other people's claims should be
honest about its own.

**A clean audit does not mean a valid task.** ehrlint checks ten known leakage
patterns. It knows nothing about whether your cohort answers your clinical
question, whether your outcome definition is the accepted one, or whether your
index date is the clinically right moment.

**It cannot see your feature pipeline.** Without `--features`, LK001 can only
report that post-index events exist — which is true of every dataset with
follow-up. Whether your pipeline *consumed* them is not visible in the event
stream, which is why that branch is an observation and not an error.

**Proxies are yours to declare.** LK003 looks for the codes you list. ehrlint
cannot infer that an insulin drip implies DKA in your cohort, and guessing
would produce findings nobody could verify.

**Duplicate detection is deterministic, not probabilistic.** LK009 matches on
the exact set of static demographic facts. It will miss duplicates that differ
by a typo, and it refuses to report anything when the demographics in your
data are too coarse for a collision to mean something. A fuzzy matcher would
produce more findings and fewer checkable ones.

**Horizons are approximate by design.** `2y` is 730.5 days. For a study design
with day-exact boundaries, express the horizon in days.

**Nothing is fixed for you.** The report tells you what is wrong with the
dataset; changing the pipeline is your job.

## Documentation

- [Getting started](docs/getting-started.md)
- [The check reference](docs/checks/index.md) — rationale, query, and worked example per check
- [Task specs](docs/task-spec.md)
- [Reading a report](docs/reports.md)
- [Writing a custom check](docs/custom-checks.md)
- [Decisions and deviations](docs/decisions.md) — what was verified, what deviates from the spec, and why

Build the site locally with `uv run mkdocs serve`.

## Related work

- **[MEDS](https://github.com/Medical-Event-Data-Standard/meds)** — the event
  format ehrlint reads natively.
- **ACES** — cohort *extraction* in the MEDS ecosystem. It helps you define a
  task; ehrlint audits the dataset that came out.
- **[clinval](https://github.com/DanPalBCM/clinval)** — validation for clinical
  LLM extraction, where ehrlint is validation for structured prediction tasks.
- **[fhirground](https://github.com/DanPalBCM/fhirground)** — grounds extracted
  mentions into FHIR resources, and exports predictions in clinval's format.

## Citation

See [CITATION.cff](CITATION.cff).

## License

Apache-2.0.
