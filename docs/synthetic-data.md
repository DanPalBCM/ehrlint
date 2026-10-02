# Synthetic data and injectors

`ehrlint.synth` generates clean EHR data. `ehrlint.inject` plants specific
leaks into it. Together they are how every check in this project is tested
against ground truth rather than against an opinion.

```bash
ehrlint synth --out clean/ --n-subjects 5000 --seed 0
ehrlint synth --out leaky/ --n-subjects 5000 --seed 0 --inject LK001,LK004
```

Same arguments, same bytes.

## Clean by construction

The generated baseline is clean on purpose:

- every predictor event predates its subject's index date
- every negative is observed for the whole horizon
- splits are disjoint and time-ordered
- every positive's outcome falls inside the horizon
- the task spec it satisfies weakens no check

That matters because the test strategy is differential: generate clean data,
assert no check fires, plant one leak, assert exactly that check fires. A
baseline that already leaked would make every injector test meaningless.

"Clean by construction" also has to hold for *every* seed, not most of them.
A property test found that it did not: gender × race × birth decade × postal
code gives about 280,000 demographic profiles, which sounds ample until the
birthday paradox is applied. At 87 subjects a chance collision happened in
roughly one dataset in seventy, and a chance collision made LK009 fire on data
documented as clean. Each subject now gets a unique high-cardinality fact —
standing in for the exact birth date deterministic matching actually keys on —
alongside a repeating postal code for realism.

## No patient data, structurally

Every value comes from one of three places: a fixed module-level vocabulary, a
sequential subject index, or an arithmetic offset from a constant epoch
(`2018-01-01`). **No free text is generated at all**, so there is no note text
to de-identify.

The argument is structural rather than an inspection: nothing is sampled from a
real record, so there is no real record to leak. Tests assert that property —
every code comes from the declared vocabulary, every subject id matches
`S\d{6}`, every timestamp falls inside a window the epoch and the generator's
constants fix — and CI fails on any data file committed outside `examples/`.

Every written dataset carries a manifest:

```json
{
  "generator": "ehrlint.synth",
  "n_subjects": 120,
  "note": "Every record here is synthetic. Subjects are sequential, codes come
           from a fixed vocabulary, and timestamps derive from a constant epoch.
           No real patient record was involved."
}
```

A pre-commit hook and a CI job both require that note. A committed data
directory with no provenance statement is exactly what a reviewer of a clinical
repository should be suspicious of.

## The injectors

One per check:

| injector | what it plants |
| --- | --- |
| `inject_lk001` | a feature matrix whose rows postdate the index date |
| `inject_lk002` | an outcome code inside the index encounter |
| `inject_lk003` | the declared proxy before the index date |
| `inject_lk004` | a negative's follow-up truncated well inside the horizon |
| `inject_lk005` | test subjects copied into the train split |
| `inject_lk006` | the earliest train subjects moved into the test split |
| `inject_lk007` | negatives relabelled positive with no in-window outcome |
| `inject_lk008` | a train and a test subject sharing an encounter id |
| `inject_lk009` | a subject duplicated under a second id |
| `inject_lk010` | an outcome occurrence beyond the horizon |

Each returns the mutated dataset **and the subject ids it touched**:

```python
from ehrlint.inject import inject
from ehrlint.synth.generator import generate

injection = inject(generate(120, seed=0), "LK004", n=5, seed=0)
injection.subjects      # ['S000006', 'S000060', ...] — the ground truth
injection.description   # what was done, for a test's failure message
```

That subject list is the point. A test that only checks "did LK004 fire" passes
for a check that fires on everyone. Asserting that the finding names the
planted subjects is what makes the test mean something.

## Composing leaks

```python
from ehrlint.inject import inject_many

dataset, planted = inject_many(generate(200, seed=0), ["LK002", "LK004"], n=5, seed=0)
planted   # {'LK002': [...], 'LK004': [...]}
```

`inject_many` plants on **disjoint** subjects, and that is a fix rather than a
nicety. Every injector is seeded identically and so chose the same people —
and then one leak masked another: LK010 gives a negative an outcome long after
the horizon, which extended the follow-up LK004 had just truncated, and LK004
went quiet. A dataset built to be ground truth cannot have that in it.

If the subject pool runs out, that is an error naming how many subjects were
left. A "leak" that was never planted would make a duality test pass vacuously.

## The duality property

SPEC §11 requires it: every check has an injector, and an injected leak is
always caught. The property test asserts that over a range of seeds and cohort
sizes, and over all ten leaks planted at once:

```python
@given(check_id=st.sampled_from(ALL_IDS), seed=..., n_subjects=...)
def test_an_injected_leak_is_always_caught(check_id, seed, n_subjects):
    injection = inject(generate(n_subjects, seed=seed), check_id, n=5, seed=seed)
    report = run_audit(make_context(injection.dataset), [check_id])
    finding = report.finding(check_id)
    assert finding is not None and finding.fired
    assert set(injection.subjects) <= set(finding.subjects)
```

And the converse, which matters just as much — a check that fires on everything
satisfies the first half trivially:

```python
def test_clean_data_fires_nothing_and_skips_nothing(seed, n_subjects):
    report = run_audit(make_context(generate(n_subjects, seed=seed)))
    assert report.skipped == []
    assert report.fired == []
```

## Writing your own injector

For a [custom check](custom-checks.md), write the injector first. It forces you
to say exactly what pattern you mean, and it gives you the ground truth to test
against:

```python
def inject_my_leak(dataset, *, n=5, seed=0, exclude=None):
    rng = random.Random(seed)
    out = dataset.copy()
    chosen = _pick(out, rng, n, exclude)
    # ... mutate out.events / out.labels / out.splits for `chosen` ...
    return Injection(
        dataset=out,
        check_id="LK101",
        subjects=chosen,
        description=f"planted my leak into {len(chosen)} subject(s)",
    )
```

Take an `exclude` argument. Without it your injector will choose the same
subjects as every other one, and composing it with another leak can hide
either.
