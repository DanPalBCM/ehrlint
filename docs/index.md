# ehrlint

**Lint your clinical prediction task before you train on it.**

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

```text
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

### Evidence or nothing

Every finding references the concrete rows that triggered it — subject ids,
event times, feature names — plus the query that found them, so you can
reproduce any finding by hand.

This is enforced, not encouraged. A `Finding` carrying a non-zero subject count
and no example rows raises at construction:

```text
LK002 reports 5 affected subject(s) but carries no example rows. Every finding
must reference the concrete rows that triggered it.
```

A separate pass checks that the subjects a finding names actually exist in the
input, so a check cannot report rows it invented.

### A skipped check is not a passed check

A check that cannot run says so, with a reason:

```text
LK003 no proxy codes are declared in the task spec, so there is nothing to look
      for. Proxies are task-specific and ehrlint cannot infer them.
```

Skips are counted separately from passes in the summary, and the report prints
them **above** the findings. A clean report that quietly skipped six of ten
checks is worse than a dirty one, because it reads as reassurance.

## Where to go next

- [Getting started](getting-started.md) — install, run the demo, audit your own data
- [Concepts](concepts.md) — events, task specs, findings
- [The check battery](checks/index.md) — one page per check
- [Reading a report](reports.md)
- [Writing a custom check](custom-checks.md)
- [Limitations](limitations.md) — what ehrlint does not tell you
- [Decisions and deviations](decisions.md) — what was verified, and what deviates from the spec
