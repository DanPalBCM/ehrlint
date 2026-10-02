# Limitations

Stated plainly, because a tool that audits other people's claims should be
honest about its own.

## A clean audit does not mean a valid task

ehrlint checks ten known leakage patterns. It knows nothing about whether your
cohort answers your clinical question, whether your outcome definition is the
accepted one in your field, whether your index date is the clinically right
moment, or whether the patients you excluded should have been.

Every report says so in its own closing section. A green ehrlint run is a
necessary condition for a defensible task, not a sufficient one.

## It cannot see your feature pipeline

Without `--features`, LK001 can only report that post-index events exist —
which is true of every dataset with follow-up. Whether your pipeline
*consumed* them is not visible in the event stream. That is why the
event-level branch is an observation rather than an error.

Even with a matrix, ehrlint sees the values and the as-of timestamp, not the
code that produced them. A feature defined as "the most recent A1c for this
patient" rather than "the most recent A1c before the index date" can pass LK001
if the matrix happens to be built from a correctly-cut extract, and fail in
production when it is not.

## Proxies and competing events are yours to declare

LK003 looks for the codes you list in `proxy_codes`. ehrlint cannot infer that
an insulin drip implies DKA in your cohort, or that a dialysis code implies
end-stage renal disease in yours — and guessing would produce findings nobody
could verify. The same applies to `competing_codes`: without them, every death
in the cohort looks like an unobserved negative to LK004.

A thin task spec therefore produces a thin audit. `ehrlint validate-task` lists
what a spec weakens, which is the best ehrlint can do about it.

## Duplicate detection is deterministic, not probabilistic

LK009 matches on the exact set of static demographic facts. It will miss
duplicates that differ by a typo, a transposed birth date, or a changed
surname. It also **refuses to report anything** when the demographics in your
data distinguish too few profiles for a collision to mean something — at that
point a collision is arithmetic rather than evidence, and reporting it would be
the fabricated statistic this project forbids.

A probabilistic matcher would find more duplicates and fewer checkable ones.
That trade was made deliberately in the other direction.

## Horizons are approximate by design

`2y` is 730.5 days and `6m` is 182.625. For a study design with day-exact
boundaries, express the horizon in days. The approximation is the right one for
a *horizon* — "within two years of the index date" is a design choice with a
tolerance of days — but it is an approximation.

## Several checks skip more often than you would like

LK002 and LK008 need an encounter id, which plain MEDS does not have. LK003
needs declared proxies. LK010 needs a horizon. LK007 needs labels. On a minimal
input, half the battery can skip.

The report is loud about this: skips are counted separately, printed above the
findings, and each one states its reason. But a skip is still a check that did
not run, and the exit code cannot express that. If you gate on ehrlint, gate on
the skip count too.

## It is a structural audit, not a statistical one

No fairness metrics, no calibration analysis, no distribution-shift detection,
no causal-inference checks. Those are different tools, and conflating them
would make this one worse at its job.

## Nothing is fixed for you

The report tells you what is wrong with the dataset. Changing the pipeline is
your job — and in most cases the fix reduces your cohort or your positive
count, which is the whole point: the number was wrong before.

## v1 does not

- apply the `exclusion` criteria in a task spec (they are recorded for provenance)
- read event formats beyond MEDS-like and OMOP CDM
- monitor a training pipeline at runtime
- de-identify anything
