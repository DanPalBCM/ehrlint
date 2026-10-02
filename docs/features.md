# Feature-matrix auditing

Supply the matrix you actually trained on and LK001 becomes a different check.

```bash
ehrlint audit --data meds/ --task task.yaml --features features.parquet
```

## Why it changes the answer

Without a matrix, ehrlint can see only the event stream. Post-index events
exist in every dataset with follow-up — outcomes live there — so reporting them
as an error would make LK001 permanently red and teach you to ignore it. That
branch is therefore an `info` observation:

```text
120 of 120 subject(s) have 314 event(s) after their index date. That is
expected -- outcomes live there. ehrlint cannot tell which events your feature
pipeline consumed; supply --features to check the matrix directly.
```

With a matrix that carries an as-of timestamp, the question becomes answerable:
did a feature row get stamped after the moment it is supposed to predict from?
That is an `error`, because it is not ambiguous.

```text
LK001 post_index_features_matrix  [error]
  5 feature row(s) are stamped after the index date, so the matrix itself
  contains future information.

  example rows:
    subject_id  prediction_time           index_date
    S000018     2018-09-29 04:53:47.77    2018-07-01 04:53:47.77
```

## What the matrix needs

| column | how it is found | required |
| --- | --- | --- |
| subject id | `subject_id`, `patient_id`, `person_id` | yes |
| as-of time | `prediction_time`, `index_date`, `as_of` | for LK001 |
| label | `label`, `y`, `outcome`, `target`, `boolean_value` | no |
| features | everything else | — |

Case-insensitive, and both identifier columns can be named explicitly:

```python
matrix = el.read_feature_matrix(
    "features.parquet",
    subject_column="mrn_surrogate",
    time_column="feature_as_of",
)
```

If no subject column can be found, that is an error listing the names it looked
for. Guessing one would risk auditing a column that is not the subject id,
which produces a confident answer to the wrong question.

**Without an as-of column, LK001's matrix branch cannot run** and says so. A
matrix with no timestamp is not evidence either way: the rows may well have
been cut at the index date, and ehrlint has no way to tell. Carrying an
explicit as-of column is the cheapest thing you can do to make your own
pipeline auditable.

## A practical note

The most common way a matrix ends up leaky is not a bug in the join but a
window: `last_a1c` computed as "the most recent A1c for this patient" rather
than "the most recent A1c before the index date". The resulting feature is
usually the model's strongest predictor, and it is entirely unavailable at
inference.

If the matrix passes LK001 and the model's top feature still looks too good,
the next thing to check is the feature *definition* rather than the data —
which is outside what ehrlint can see. See [limitations](limitations.md).
