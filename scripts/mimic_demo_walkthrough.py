#!/usr/bin/env python
"""Audit the MIMIC-IV clinical database demo with ehrlint.

Run it with::

    uv run python scripts/mimic_demo_walkthrough.py

This is the one part of ehrlint that touches real patient records, and it is
why nothing it downloads is committed:

* The **MIMIC-IV demo** (v2.2) is a 100-patient subset published openly under
  the ODC Open Database License. No PhysioNet credentials are needed -- that
  was verified against the live server, not assumed -- which is what makes this
  walkthrough runnable by anyone.
* It still contains de-identified records of real people. The files download at
  runtime into ``mimic_demo/``, which is gitignored, and **no test depends on
  this script**. The test suite makes no network calls.
* The full MIMIC-IV is credentialed. Nothing here applies to it beyond the
  mapping below, which you are free to reuse.

The mapping from MIMIC's own schema to ehrlint's canonical event table lives in
this script rather than in the package. MEDS and OMOP are the formats ehrlint
supports; MIMIC is an example of writing a converter for a third, and keeping
it here makes that distinction honest.

Citation: Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., &
Mark, R. (2023). MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet.
https://doi.org/10.13026/dp1f-ex47
"""

from __future__ import annotations

import gzip
import io
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import polars as pl

import ehrlint as el

BASE = "https://physionet.org/files/mimic-iv-demo/2.2"
HERE = Path(__file__).resolve().parents[1]
WORK = HERE / "mimic_demo"
OUT = HERE / "demo_output" / "mimic"

#: Only what the audit needs. The demo holds far more.
FILES: tuple[str, ...] = (
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/labevents.csv.gz",
    "LICENSE.txt",
)

#: ICD codes for type 2 diabetes in both vocabularies the demo uses. A
#: diagnosis outcome rather than readmission, deliberately: with readmission,
#: the outcome code *is* the index code, so the index admission is itself an
#: outcome occurrence in the index encounter and LK002 fires on every subject.
#: That is a correct finding about a circular task definition and a useless
#: demonstration, since it would happen on any dataset.
OUTCOME_CODES: tuple[str, ...] = ("25000", "E119")

#: The task audited below. It is defined from the demo's own tables without a
#: vocabulary download, not because it is the most interesting clinical
#: question.
TASK = """
task: mimic_demo_t2d_1y
description: >-
  A type 2 diabetes diagnosis within one year of the first hospital admission,
  in the MIMIC-IV demo. Defined here only to give ehrlint something concrete to
  audit.
index:
  event: ADMISSION
  code_system: MIMIC_ADMISSION
  occurrence: first
outcome:
  codes:
    - system: ICD9CM
      values: ["25000"]
    - system: ICD10CM
      values: [E119]
  window: [index_date, index_date + 1y]
  competing_codes:
    - system: null
      values: [MEDS_DEATH]
horizon: 1y
followup:
  require_full_horizon_for_negatives: true
  allow_death_as_censoring: true
splits:
  strategy: subject
  temporal: true
  group_column: encounter_id
"""


def fetch(relative: str) -> Path:
    """Download one file into `mimic_demo/`, skipping it if already present."""
    target = WORK / relative
    if target.exists() and target.stat().st_size > 0:
        print(f"  have  {relative}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{relative}"
    print(f"  get   {relative}", end="", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print()
        raise SystemExit(
            f"could not download {url}: {exc}\n"
            "The MIMIC-IV demo is openly downloadable and needs no credentials, so "
            "this is most likely a network problem on this machine."
        ) from exc
    target.write_bytes(payload)
    print(f"  ({len(payload):,} bytes)")
    return target


def read_csv_gz(path: Path) -> pl.DataFrame:
    """Read a gzipped MIMIC CSV with every column as text.

    Everything is read as a string and cast explicitly afterwards. MIMIC's
    identifier columns are numeric-looking but are identifiers, and letting a
    schema inference turn `subject_id` into an integer here is the first step
    towards the silent empty join that makes an audit look clean.
    """
    with gzip.open(path, "rb") as handle:
        return pl.read_csv(io.BytesIO(handle.read()), infer_schema_length=0)


def to_canonical() -> pl.DataFrame:
    """Map the MIMIC demo tables onto ehrlint's canonical event table."""
    patients = read_csv_gz(WORK / "hosp/patients.csv.gz")
    admissions = read_csv_gz(WORK / "hosp/admissions.csv.gz")
    diagnoses = read_csv_gz(WORK / "hosp/diagnoses_icd.csv.gz")
    labs = read_csv_gz(WORK / "hosp/labevents.csv.gz")

    frames: list[pl.DataFrame] = []

    # --- admissions: the index event, and the outcome -------------------
    # `admittime` throughout, never `dischtime`. A discharge time postdates
    # everything that happened in the visit, so indexing on it would shift the
    # prediction moment past the information it is meant to exclude.
    frames.append(
        admissions.select(
            pl.col("subject_id").alias("subject_id"),
            pl.col("admittime").str.to_datetime(strict=False).alias("time"),
            pl.lit("ADMISSION").alias("code"),
            pl.lit("MIMIC_ADMISSION").alias("code_system"),
            pl.lit(None, dtype=pl.Float64).alias("numeric_value"),
            pl.col("admission_type").alias("text_value"),
            pl.col("hadm_id").alias("encounter_id"),
        )
    )

    # --- deaths: a competing event, so LK004 does not call them negatives
    if "deathtime" in admissions.columns:
        frames.append(
            admissions.filter(pl.col("deathtime").is_not_null()).select(
                pl.col("subject_id"),
                pl.col("deathtime").str.to_datetime(strict=False).alias("time"),
                pl.lit("MEDS_DEATH").alias("code"),
                pl.lit(None, dtype=pl.String).alias("code_system"),
                pl.lit(None, dtype=pl.Float64).alias("numeric_value"),
                pl.lit(None, dtype=pl.String).alias("text_value"),
                pl.col("hadm_id").alias("encounter_id"),
            )
        )

    # --- diagnoses: dated by their admission, which is the catch ---------
    # `diagnoses_icd` carries no timestamp of its own. The only date available
    # is the admission's, so every diagnosis for a visit is stamped at
    # `admittime` -- including the ones coded at discharge. That is exactly the
    # pattern LK002 exists to find, and it is real rather than contrived: this
    # is how most EHR extracts date their diagnosis codes.
    admit_times = admissions.select(
        "hadm_id", pl.col("admittime").str.to_datetime(strict=False).alias("admit_time")
    )
    frames.append(
        diagnoses.join(admit_times, on="hadm_id", how="inner").select(
            pl.col("subject_id"),
            pl.col("admit_time").alias("time"),
            pl.col("icd_code").str.strip_chars().alias("code"),
            pl.when(pl.col("icd_version") == "9")
            .then(pl.lit("ICD9CM"))
            .otherwise(pl.lit("ICD10CM"))
            .alias("code_system"),
            pl.lit(None, dtype=pl.Float64).alias("numeric_value"),
            pl.lit(None, dtype=pl.String).alias("text_value"),
            pl.col("hadm_id").alias("encounter_id"),
        )
    )

    # --- labs: genuinely timestamped -----------------------------------
    frames.append(
        labs.select(
            pl.col("subject_id"),
            pl.col("charttime").str.to_datetime(strict=False).alias("time"),
            pl.col("itemid").alias("code"),
            pl.lit("MIMIC_LABITEM").alias("code_system"),
            pl.col("valuenum").cast(pl.Float64, strict=False).alias("numeric_value"),
            pl.col("value").alias("text_value"),
            pl.col("hadm_id").alias("encounter_id"),
        )
    )

    # --- demographics: static, with a null time ------------------------
    # MIMIC shifts dates per patient for de-identification, so `anchor_year`
    # is not a real year. It is still a per-patient constant, which is what
    # LK009's signature needs.
    for column, system in (
        ("gender", "MIMIC_GENDER"),
        ("anchor_age", "MIMIC_ANCHOR_AGE"),
        ("anchor_year_group", "MIMIC_ANCHOR_YEAR_GROUP"),
    ):
        if column not in patients.columns:
            continue
        frames.append(
            patients.select(
                pl.col("subject_id"),
                pl.lit(None, dtype=pl.Datetime("us")).alias("time"),
                pl.col(column).alias("code"),
                pl.lit(system).alias("code_system"),
                pl.lit(None, dtype=pl.Float64).alias("numeric_value"),
                pl.lit(None, dtype=pl.String).alias("text_value"),
                pl.lit(None, dtype=pl.String).alias("encounter_id"),
            ).drop_nulls("code")
        )

    combined = pl.concat(frames, how="vertical_relaxed")
    return el.coerce_events(combined, source="mimic-iv-demo")


def build_labels(events: pl.DataFrame) -> pl.DataFrame:
    """A diabetes label: an outcome code within a year of the first admission.

    Written out deliberately rather than hidden in a helper, because the audit
    below is an audit of *this* definition. Note the upper bound: dropping it
    is the mistake LK010 exists to catch, and it is one line away.
    """
    admissions = (
        events.filter(pl.col("code") == "ADMISSION")
        .select("subject_id", "time")
        .drop_nulls()
        .sort(["subject_id", "time"])
    )
    first = admissions.group_by("subject_id").agg(pl.col("time").min().alias("index_date"))

    outcomes = (
        events.filter(pl.col("code").is_in(list(OUTCOME_CODES)))
        .select("subject_id", "time")
        .drop_nulls()
    )
    positive = (
        outcomes.join(first, on="subject_id", how="inner")
        .filter(
            (pl.col("time") >= pl.col("index_date"))
            & ((pl.col("time") - pl.col("index_date")).dt.total_days() <= 365.25)
        )
        .select("subject_id")
        .unique()
        .with_columns(pl.lit(True).alias("label"))
    )
    return (
        first.join(positive, on="subject_id", how="left")
        .with_columns(pl.col("label").fill_null(False))
        .sort("subject_id")
    )


def build_splits(labels: pl.DataFrame) -> dict[str, list[str]]:
    """A temporal 60/20/20 split by subject, on the index date."""
    ordered = labels.sort("index_date")["subject_id"].to_list()
    n = len(ordered)
    n_train, n_valid = int(n * 0.6), int(n * 0.2)
    return {
        "train": sorted(ordered[:n_train]),
        "valid": sorted(ordered[n_train : n_train + n_valid]),
        "test": sorted(ordered[n_train + n_valid :]),
    }


def main() -> int:
    print("ehrlint MIMIC-IV demo walkthrough")
    print(
        "The MIMIC-IV demo is 100 de-identified patients, published openly under the\n"
        "ODbL. It downloads into mimic_demo/, which is gitignored; nothing from it is\n"
        "committed and no test depends on it.\n"
    )

    print(f"[1/5] Download into {WORK.name}/")
    WORK.mkdir(parents=True, exist_ok=True)
    for relative in FILES:
        fetch(relative)
    print(f"  license at {(WORK / 'LICENSE.txt').relative_to(HERE)}")

    print("\n[2/5] Map MIMIC's schema onto the canonical event table")
    events = to_canonical()
    print(
        f"  {events.height:,} events, {events['subject_id'].n_unique()} subjects, "
        f"{events['time'].null_count():,} static"
    )
    for system, count in sorted(
        events.group_by("code_system").len().iter_rows(), key=lambda r: -r[1]
    ):
        print(f"    {system!s:<26} {count:,}")

    print("\n[3/5] Define a task and a label, then split")
    task_path = WORK / "task.yaml"
    task_path.write_text(TASK.strip() + "\n")
    labels = build_labels(events)
    splits = build_splits(labels)
    print(
        f"  {labels.height} subjects, {int(labels['label'].sum())} with a "
        "diabetes code within 1y of their first admission"
    )
    print("  splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    print("\n[4/5] Audit")
    ctx = el.AuditContext.from_frames(events, task_spec=task_path, labels=labels, splits=splits)
    report = el.run_audit(ctx)

    for warning in report.warnings:
        print(f"  !  {warning}")
    if report.skipped:
        print(f"\n  {len(report.skipped)} check(s) did not run. A skip is not a pass:")
        for skip in report.skipped:
            print(f"     {skip.check_id}  {skip.reason}")
    print()
    if report.fired:
        print(f"  {'check':<32} {'severity':<9} {'subjects':>8} {'rate':>7}")
        for finding in report.fired:
            rate = "-" if finding.subject_rate is None else f"{finding.subject_rate * 100:.1f}%"
            label = f"{finding.check_id} {finding.name}"
            print(f"  {label:<32} {finding.severity.value:<9} {finding.n_subjects:>8} {rate:>7}")
    else:
        print("  nothing fired")

    print("\n[5/5] Write the report")
    written = report.write_all(OUT)
    for name, path in sorted(written.items()):
        print(f"  {name:<9} {path.relative_to(HERE)}")

    print(f"\nexit code {report.exit_code}. Open {written['html'].relative_to(HERE)} in a browser.")
    print(
        "\nWhat to expect, and why it is worth seeing on real data: MIMIC's\n"
        "`diagnoses_icd` table carries no timestamp of its own, so every diagnosis is\n"
        "dated by its admission -- including codes entered at discharge. LK002 finds\n"
        "that, and it is not a contrived example: it is how most EHR extracts date\n"
        "their diagnosis codes."
    )
    print(f"\nTo remove the downloaded data: rm -rf {WORK.relative_to(HERE)}")
    return 0


def clean() -> int:
    """Delete the downloaded data."""
    if WORK.exists():
        shutil.rmtree(WORK)
        print(f"removed {WORK}")
    else:
        print(f"{WORK} does not exist")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        raise SystemExit(clean())
    raise SystemExit(main())
