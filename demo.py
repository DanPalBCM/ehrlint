#!/usr/bin/env python
"""ehrlint demo: audit two datasets in this repository, one clean and one leaky.

Run it with::

    uv sync
    uv run python demo.py

Everything it reads lives in ``examples/`` and was produced by ``ehrlint.synth``,
a deterministic generator with a fixed code vocabulary and sequential subject
ids. No real patient record was involved at any point, and nothing here reaches
the network.

The demo audits the clean cohort first, because a leakage auditor that fires on
everything is useless and the clean pass is what makes the leaky one mean
something. It then audits a cohort with four planted leaks and checks the
findings against the ground truth recorded in
``examples/leaky_meds/planted_leaks.json`` -- so the demo verifies the tool
rather than just running it.

Output lands in ``demo_output/``. Open ``demo_output/leaky/report.html``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ehrlint as el

HERE = Path(__file__).parent
CLEAN = HERE / "examples" / "clean_meds"
LEAKY = HERE / "examples" / "leaky_meds"
OUT = HERE / "demo_output"

N_STEPS = 6

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[2m",
    "\033[1m",
    "\033[0m",
)
if not sys.stdout.isatty():  # keep piped output clean
    GREEN = YELLOW = RED = DIM = BOLD = RESET = ""


def step(n: int, title: str) -> None:
    print(f"\n{BOLD}[{n}/{N_STEPS}] {title}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}   {msg}")


def bad(msg: str) -> None:
    print(f"  {RED}FAIL{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET}    {msg}")


def note(msg: str) -> None:
    print(f"       {DIM}{msg}{RESET}")


def load(root: Path) -> el.AuditContext:
    """Build a context from one of the committed MEDS datasets."""
    features = root / "features.parquet"
    return el.AuditContext.from_meds(
        root,
        task_spec=root / "task.yaml",
        labels_path=root / "labels.parquet",
        splits_path=root / "splits.json",
        features_path=features if features.exists() else None,
    )


def describe(ctx: el.AuditContext) -> None:
    note(
        f"{ctx.n_subjects} subjects, {ctx.events.height} events "
        f"({el.schemas.timed_events(ctx.events).height} timed), "
        f"task {ctx.task.task!r}, horizon {ctx.task.horizon}"
    )


def main() -> int:
    print(f"{BOLD}ehrlint {el.__version__}{RESET} -- lint your prediction task before you train")
    print(f"{DIM}All data below is synthetic. No real patient record is involved.{RESET}")

    if not CLEAN.exists() or not LEAKY.exists():
        bad(f"the example data is missing; expected {CLEAN} and {LEAKY}")
        return 1

    # ------------------------------------------------------------------
    step(1, "Validate the task spec before looking at any data")
    task = el.TaskSpec.from_yaml(CLEAN / "task.yaml")
    ok(f"task {task.task!r} parsed")
    note(f"index on {task.index.event!r}, outcome from {len(task.outcome.all_codes())} codes")
    weaknesses = task.warnings()
    if weaknesses:
        for line in weaknesses:
            warn(line)
    else:
        ok("nothing in this spec weakens a check")

    # ------------------------------------------------------------------
    step(2, "Audit the clean cohort -- the baseline that makes the rest mean something")
    clean_ctx = load(CLEAN)
    describe(clean_ctx)
    clean = el.run_audit(clean_ctx)

    if clean.skipped:
        bad(f"{len(clean.skipped)} check(s) skipped on the clean cohort")
        for skip in clean.skipped:
            note(f"{skip.check_id}: {skip.reason}")
        return 1
    ok(f"all {clean.summary.n_checks_run} checks ran; none skipped")

    if clean.fired:
        bad("a check fired on data that is clean by construction")
        for finding in clean.fired:
            note(f"{finding.check_id}: {finding.message}")
        return 1
    ok(f"nothing fired, exit code {clean.exit_code}")
    note("a leakage auditor that fires on everything tells you nothing")

    for finding in clean.findings:
        if finding.severity is el.Severity.INFO:
            note(f"observation: {finding.message}")

    # ------------------------------------------------------------------
    step(3, "Audit a cohort with four planted leaks")
    planted: dict[str, list[str]] = json.loads((LEAKY / "planted_leaks.json").read_text())
    note("planted: " + ", ".join(f"{k} on {len(v)} subject(s)" for k, v in sorted(planted.items())))

    leaky_ctx = load(LEAKY)
    describe(leaky_ctx)
    leaky = el.run_audit(leaky_ctx)
    print()
    print(f"  {'check':<34} {'severity':<9} {'subjects':>8} {'rate':>7}")
    for finding in leaky.fired:
        rate = "-" if finding.subject_rate is None else f"{finding.subject_rate * 100:.1f}%"
        label = f"{finding.check_id} {finding.name}"
        print(f"  {label:<34} {finding.severity.value:<9} {finding.n_subjects:>8} {rate:>7}")
    print()
    ok(
        f"{leaky.summary.n_errors} error(s), {leaky.summary.n_warnings} warning(s), "
        f"exit code {leaky.exit_code}"
    )

    # ------------------------------------------------------------------
    step(4, "Check the findings against the ground truth")
    note("the leaks were planted by ehrlint.inject, so the right answer is known")
    failures = 0
    for check_id, subjects in sorted(planted.items()):
        finding = leaky.finding(check_id)
        if finding is None or not finding.fired:
            bad(f"{check_id} did not fire on its own planted leak")
            failures += 1
            continue
        missed = sorted(set(subjects) - set(finding.subjects))
        if missed:
            bad(f"{check_id} fired but missed {len(missed)} planted subject(s): {missed[:3]}")
            failures += 1
            continue
        ok(f"{check_id} found all {len(subjects)} planted subject(s)")
    if failures:
        return 1

    # ------------------------------------------------------------------
    step(5, "Look at the evidence behind one finding")
    note("SPEC rule: every finding references the rows that triggered it")
    example = next((f for f in leaky.fired if f.examples), None)
    if example is None:
        bad("no finding carried example rows -- that is a bug, not a subtle result")
        return 1

    print(f"       {BOLD}{example.check_id} {example.name}{RESET}: {example.message}")
    print(f"\n       {DIM}the query that found it{RESET}")
    for line in example.query.splitlines():
        print(f"         {DIM}{line}{RESET}")
    print(f"\n       {DIM}rows from the data, not aggregates{RESET}")
    columns = example.example_columns[:5]
    print("         " + "  ".join(f"{c:<22}" for c in columns))
    for row in example.examples[:4]:
        print("         " + "  ".join(f"{str(row.get(c, ''))[:22]:<22}" for c in columns))

    # ------------------------------------------------------------------
    step(6, "Write the reports")
    clean_written = clean.write_all(OUT / "clean")
    leaky_written = leaky.write_all(OUT / "leaky")
    for label, written in (("clean", clean_written), ("leaky", leaky_written)):
        for name, path in sorted(written.items()):
            size = path.stat().st_size
            print(f"  {label:<6} {name:<9} {path.relative_to(HERE)}  {DIM}{size:,} bytes{RESET}")

    html = leaky_written["html"]
    print(
        f"\n{BOLD}Done.{RESET} The clean cohort passed; the leaky one failed with "
        f"{leaky.summary.n_errors} error(s), and every finding was checked against "
        "the planted ground truth."
    )
    print(
        f"Open {BOLD}{html.relative_to(HERE)}{RESET} in a browser -- it is a single "
        "self-contained file."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
