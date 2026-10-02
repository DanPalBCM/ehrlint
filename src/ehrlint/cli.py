"""The ``ehrlint`` command line.

Three commands, matching SPEC §5.1: ``audit``, ``synth``, ``validate-task``.

Exit codes are the contract, because the main use of this tool is a CI gate:

* ``0`` -- nothing fired at or above ``--fail-on``.
* ``1`` -- something did.
* ``2`` -- usage or input error: a missing file, an unreadable task spec, an
  unknown check id. Kept distinct from ``1`` so a broken invocation can never
  be mistaken for a clean audit, which is the failure mode that would quietly
  disable the gate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ehrlint import __version__
from ehrlint.exceptions import EhrlintError
from ehrlint.schemas.findings import Severity

#: Exit code for a usage or input error (SPEC §5.1).
EXIT_USAGE = 2

app = typer.Typer(
    name="ehrlint",
    help=(
        "Lint your clinical prediction task before you train on it. "
        "Audits a finished EHR dataset for the leakage patterns that make "
        "clinical models look good in a paper and fail in deployment."
    ),
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True)

SEVERITY_STYLE = {"error": "bold red", "warning": "yellow", "info": "cyan"}


def _fail(message: str, *, code: int = EXIT_USAGE) -> NoReturn:
    """Print an error and exit.

    The message is escaped: findings and paths contain square brackets, and
    rich would otherwise read ``[path:12]`` as markup and swallow the line --
    an error that prints nothing is worse than no error handling at all.
    """
    err_console.print(f"[bold red]error[/bold red] {escape(message)}")
    raise typer.Exit(code)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"ehrlint {__version__}")
        raise typer.Exit(0)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show the version."
        ),
    ] = False,
) -> None:
    """ehrlint: a leakage auditor for EHR prediction tasks."""


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


@app.command()
def audit(
    data: Annotated[
        Path,
        typer.Option("--data", help="Dataset directory (MEDS layout, or OMOP with --format omop)."),
    ],
    task: Annotated[Path, typer.Option("--task", help="Task spec YAML.")],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Directory for report.html, report.md, report.json."),
    ] = None,
    data_format: Annotated[
        str, typer.Option("--format", help="Input format: 'meds' or 'omop'.")
    ] = "meds",
    splits: Annotated[
        Path | None,
        typer.Option("--splits", help="Splits JSON or parquet. Defaults to the dataset's own."),
    ] = None,
    labels: Annotated[
        Path | None,
        typer.Option("--labels", help="Labels parquet (subject_id, prediction_time, label)."),
    ] = None,
    features: Annotated[
        Path | None,
        typer.Option("--features", help="Training feature matrix, to check it directly."),
    ] = None,
    checks: Annotated[
        str | None,
        typer.Option("--checks", help="Comma-separated check ids. Default: all of them."),
    ] = None,
    fail_on: Annotated[
        str, typer.Option("--fail-on", help="Severity that fails the run: 'error' or 'warning'.")
    ] = "error",
    quiet: Annotated[
        bool, typer.Option("--quiet", help="Print nothing but the exit code.")
    ] = False,
) -> None:
    """Audit a dataset against a task spec.

    Exits 0 when nothing fired at or above --fail-on, 1 when something did,
    and 2 on a usage or input error.
    """
    from ehrlint.audit import run_audit
    from ehrlint.context import AuditContext

    threshold = _parse_fail_on(fail_on)

    requested = _parse_checks(checks)

    if not data.exists():
        _fail(f"--data does not exist: {data}")
    if not task.exists():
        _fail(f"--task does not exist: {task}")
    for label, path in (("--splits", splits), ("--labels", labels), ("--features", features)):
        if path is not None and not path.exists():
            _fail(f"{label} does not exist: {path}")

    fmt = data_format.strip().lower()
    if fmt not in {"meds", "omop"}:
        _fail(f"--format must be 'meds' or 'omop'; got {data_format!r}")

    try:
        loader = AuditContext.from_meds if fmt == "meds" else AuditContext.from_omop
        ctx = loader(
            data,
            task_spec=task,
            splits_path=splits,
            labels_path=labels,
            features_path=features,
        )
    except EhrlintError as exc:
        _fail(str(exc))

    try:
        report = run_audit(ctx, requested, fail_on=threshold)
    except EhrlintError as exc:
        _fail(str(exc))

    if out is not None:
        from ehrlint.report.builder import write_report

        try:
            written = write_report(report, out)
        except EhrlintError as exc:
            _fail(str(exc))
        if not quiet:
            console.print()
            for name, path in written.items():
                console.print(f"  wrote {name:<9} {escape(str(path))}")

    if not quiet:
        _print_audit(report)

    raise typer.Exit(report.exit_code)


def _parse_fail_on(raw: str) -> Severity:
    """Parse ``--fail-on``, which accepts only ``error`` and ``warning``.

    ``info`` is deliberately rejected rather than accepted. Info findings are
    descriptive statistics (SPEC §4.4) -- "200 of 200 subjects have events
    after their index date" is true of every dataset with follow-up -- so they
    never fire. Accepting ``--fail-on info`` would give you a flag that behaves
    exactly like ``--fail-on warning`` while reading as if it were stricter,
    and a threshold that does not do what it says is worse than one less
    option.
    """
    value = raw.strip().lower()
    if value in {"error", "warning"}:
        return Severity(value)
    if value == "info":
        _fail(
            "--fail-on info is not accepted. Info findings are descriptive "
            "statistics, not problems, so they never fail a run; use "
            "--fail-on warning for the strictest gate"
        )
    _fail(f"--fail-on must be 'error' or 'warning'; got {raw!r}")


def _parse_checks(raw: str | None) -> list[str] | None:
    """Parse ``--checks``, failing loudly on an unknown id.

    A typo must not silently narrow the battery. Running nine checks when you
    asked for ten and exiting 0 is the exact outcome this tool exists to
    prevent.
    """
    if raw is None or not raw.strip():
        return None
    from ehrlint.checks import check_ids

    requested = [part.strip().upper() for part in raw.split(",") if part.strip()]
    if not requested:
        _fail("--checks was given but lists no check id")
    known = set(check_ids())
    unknown = [c for c in requested if c not in known]
    if unknown:
        _fail(f"unknown check id(s): {', '.join(unknown)}. Available: {', '.join(sorted(known))}")
    return requested


def _print_audit(report: object) -> None:
    """Print the console summary: skips first, then findings."""
    rep = report  # typed loosely to keep the import graph lazy
    summary = rep.summary  # type: ignore[attr-defined]

    console.print()
    console.print(
        f"[bold]ehrlint[/bold] {escape(rep.task)}  "  # type: ignore[attr-defined]
        f"{summary.n_subjects} subjects, {summary.n_events} events, "
        f"{summary.n_checks_run}/{summary.n_checks_requested} checks run"
    )

    # Skips before findings. A reader who sees "0 errors" first stops there,
    # and "0 errors out of 4 checks that ran" means something very different.
    if rep.skipped:  # type: ignore[attr-defined]
        console.print()
        console.print(
            f"[bold]{len(rep.skipped)} check(s) did not run.[/bold] "  # type: ignore[attr-defined]
            "A skipped check is not a passed check."
        )
        for skip in rep.skipped:  # type: ignore[attr-defined]
            console.print(f"  [dim]{skip.check_id}[/dim] {escape(skip.reason)}")

    if rep.warnings:  # type: ignore[attr-defined]
        console.print()
        console.print("[bold]Input warnings[/bold]")
        for warning in rep.warnings:  # type: ignore[attr-defined]
            console.print(f"  [yellow]![/yellow] {escape(warning)}")

    fired = rep.fired  # type: ignore[attr-defined]
    console.print()
    if fired:
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("check")
        table.add_column("severity")
        table.add_column("subjects", justify="right")
        table.add_column("rate", justify="right")
        table.add_column("what")
        for finding in fired:
            rate = "-" if finding.subject_rate is None else f"{finding.subject_rate * 100:.1f}%"
            style = SEVERITY_STYLE.get(finding.severity.value, "")
            table.add_row(
                f"{finding.check_id} {finding.name}",
                f"[{style}]{finding.severity.value}[/{style}]" if style else finding.severity.value,
                str(finding.n_subjects),
                rate,
                escape(_first_sentence(finding.message)),
            )
        console.print(table)
    else:
        console.print("[green]No check fired.[/green]")

    console.print()
    if rep.ok:  # type: ignore[attr-defined]
        console.print(
            f"[green]PASS[/green] nothing at or above [bold]{rep.fail_on.value}[/bold]"  # type: ignore[attr-defined]
        )
    else:
        console.print(
            f"[red]FAIL[/red] {summary.n_errors} error(s), {summary.n_warnings} warning(s) "
            f"at threshold [bold]{rep.fail_on.value}[/bold]"  # type: ignore[attr-defined]
        )
    console.print(
        "[dim]Every finding carries the rows that produced it; see the report for them.[/dim]"
    )


def _first_sentence(text: str, limit: int = 72) -> str:
    """The first sentence of a message, truncated for the console table."""
    head = text.split(". ")[0].rstrip(".")
    return head if len(head) <= limit else head[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# synth
# ---------------------------------------------------------------------------


@app.command()
def synth(
    out: Annotated[Path, typer.Option("--out", help="Directory to write the dataset into.")],
    n_subjects: Annotated[
        int, typer.Option("--n-subjects", help="How many subjects to generate.")
    ] = 500,
    seed: Annotated[int, typer.Option("--seed", help="Seed; same seed, same bytes.")] = 0,
    inject: Annotated[
        str | None,
        typer.Option("--inject", help="Comma-separated check ids whose leak to plant."),
    ] = None,
    positive_rate: Annotated[
        float, typer.Option("--positive-rate", help="Share of subjects with the outcome.")
    ] = 0.2,
    quiet: Annotated[bool, typer.Option("--quiet", help="Print nothing on success.")] = False,
) -> None:
    """Generate synthetic data, optionally with specific leaks planted.

    The baseline is clean by construction, so `synth` with no --inject should
    audit to exit code 0. That is what makes the injected cases meaningful.
    """
    from ehrlint.inject import inject_many, injector_ids
    from ehrlint.synth.generator import generate

    if n_subjects < 1:
        _fail(f"--n-subjects must be at least 1; got {n_subjects}")
    if not 0.0 <= positive_rate <= 1.0:
        _fail(f"--positive-rate must be in [0, 1]; got {positive_rate}")

    requested: list[str] = []
    if inject and inject.strip():
        requested = [part.strip().upper() for part in inject.split(",") if part.strip()]
        available = set(injector_ids())
        unknown = [c for c in requested if c not in available]
        if unknown:
            _fail(
                f"no injector for {', '.join(unknown)}. Available: {', '.join(sorted(available))}"
            )

    try:
        dataset = generate(n_subjects, seed=seed, positive_rate=positive_rate)
        planted: dict[str, list[str]] = {}
        if requested:
            dataset, planted = inject_many(dataset, requested, seed=seed)
    except (EhrlintError, ValueError) as exc:
        _fail(str(exc))

    try:
        written = dataset.write(out)
    except OSError as exc:
        _fail(f"could not write to {out}: {exc}")

    if quiet:
        return

    console.print()
    console.print(
        f"[bold]ehrlint synth[/bold] {dataset.n_subjects} subjects, "
        f"{dataset.events.height} events, seed {seed}"
    )
    console.print(
        "[dim]Entirely synthetic: sequential ids, a fixed code vocabulary, "
        "and timestamps derived from a constant epoch.[/dim]"
    )
    if planted:
        console.print()
        console.print("[bold]Leaks planted[/bold]")
        for check_id, subjects in sorted(planted.items()):
            console.print(f"  [yellow]{check_id}[/yellow] on {len(subjects)} subject(s)")
    console.print()
    for name, path in written.items():
        console.print(f"  wrote {name:<13} {escape(str(path))}")


# ---------------------------------------------------------------------------
# validate-task
# ---------------------------------------------------------------------------


@app.command("validate-task")
def validate_task(
    spec: Annotated[Path, typer.Argument(help="Path to a task spec YAML.")],
) -> None:
    """Check a task spec without touching any data.

    Reports which checks the spec weakens: a spec that omits `proxy_codes`
    cannot support LK003, and the right time to learn that is before the
    audit, not from a skip in the report.
    """
    from ehrlint.schemas.task import TaskSpec

    if not spec.exists():
        _fail(f"task spec does not exist: {spec}")

    try:
        task = TaskSpec.from_yaml(spec)
    except EhrlintError as exc:
        _fail(str(exc))
    except Exception as exc:  # malformed YAML, wrong types
        _fail(f"could not parse {spec}: {exc}")

    console.print()
    console.print(f"[green]valid[/green] {escape(task.task)}")

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("field", style="dim")
    table.add_column("value")
    for key, value in task.describe().items():
        table.add_row(str(key), escape(str(value)))
    console.print(table)

    warnings = task.warnings()
    if warnings:
        console.print()
        console.print("[bold]This spec weakens some checks[/bold]")
        for warning in warnings:
            console.print(f"  [yellow]![/yellow] {escape(warning)}")
    else:
        console.print()
        console.print("[dim]Nothing in this spec weakens a check.[/dim]")

    raise typer.Exit(0)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


@app.command("list-checks")
def list_checks() -> None:
    """List the registered checks and what each one looks for."""
    from ehrlint.checks import all_checks

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("id")
    table.add_column("name")
    table.add_column("default")
    table.add_column("looks for")
    for check in all_checks():
        style = SEVERITY_STYLE.get(check.severity_default.value, "")
        table.add_row(
            check.id,
            check.name,
            f"[{style}]{check.severity_default.value}[/{style}]"
            if style
            else check.severity_default.value,
            escape(check.description),
        )
    console.print(table)


def run() -> None:  # pragma: no cover - thin entry point
    """Console-script entry point."""
    try:
        app()
    except EhrlintError as exc:  # a path that escaped a command's own handling
        err_console.print(f"[bold red]error[/bold red] {escape(str(exc))}")
        sys.exit(EXIT_USAGE)


if __name__ == "__main__":  # pragma: no cover
    run()
