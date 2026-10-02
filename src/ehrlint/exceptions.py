"""Exception hierarchy for ehrlint.

Errors carry enough context to locate the problem: a file path, a column name,
a subject id. None of them carry patient data beyond the identifiers needed to
find the offending row -- which is the same discipline the findings follow.
"""

from __future__ import annotations


class EhrlintError(Exception):
    """Base class for every error raised by ehrlint."""


class InputError(EhrlintError):
    """Input data could not be read, or is missing something required."""

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        table: str | None = None,
        column: str | None = None,
    ) -> None:
        self.path = path
        self.table = table
        self.column = column
        bits = [b for b in (path, table and f"table={table}", column and f"column={column}") if b]
        suffix = f" [{', '.join(bits)}]" if bits else ""
        super().__init__(f"{message}{suffix}")


class SchemaError(InputError):
    """Input data does not satisfy the schema ehrlint requires."""


class TaskSpecError(EhrlintError):
    """A task specification is missing, malformed, or self-contradictory."""


class SplitsError(EhrlintError):
    """A splits definition is missing, malformed, or inconsistent with the data."""


class CheckError(EhrlintError):
    """A check could not run.

    Distinct from a check *firing*: a finding means a leak was found, this means
    the check itself failed. The audit records it as skipped with a reason
    rather than silently reporting a clean result.
    """

    def __init__(self, message: str, *, check_id: str | None = None) -> None:
        self.check_id = check_id
        prefix = f"{check_id}: " if check_id else ""
        super().__init__(f"{prefix}{message}")


class RegistryError(EhrlintError):
    """A check could not be registered or resolved."""


class ReportError(EhrlintError):
    """A report could not be rendered or written."""
