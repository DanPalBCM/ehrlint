"""The check protocol and registry.

A check is a small class with an id, a name, a rationale, a default severity,
and a `run(ctx)` that returns findings. Third parties register their own with
:func:`register_check`.

Two rules every check obeys:

1. **Evidence or nothing.** A finding with a non-zero subject count must carry
   the rows that produced it. :func:`~ehrlint.schemas.findings.make_finding`
   is the only constructor, and it derives counts from the offending frame
   rather than taking them on trust.
2. **Skip loudly.** A check that cannot run — no encounter ids, no horizon, no
   splits — returns a :class:`~ehrlint.schemas.findings.SkippedCheck` with a
   reason. It never returns "no findings", because a skipped check is not a
   pass and a reader must be able to tell the difference.

Checks are also expected to be deterministic: same input, same findings, same
examples, same order. A property test asserts it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ehrlint.exceptions import RegistryError
from ehrlint.schemas.findings import Finding, Severity, SkippedCheck

if TYPE_CHECKING:
    from ehrlint.context import AuditContext

#: Result of running one check: findings, or a reason it could not run.
CheckResult = list[Finding] | SkippedCheck


@runtime_checkable
class Check(Protocol):
    """Anything that can audit an :class:`~ehrlint.context.AuditContext`."""

    id: str
    name: str
    description: str
    severity_default: Severity

    def run(self, ctx: AuditContext) -> CheckResult:
        """Audit the context and return findings, or a skip with a reason."""
        ...


class BaseCheck:
    """Convenience base for the built-in checks.

    Supplies the metadata attributes and a few helpers every check wants:
    building a skip, counting the denominator, and resolving code sets.
    """

    id: str = ""
    name: str = ""
    description: str = ""
    rationale: str = ""
    severity_default: Severity = Severity.WARNING

    def run(self, ctx: AuditContext) -> CheckResult:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- helpers --------------------------------------------------------

    def skip(self, reason: str) -> SkippedCheck:
        """Record that this check could not run, and why."""
        return SkippedCheck(check_id=self.id, name=self.name, reason=reason)

    def clean(self, ctx: AuditContext, message: str, query: str) -> list[Finding]:
        """A zero-count finding, meaning the check ran and found nothing.

        Distinct from a skip: this is a genuine pass, and it appears in the
        report as such so a reader can see what was actually verified.
        """
        import polars as pl

        from ehrlint.schemas.findings import make_finding

        return [
            make_finding(
                self.id,
                name=self.name,
                severity=self.severity_default,
                message=message,
                offending=pl.DataFrame(),
                n_total_subjects=ctx.n_subjects,
                query=query,
            )
        ]

    def metadata(self) -> dict[str, str]:
        """Check metadata for the report and the docs."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "rationale": self.rationale,
            "severity_default": self.severity_default.value,
        }


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Check] = {}


def register(check: Check, *, replace: bool = False) -> Check:
    """Add a check instance to the registry.

    Args:
        check: The check.
        replace: Allow overwriting an existing id. Off by default, because two
            checks silently sharing an id would make findings ambiguous.

    Raises:
        RegistryError: On a duplicate id, or on a check missing its metadata.
    """
    if not getattr(check, "id", ""):
        raise RegistryError(f"{type(check).__name__} has no id")
    if not getattr(check, "name", ""):
        raise RegistryError(f"check {check.id} has no name")
    if check.id in _REGISTRY and not replace:
        existing = type(_REGISTRY[check.id]).__name__
        raise RegistryError(
            f"check id {check.id!r} is already registered to {existing}. Pass "
            "replace=True to override deliberately."
        )
    _REGISTRY[check.id] = check
    return check


def register_check(
    *,
    id: str,
    name: str,
    severity: Severity | str = Severity.WARNING,
    description: str = "",
    rationale: str = "",
    replace: bool = False,
) -> Callable[[Callable[[AuditContext], CheckResult]], Callable[[AuditContext], CheckResult]]:
    """Decorator registering a plain function as a check.

    The SPEC §5.2 extension point::

        @el.register_check(id="LK099", name="my_check", severity="warning")
        def my_check(ctx: el.AuditContext) -> list[el.Finding]:
            ...

    Returns the undecorated function, so it stays directly callable and
    testable.
    """
    severity_value = Severity(severity) if isinstance(severity, str) else severity

    def decorator(
        fn: Callable[[AuditContext], CheckResult],
    ) -> Callable[[AuditContext], CheckResult]:
        class _FunctionCheck(BaseCheck):
            pass

        _FunctionCheck.id = id
        _FunctionCheck.name = name
        _FunctionCheck.description = description or (fn.__doc__ or "").strip().split("\n")[0]
        _FunctionCheck.rationale = rationale
        _FunctionCheck.severity_default = severity_value
        _FunctionCheck.run = staticmethod(fn)  # type: ignore[assignment]
        _FunctionCheck.__name__ = f"Check_{id}"
        _FunctionCheck.__qualname__ = f"Check_{id}"

        register(_FunctionCheck(), replace=replace)
        return fn

    return decorator


def get_check(check_id: str) -> Check:
    """Look up a check by id.

    Raises:
        RegistryError: If unknown, listing what is registered — a typo in a
            `--checks` list should not silently run nothing.
    """
    key = check_id.strip().upper()
    if key not in _REGISTRY:
        raise RegistryError(
            f"unknown check {check_id!r}; registered: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key]


def all_checks() -> list[Check]:
    """Every registered check, in id order so runs are reproducible."""
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def check_ids() -> list[str]:
    """Registered check ids, sorted."""
    return sorted(_REGISTRY)


def resolve_checks(selection: list[str] | None = None) -> list[Check]:
    """Resolve a selection of check ids to check instances.

    Args:
        selection: Ids to run. None or empty means all of them.

    Returns:
        Checks in id order.
    """
    if not selection:
        return all_checks()
    wanted = [s.strip().upper() for s in selection if s and s.strip()]
    return [get_check(cid) for cid in sorted(dict.fromkeys(wanted))]


def clear_registry() -> None:
    """Empty the registry. For tests that register throwaway checks."""
    _REGISTRY.clear()


def registry_snapshot() -> dict[str, Check]:
    """A copy of the registry, so a test can restore it."""
    return dict(_REGISTRY)


def restore_registry(snapshot: dict[str, Check]) -> None:
    """Replace the registry contents with a snapshot."""
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)
