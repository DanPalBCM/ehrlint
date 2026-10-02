"""The check registry and the built-in leakage battery.

Importing this module registers LK001 through LK010.
"""

from ehrlint.checks.base import (
    BaseCheck,
    Check,
    CheckResult,
    all_checks,
    check_ids,
    clear_registry,
    get_check,
    register,
    register_check,
    registry_snapshot,
    resolve_checks,
    restore_registry,
)
from ehrlint.checks.leakage import (
    BUILTIN_CHECKS,
    DuplicateSubjects,
    EncounterGroup,
    HorizonViolation,
    ImmortalTime,
    LabelDefinition,
    OutcomeProxy,
    PostIndexFeatures,
    SameEncounter,
    SplitOverlap,
    TemporalSplit,
    register_builtins,
)

register_builtins()

__all__ = [
    "BUILTIN_CHECKS",
    "BaseCheck",
    "Check",
    "CheckResult",
    "DuplicateSubjects",
    "EncounterGroup",
    "HorizonViolation",
    "ImmortalTime",
    "LabelDefinition",
    "OutcomeProxy",
    "PostIndexFeatures",
    "SameEncounter",
    "SplitOverlap",
    "TemporalSplit",
    "all_checks",
    "check_ids",
    "clear_registry",
    "get_check",
    "register",
    "register_builtins",
    "register_check",
    "registry_snapshot",
    "resolve_checks",
    "restore_registry",
]
