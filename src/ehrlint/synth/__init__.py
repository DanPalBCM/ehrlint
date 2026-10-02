"""Deterministic synthetic EHR data. The only data ehrlint ships or tests against."""

from ehrlint.synth.generator import (
    EPOCH,
    HISTORY_CODES,
    INDEX_CODE,
    LAB_CODES,
    OUTCOME_CODES,
    PROXY_CODES,
    SyntheticDataset,
    default_task,
    generate,
)

__all__ = [
    "EPOCH",
    "HISTORY_CODES",
    "INDEX_CODE",
    "LAB_CODES",
    "OUTCOME_CODES",
    "PROXY_CODES",
    "SyntheticDataset",
    "default_task",
    "generate",
]
