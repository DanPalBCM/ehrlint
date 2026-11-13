# Changelog

All notable changes to ehrlint are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-11-13

First release. Published to PyPI through Trusted Publishing
(`.github/workflows/release.yml`), archived on Zenodo with a DOI, and
documented at <https://danpalbcm.github.io/ehrlint/>.

**Ingestion.** MEDS dataset directories (verified against `meds` 0.4.1) and
OMOP CDM extracts as parquet or CSV, both converted to one canonical event
table. Optional feature matrices and labels tables in either ehrlint's or
MEDS's shape.

**The check battery.** LK001–LK010, each with a rationale, the exact query it
runs, example rows from your data, and a matching injector that plants the leak
it looks for.

**Reports.** A single self-contained offline HTML file, plus Markdown, JSON,
and findings JSONL. Skips and input warnings print above the findings.

**CLI.** `audit`, `synth`, `validate-task`, `list-checks`, with exit codes 0,
1, and 2 kept distinct so a broken invocation cannot look like a clean audit.

**Synthetic data.** `ehrlint synth` generates cohorts that are clean by
construction, and `ehrlint.inject` plants any of the ten leaks on disjoint
subjects with the affected ids returned as ground truth.

**Measured.** 100,000 subjects audit in 9.3 s and 10.3 M events in 67.5 s,
against SPEC §4.5's five-minute budget.

See [decisions and deviations](https://danpalbcm.github.io/ehrlint/decisions/) for
what was verified, what departs from the specification, and the bugs that
changed a design.

[1.0.0]: https://github.com/DanPalBCM/ehrlint/releases/tag/v1.0.0
