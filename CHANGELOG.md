# Changelog

This file records the public releases of this package.

## 1.0.2 — 2026-09-24

Three findings from an external audit of the published package. `requirements.txt` now declares
PyYAML and mpmath, which two and three shipped scripts respectively import and which it had
omitted. The derivation README says that `d98` and `d129` are stored results whose generating
script does not survive, instead of leaving a reader to hunt for it. The 1.0.0 entry below no
longer claims the pre-registrations are purely pre-run: one of them carries a post-run
correction as its own section, and the entry now says so. No code, data or result file changes.

## 1.0.1 — 2026-09-24

The concept DOI issued at first publication, 10.5281/zenodo.22931805, is written into
`CITATION.cff`, `CITATION.md` and `README.md`. Not one code, data, result, figure or engine
file differs from 1.0.0 — verified by comparing the two trees file by file. The six files that
do differ are those three, this changelog, `VERSION`, and `MANIFEST.sha256`, which records
their hashes.

## 1.0.0 — 2026-09-23

First public release. It contains the analysis and experiment code, the stored result files the
figures and tables are built from, the third-party-controller harness, the pre-registrations,
and the crowd simulation engine used as the study's reference system. Each pre-registration is
kept as it was written before the runs it predicts; where a run later corrected the record, the
correction is appended to it as its own dated section rather than edited into the original.
