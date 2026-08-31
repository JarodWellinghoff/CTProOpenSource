# Changelog

All notable changes to this project will be documented here.

The format follows Keep a Changelog principles. The project does not yet claim
semantic-versioning stability because its scientific validation baseline has
not been published.

## [Unreleased]

### Required before public release

- Approve and add an open-source license.
- Add method references and a public or access-controlled golden validation set.
- Confirm numerical equivalence or intentionally accepted differences against
  the project reference implementation.

## [0.1.0] - 2026-08-31

### Added

- Reusable `ctpro` package with typed result dataclasses.
- DICOM loading with per-slice HU rescaling and metadata-based ordering.
- Array-first APIs for 2-D/3-D NPS, CHO, thin-wire MTF, and circular-insert MTFc.
- `ctpro` command-line interface with JSON and NPZ output.
- Synthetic tests, lint configuration, packaging metadata, and CI workflow.
- Privacy, contribution, release, migration, and scientific-review guidance.

### Changed

- Standardized image stacks to `(slice, row, column)` and ROI centers to
  `(x, y) = (column, row)`.
- Removed hard-coded internal paths and import-time execution.
- Replaced broad, fully pinned transitive requirements with direct library
  dependencies and development extras.
- Added explicit CHO random seeds and covariance regularization.
- Made plotting and interactive selection opt-in.

### Fixed

- Uninitialized HU image data when DICOM rescale tags are absent.
- Inconsistent row/column use during ROI display versus extraction.
- A 3-D NPS off-by-one volume index.
- NPS statistics handling of nested Python lists.
- Fragile filename-only DICOM ordering.
- MTF processing of pre-sliced filename lists.
- MTFc reuse of the first rather than final interactive ROI.
- MTFc optimizer dependence on module-level global data.
- MTFc radial-bin wraparound at the final bin.
