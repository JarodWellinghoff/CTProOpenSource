# Public release checklist

## Ownership and legal

- [ ] Confirm code ownership and contributor authorization.
- [ ] Review whether the implementation derives from third-party code.
- [ ] Approve and add a top-level `LICENSE`.
- [ ] Add the license identifier to package and citation metadata.
- [ ] Review names, trademarks, institutional affiliations, and disclaimers.

## Scientific validation

- [ ] Add method references and document every intentional deviation.
- [ ] Define supported phantom types and acquisition/reconstruction assumptions.
- [ ] Establish de-identified golden datasets for NPS, CHO, MTF, and MTFc.
- [ ] Compare v0.1.0 results against the project reference implementation.
- [ ] Define metric-specific numerical tolerances.
- [ ] Confirm units using independent Parseval/frequency-axis checks.
- [ ] Review CHO internal-noise calibration and covariance regularization.
- [ ] Test low- and high-contrast MTFc materials.
- [ ] Test multiple vendors, matrix sizes, pixel spacings, slice spacings, and
      reconstruction kernels.
- [ ] Have a medical physicist or designated method owner approve the release.

## Data governance

- [ ] Confirm no clinical or identifiable DICOM data are tracked in Git history.
- [ ] Scan repository history for internal paths, user names, UIDs, and secrets.
- [ ] Document the approved de-identification process for any example data.
- [ ] Use phantom-only or generated public test data.

## Engineering

- [ ] CI passes on all supported Python versions and operating systems.
- [ ] `ruff check .`, `pytest`, and `python -m build` pass from a clean clone.
- [ ] Install and run each CLI example in a fresh virtual environment.
- [ ] Verify interactive workflows on Windows, macOS, and Linux as applicable.
- [ ] Confirm memory behavior for intended 3-D NPS VOI sizes.
- [ ] Add repository description, topics, issue templates, and private security
      reporting.
- [ ] Tag the release and publish release notes from `CHANGELOG.md`.

## Documentation

- [ ] Replace generic contributor identity with approved names/affiliations.
- [ ] Complete `CITATION.cff`.
- [ ] Add screenshots or figures generated from non-identifiable data.
- [ ] Document known limitations and expected failure modes.
- [ ] Document how users should cite the algorithms and software.
