# Contributing

Thank you for improving CTPro Metrics.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ruff check .
pytest
```

## Pull requests

Keep pull requests focused. Include:

1. A clear description of the behavior being changed.
2. Tests that fail before the change and pass after it.
3. Documentation for public API or CLI changes.
4. A changelog entry for user-visible changes.
5. For numerical changes, before/after results on a de-identified reference
   dataset, including acquisition/reconstruction parameters and the reason the
   new output is preferable.

Do not commit DICOM files, screenshots containing identifiers, local network
paths, credentials, or institution-only metadata.

## Numerical compatibility

Changes to any of the following are considered results-affecting and require
method-owner review:

- ROI indexing, placement, or coordinate conventions
- HU conversion or DICOM ordering
- detrending, windowing, interpolation, or resampling
- FFT size, normalization, radial averaging, or frequency units
- covariance estimation, regularization, observer channels, or internal noise
- threshold crossing or metric interpolation

Use explicit tolerances in tests. Avoid asserting exact floating-point equality.

## Style

- Public functions need docstrings and type annotations.
- Numerical functions should accept arrays and return values without printing,
  opening GUI windows, or reading global configuration.
- Keep plotting, interaction, and file I/O in separate functions.
- Raise actionable exceptions instead of calling `sys.exit`.
- Use `(x, y)` for public ROI coordinates and document units at every boundary.

## Reporting problems

Use a minimal synthetic example where possible. Never attach identifiable
clinical data to a public issue.
