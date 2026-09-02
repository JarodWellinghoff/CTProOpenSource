# CTPro Metrics

Research-oriented Python tools for quantitative CT image-quality analysis:

- **NPS** — two-dimensional and three-dimensional Noise Power Spectrum
- **CHO** — Channelized Hotelling Observer for SKE/BKE detection tasks
- **Thin-wire MTF** — Point Spread Function and radially averaged MTF
- **MTFc** — contrast-dependent MTF from circular phantom inserts

This repository is a cleaned, reusable successor to four stand-alone analysis
scripts. The numerical routines are import-safe, use explicit inputs, return
structured results, and do not contain project-specific paths.

> **Validation status**
>
> Version 0.1.0 is a release candidate for research and CT quality-assurance
> development. It has synthetic unit tests, but no public reference DICOM
> dataset or cross-vendor acceptance limits are included. Validate the package
> against your institution's reference implementation and phantom data before
> relying on reported values. It is not a medical device and is not intended
> for diagnosis or direct patient-care decisions.

## Installation

CTPro requires Python 3.10 or newer.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
ctpro --help
```

For development:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m build
```

## Command-line examples

Slice numbers passed to the CLI are **one-based and inclusive**.

### 2-D NPS

```bash
ctpro nps /path/to/uniform-series \
  --dimension 2d \
  --layout circle \
  --radius-px 120 \
  --roi-size 40 \
  --start 1 --end 60 \
  --interactive \
  --json nps-summary.json \
  --npz nps-arrays.npz \
  --plot
```

For 3-D NPS, each independent cubic VOI consumes `--roi-size` consecutive
slices:

```bash
ctpro nps /path/to/uniform-series \
  --dimension 3d \
  --roi-size 32 \
  --volumes 2 \
  --max-memory-gib 2
```

### CHO

Pass one directory per repeated scan. Each scan must contribute the same slice
range and image geometry.

```bash
ctpro cho /path/to/scan-1 /path/to/scan-2 /path/to/scan-3 \
  --start 102 --end 150 \
  --signal-center 161,156 \
  --background-center 161,256 \
  --roi-size 30 \
  --channels gabor \
  --resamples 500 \
  --seed 2026 \
  --json cho-summary.json
```

`--internal-noise` defaults to zero. A non-zero value is an observer-model
calibration parameter, not a scanner-independent constant.

### Thin-wire MTF

```bash
ctpro mtf /path/to/wire-series \
  --start 1 --end 5 \
  --roi-size 20 \
  --interactive \
  --npz mtf-arrays.npz \
  --plot
```

The wire is detected automatically when `--center X,Y` is omitted.

### Circular-insert MTFc

Choose the HU threshold for the material being analyzed.

```bash
ctpro mtfc /path/to/insert-series \
  --start 16 --end 63 \
  --roi 255,255,100 \
  --hu-range -110 -80 \
  --json mtfc-summary.json \
  --plot
```

Omit `--roi` and add `--interactive` to draw the square insert ROI.

## Python API

```python
import numpy as np

from ctpro import compute_nps_2d, load_ct_series, place_rois_on_circle

series = load_ct_series("/path/to/uniform-series", start=0, stop=60)

row_mm, col_mm = series.pixel_spacing_mm
if not np.isclose(row_mm, col_mm):
    raise ValueError("This example assumes square pixels.")

centers = place_rois_on_circle(
    series.images_hu.shape[1:],
    radius_px=120 / col_mm,
    roi_size_px=40,
)

result = compute_nps_2d(
    series.images_hu,
    centers,
    roi_size_px=40,
    pixel_spacing_mm=series.pixel_spacing_mm,
)

print(result.metrics.peak_frequency_per_cm)
print(result.noise_sd_hu)
```

All compute functions accept NumPy arrays, so DICOM loading can be replaced by
an existing image pipeline.

## Data and coordinate conventions

| Item | Convention |
|---|---|
| Image stack | `(slices, rows, columns)` |
| ROI center | `(x, y)` = `(column, row)`, zero-based pixels |
| DICOM pixel spacing | `(row_mm, column_mm)` |
| CLI slice range | one-based, inclusive |
| Python slice range | zero-based, stop-exclusive |
| NPS frequency | `1/cm` |
| Thin-wire MTF frequency | `1/cm` |
| MTFc frequency | `lp/cm` |

DICOM loading applies each image's `RescaleSlope` and `RescaleIntercept`.
Files are sorted by patient-space slice position when available, then by
`InstanceNumber`, with natural filename ordering only as a final fallback.
A directory containing multiple `SeriesInstanceUID` values is rejected unless
a series is selected explicitly through the Python API.

## Reproducibility

- CHO uses NumPy's `Generator`; pass `seed=` or CLI `--seed`.
- Numerical functions do not open windows or print results.
- Plotting and interactive ROI selection are explicit opt-in functions.
- CLI summaries can be saved as JSON and arrays as compressed NPZ.
- Direct dependencies and minimum supported versions are declared in `pyproject.toml`; applications that
  require full environment locking should generate and retain a lock file.

## Privacy and data governance

Do not commit clinical DICOM images. DICOM headers can contain protected health
information even when pixel data look like phantom scans. Use institutionally
approved de-identification and verification procedures before sharing any data.
The repository `.gitignore` excludes common DICOM extensions as a precaution,
but filename filtering is not a de-identification control.

## Repository layout

```text
src/ctpro/        reusable package
tests/            synthetic unit and smoke tests
examples/         usage notes
.github/workflows continuous integration
review/           release and scientific-review notes
```

## Method provenance

The source scripts did not include literature citations or a formal algorithm
specification. Before a public scientific release, add the exact references,
phantom protocols, reconstruction assumptions, and expected tolerances used by
the project. `review/SCIENTIFIC_AND_CODE_REVIEW.md` identifies the items that
need method-owner confirmation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes that may alter numerical output
must include a before/after comparison on a non-identifiable reference dataset
and an entry in [CHANGELOG.md](CHANGELOG.md).

## License

No open-source license is activated in this preparation bundle because code
ownership and institutional approval were not supplied. Select and approve a
license **before making the repository public**; without one, outside users do
not receive permission to copy, modify, or redistribute the code. See
[LICENSING.md](LICENSING.md).
