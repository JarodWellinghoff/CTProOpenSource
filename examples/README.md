# Examples

The command-line examples in the top-level README are the recommended starting
point. They intentionally use placeholder paths and do not bundle DICOM data.

For programmatic use:

```python
from ctpro import load_ct_series
from ctpro.mtf import compute_wire_mtf, detect_wire_center

series = load_ct_series("/path/to/wire-series", start=0, stop=5)
reference = series.images_hu.mean(axis=0)
center = detect_wire_center(reference, roi_size_px=20)

result = compute_wire_mtf(
    series.images_hu,
    series.pixel_spacing_mm,
    center,
    roi_size_px=20,
)

print(result.metrics)
```

For automated pipelines, avoid interactive selection and store protocol-defined
ROI coordinates in a version-controlled configuration file. Save raw curves,
not only scalar metrics, so future algorithm changes can be audited.
