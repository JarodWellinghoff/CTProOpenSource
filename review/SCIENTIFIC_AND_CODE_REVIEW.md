# Scientific and code review for v0.1.0

This review records publication blockers found while converting the four
stand-alone scripts into a package. It distinguishes straightforward software
defects from choices that can alter quantitative results.

## High-priority defects corrected

### Import-time execution and internal paths

Every original module ran an example immediately when imported and included a
hard-coded Windows or network path. This made reuse, testing, packaging, and
safe publication impossible. Examples are now represented by a CLI and
documentation; imports have no data access or GUI side effects.

### Inconsistent image coordinates

The NPS viewer drew rectangles using one coordinate as the horizontal axis but
the numerical extraction used that coordinate as the NumPy row index. This can
select a transposed location when offsets or asymmetric patterns are used.
Public centers are now consistently `(x, y) = (column, row)`.

### DICOM ordering and HU conversion

The scripts primarily sorted filenames and sometimes reused rescale metadata
from the first image. One NPS path left the HU image uninitialized when rescale
tags were absent. The package now:

- orders slices by patient-space position, then `InstanceNumber`;
- rejects accidental mixing of multiple Series Instance UIDs;
- applies each image's slope and intercept, defaulting to 1 and 0;
- checks image dimensions and pixel spacing across the series.

### Indexing/runtime defects

- The original 3-D NPS code wrote the first slice of each VOI to index `-1`.
  Magnitude spectra are translation-invariant, but the indexing was misleading
  and unsafe to maintain.
- NPS statistics could fail when passed the nested list returned by the 2-D
  routine.
- Thin-wire MTF could index a pre-sliced filename list with the original
  one-based range.
- The MTFc selector retained multiple callbacks but analyzed the first ROI.
- MTFc optimization depended on module-level `xData` and `yData`.
- The MTFc histogram-style binning wrapped samples at the last edge into the
  first bin.

## Results-affecting changes that require method-owner validation

### NPS radial average

The original code sampled 360 nearest-neighbor angular profiles. The package
uses annular bins defined on the physical frequency grid. This supports
anisotropic frequency axes and avoids repeated nearest-neighbor samples, but it
can produce small differences in the 1-D curve and derived frequencies.

The 2-D periodogram normalization remains:

```text
|FFT(ROI)|² × row_spacing_cm × column_spacing_cm / ROI_pixel_count
```

The 3-D normalization analogously includes slice spacing and cubic voxel count.
Validate both with a Parseval-style variance check and known synthetic noise.

### NPS ROI placement and count

The original settings described circle radius and matrix side length as
millimetres but used them directly as pixel offsets. The package CLI converts
these physical distances through DICOM pixel spacing before placing ROIs. This
can substantially change ROI locations when pixel spacing is not 1 mm.

The original circle ROI lookup table also had unclear unit/index semantics and
unsafe array bounds. The package instead computes a non-overlapping geometric
count when none is supplied. Use CLI `--count` to reproduce a protocol-defined
ROI count.

### CHO covariance and AUC

The original implementation used a direct covariance inverse, which can fail
for small or correlated channel ensembles. The package uses a pseudoinverse and
adds a small configurable diagonal regularizer by default. This stabilizes the
calculation but changes d′. Validate and freeze the regularization policy for
published studies.

Empirical AUC is now calculated by the rank-statistic identity rather than
manual threshold integration. The definitions are equivalent for empirical ROC
data when ties are handled consistently; verify tie behavior against the
reference implementation.

### CHO internal noise

The original example used `internalNoise = 2.25` without a citation or
calibration description. The package defaults to zero and requires an explicit
value. The method owner must document how any non-zero value was estimated and
for which observer/task it is valid.

### Thin-wire MTF averaging

The package computes radial profiles in physical annular bins and derives final
threshold crossings from the average curve. The original code averaged
per-slice crossing frequencies. Those operations are not mathematically
identical. Compare both approaches on the reference wire dataset and document
the chosen estimator.

### MTFc radial bins and center refinement

The corrected radial binning removes edge wraparound and uses the final ROI
selection. These are defect fixes but can materially change MTFc, particularly
at low frequency. The package retains the main strategy: upsampled Hough
detection, ±2-pixel center search, ESF smoothness/slope selection, Hann-windowed
LSF, and a median curve.

The default output grid remains limited to four-times pixel oversampling even
when the high-contrast branch uses ten-times ESF oversampling. Confirm that this
is intentional.

## Method details not supported by the source files

The supplied files do not establish:

- primary literature or standards for each implementation;
- accepted phantom models and insert dimensions beyond examples;
- minimum ensemble sizes with statistical justification;
- scanner/reconstruction conditions under which the methods are validated;
- expected numerical results or tolerances;
- provenance of the CHO internal-noise value;
- whether output is intended for research, routine QA, accreditation, or
  regulatory use.

These gaps should not be filled from general knowledge without method-owner
review. Add references and a protocol document before a scientific v1.0
release.

## Recommended validation matrix

At minimum, compare old and new outputs for:

- 2-D NPS: multiple kernels, doses, ROI sizes, and offsets;
- 3-D NPS: isotropic and anisotropic voxels, at least two VOI sizes;
- CHO: Gabor and Laguerre-Gauss channels, one and three repeated scans,
  regularization sweep, fixed seeds;
- thin-wire MTF: centered and subpixel-offset wires, several reconstruction
  fields of view;
- MTFc: air, polyethylene, acrylic, and bone-like inserts, including CNR values
  on both sides of the low/high contrast threshold.

Archive raw curves as well as scalar metrics so differences can be localized.
