# Migration from the stand-alone scripts

The original files mixed configuration, DICOM I/O, GUI interaction, numerical
work, plotting, and printing. The package separates those responsibilities.

## Function mapping

| Original entry point | Package replacement |
|---|---|
| `NPS_water2D` | `ctpro.compute_nps_2d` |
| `NPS_water3D` | `ctpro.compute_nps_3d` |
| `NPS_statistics` | `ctpro.nps.nps_statistics` |
| `Display_Image_ROI` | `ctpro.nps.interactively_adjust_rois` plus ROI placement helpers |
| `CHO_function_with_resampling` | `ctpro.compute_cho` |
| `CHO_FOMs_Single` | `ctpro.analyze_cho` |
| `ChannelMatrix_Generation` | `GaborChannels.matrix` or `LaguerreGaussChannels.matrix` |
| `Func_for_mtf` | `ctpro.mtf.compute_slice_mtf` |
| `MTF` | `ctpro.compute_wire_mtf` |
| top-level MTFc script | `ctpro.compute_mtfc` |

## Important convention changes

- Image stacks are always `(slice, row, column)`.
- Public centers are always `(x, y) = (column, row)`.
- Python APIs use zero-based, stop-exclusive slice ranges through
  `load_ct_series(start=..., stop=...)`.
- The CLI accepts one-based, inclusive `--start` and `--end`.
- MTFc ROI bounds use `(row_start, row_stop, column_start, column_stop)`.
- Compute functions return dataclasses rather than printing or displaying
  figures.
- Interactive windows are opt-in.

## Dependency changes

The original requirements file pinned a complete environment, including
transitive and unused packages. The package declares only direct runtime
dependencies. Projects that require exact reproducibility should generate a
lock file for their target operating system and Python version.

## Numerical differences requiring baseline review

See `review/SCIENTIFIC_AND_CODE_REVIEW.md`. The most consequential changes are
consistent row/column extraction, DICOM metadata ordering, CHO covariance
regularization, annular radial averaging, and corrected MTFc radial binning.
