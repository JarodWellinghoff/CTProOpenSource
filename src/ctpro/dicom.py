"""DICOM loading utilities shared by the CT image-quality metrics."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._utils import ensure_directory, natural_sort_key, validate_pixel_spacing

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CTSeries:
    """A CT image series converted to Hounsfield units.

    Attributes
    ----------
    images_hu:
        Array with shape ``(slices, rows, columns)``.
    pixel_spacing_mm:
        DICOM order: ``(row_spacing_mm, column_spacing_mm)``.
    slice_spacing_mm:
        Median spacing between slice locations, when it can be determined.
    files:
        Source files in the order used to build ``images_hu``.
    series_instance_uid:
        DICOM Series Instance UID, when present.
    """

    images_hu: NDArray[np.float32]
    pixel_spacing_mm: tuple[float, float]
    slice_spacing_mm: float | None
    files: tuple[Path, ...]
    series_instance_uid: str | None = None


@dataclass(frozen=True)
class _HeaderRecord:
    path: Path
    series_uid: str | None
    instance_number: int | None
    z_position_mm: float | None


def _pydicom():
    try:
        import pydicom
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only in minimal environments
        raise ImportError(
            "DICOM support requires pydicom. Install CTPro with "
            "`pip install ctpro-metrics` or install pydicom directly."
        ) from exc
    return pydicom


def discover_dicom_files(
    directory: str | Path, *, recursive: bool = False
) -> list[Path]:
    """Discover readable DICOM files and return them in acquisition order.

    Non-DICOM files are skipped with a debug log message. If the directory
    contains multiple DICOM series, use :func:`load_ct_series` with an explicit
    ``series_uid``.
    """
    folder = ensure_directory(directory)
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    paths = sorted(
        (path for path in candidates if path.is_file()), key=natural_sort_key
    )
    if not paths:
        raise FileNotFoundError(f"No files were found in {folder}.")

    records = _read_header_records(paths)
    if not records:
        raise ValueError(f"No readable DICOM files were found in {folder}.")

    series_uids = {record.series_uid for record in records}
    if len(series_uids) > 1:
        # Keep discovery useful without silently combining series: group first,
        # and let load_ct_series produce the actionable error.
        LOGGER.debug("Directory contains multiple DICOM series: %s", series_uids)

    return [record.path for record in _sort_records(records)]


def _read_header_records(paths: Iterable[Path]) -> list[_HeaderRecord]:
    pydicom = _pydicom()
    records: list[_HeaderRecord] = []
    for path in paths:
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            LOGGER.debug(
                "Skipping non-DICOM or unreadable file: %s", path, exc_info=True
            )
            continue

        series_uid = (
            str(dataset.SeriesInstanceUID) if "SeriesInstanceUID" in dataset else None
        )

        instance_number: int | None = None
        if "InstanceNumber" in dataset:
            try:
                instance_number = int(dataset.InstanceNumber)
            except (TypeError, ValueError):
                pass

        z_position: float | None = None
        if "ImagePositionPatient" in dataset:
            try:
                z_position = float(dataset.ImagePositionPatient[2])
            except (TypeError, ValueError, IndexError):
                pass
        elif "SliceLocation" in dataset:
            try:
                z_position = float(dataset.SliceLocation)
            except (TypeError, ValueError):
                pass

        records.append(
            _HeaderRecord(
                path=path,
                series_uid=series_uid,
                instance_number=instance_number,
                z_position_mm=z_position,
            )
        )
    return records


def _sort_records(records: Sequence[_HeaderRecord]) -> list[_HeaderRecord]:
    def key(record: _HeaderRecord) -> tuple[float, float, list[object]]:
        z = record.z_position_mm if record.z_position_mm is not None else float("inf")
        instance = (
            float(record.instance_number)
            if record.instance_number is not None
            else float("inf")
        )
        return z, instance, natural_sort_key(record.path.name)

    if any(record.z_position_mm is not None for record in records):
        return sorted(records, key=key)
    if any(record.instance_number is not None for record in records):
        return sorted(
            records,
            key=lambda record: (
                (
                    record.instance_number
                    if record.instance_number is not None
                    else float("inf")
                ),
                natural_sort_key(record.path.name),
            ),
        )
    return sorted(records, key=lambda record: natural_sort_key(record.path.name))


def read_hu_image(path: str | Path) -> tuple[NDArray[np.float32], object]:
    """Read one DICOM image and apply its rescale slope/intercept."""
    pydicom = _pydicom()
    dataset = pydicom.dcmread(Path(path))
    image = np.asarray(dataset.pixel_array, dtype=np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    image = image * slope + intercept
    return image.astype(np.float32, copy=False), dataset


def load_ct_series(
    directory: str | Path,
    *,
    start: int = 0,
    stop: int | None = None,
    recursive: bool = False,
    series_uid: str | None = None,
) -> CTSeries:
    """Load a single CT series.

    ``start`` and ``stop`` use normal zero-based Python slicing. The CLI uses
    one-based inclusive slice numbers and converts them before calling this
    function.
    """
    if start < 0:
        raise ValueError("start must be non-negative.")
    if stop is not None and stop <= start:
        raise ValueError("stop must be greater than start.")

    folder = ensure_directory(directory)
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    paths = sorted(
        (path for path in candidates if path.is_file()), key=natural_sort_key
    )
    records = _read_header_records(paths)
    if not records:
        raise ValueError(f"No readable DICOM files were found in {folder}.")

    grouped: dict[str | None, list[_HeaderRecord]] = {}
    for record in records:
        grouped.setdefault(record.series_uid, []).append(record)

    if series_uid is None:
        if len(grouped) != 1:
            choices = ", ".join(repr(uid) for uid in grouped)
            raise ValueError(
                f"{folder} contains {len(grouped)} DICOM series ({choices}). "
                "Specify series_uid to avoid mixing acquisitions."
            )
        selected_uid = next(iter(grouped))
    else:
        if series_uid not in grouped:
            raise ValueError(
                f"Series UID {series_uid!r} was not found. Available values: "
                f"{', '.join(repr(uid) for uid in grouped)}"
            )
        selected_uid = series_uid

    ordered = _sort_records(grouped[selected_uid])
    selected = ordered[start:stop]
    if not selected:
        raise ValueError(
            f"The requested slice range selected no images from {len(ordered)} available slices."
        )

    images: list[NDArray[np.float32]] = []
    row_col_spacing: tuple[float, float] | None = None
    z_positions: list[float] = []
    first_shape: tuple[int, int] | None = None

    for record in selected:
        image, dataset = read_hu_image(record.path)
        if image.ndim != 2:
            raise ValueError(
                f"Expected a 2-D DICOM image in {record.path}, got {image.shape}."
            )
        if first_shape is None:
            first_shape = image.shape
        elif image.shape != first_shape:
            raise ValueError(
                f"Inconsistent image dimensions: expected {first_shape}, "
                f"found {image.shape} in {record.path}."
            )

        if "PixelSpacing" in dataset:
            spacing = validate_pixel_spacing(dataset.PixelSpacing)
        elif (
            "ReconstructionDiameter" in dataset
            and "Rows" in dataset
            and "Columns" in dataset
        ):
            spacing = (
                float(dataset.ReconstructionDiameter) / int(dataset.Rows),
                float(dataset.ReconstructionDiameter) / int(dataset.Columns),
            )
        else:
            raise ValueError(
                f"Pixel spacing is unavailable in {record.path}. Expected PixelSpacing "
                "or ReconstructionDiameter with Rows/Columns."
            )

        if row_col_spacing is None:
            row_col_spacing = spacing
        elif not np.allclose(row_col_spacing, spacing, rtol=0, atol=1e-6):
            raise ValueError(
                f"Inconsistent pixel spacing: expected {row_col_spacing}, found {spacing} "
                f"in {record.path}."
            )

        if "ImagePositionPatient" in dataset:
            try:
                z_positions.append(float(dataset.ImagePositionPatient[2]))
            except (TypeError, ValueError, IndexError):
                pass
        images.append(image)

    slice_spacing: float | None = None
    if len(z_positions) >= 2:
        differences = np.abs(np.diff(np.asarray(z_positions, dtype=float)))
        nonzero = differences[differences > 0]
        if nonzero.size:
            slice_spacing = float(np.median(nonzero))
    if slice_spacing is None:
        _, first_dataset = read_hu_image(selected[0].path)
        for attribute in ("SpacingBetweenSlices", "SliceThickness"):
            if hasattr(first_dataset, attribute):
                candidate = abs(float(getattr(first_dataset, attribute)))
                if candidate > 0:
                    slice_spacing = candidate
                    break

    assert row_col_spacing is not None
    stack = np.stack(images, axis=0).astype(np.float32, copy=False)
    return CTSeries(
        images_hu=stack,
        pixel_spacing_mm=row_col_spacing,
        slice_spacing_mm=slice_spacing,
        files=tuple(record.path for record in selected),
        series_instance_uid=selected_uid,
    )


def load_repeated_ct_series(
    directories: Sequence[str | Path],
    *,
    start: int = 0,
    stop: int | None = None,
) -> CTSeries:
    """Load and concatenate equal slice ranges from repeated scans."""
    if not directories:
        raise ValueError("At least one scan directory is required.")
    series = [load_ct_series(path, start=start, stop=stop) for path in directories]
    reference = series[0]

    for item in series[1:]:
        if item.images_hu.shape[1:] != reference.images_hu.shape[1:]:
            raise ValueError("Repeated scans have inconsistent image dimensions.")
        if not np.allclose(
            item.pixel_spacing_mm, reference.pixel_spacing_mm, atol=1e-6
        ):
            raise ValueError("Repeated scans have inconsistent pixel spacing.")
        if item.images_hu.shape[0] != reference.images_hu.shape[0]:
            raise ValueError(
                "Repeated scans must contribute the same number of slices."
            )

    combined = np.concatenate([item.images_hu for item in series], axis=0)
    files = tuple(path for item in series for path in item.files)
    return CTSeries(
        images_hu=combined,
        pixel_spacing_mm=reference.pixel_spacing_mm,
        slice_spacing_mm=reference.slice_spacing_mm,
        files=files,
        series_instance_uid=None,
    )
