"""Internal numerical and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating]


def normal_round(value: float) -> int:
    """Round halves away from zero.

    Python's built-in ``round`` uses bankers' rounding. The original CTPro
    scripts used a half-up convention, so this helper keeps that behavior.
    """
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def natural_sort_key(value: str | Path) -> list[object]:
    """Return a case-insensitive natural-sort key."""
    text = str(value)
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", text)]


@dataclass(frozen=True)
class SquareROI:
    """A square image ROI using Cartesian center coordinates.

    Parameters
    ----------
    center_xy:
        ``(x, y)`` = ``(column, row)`` in zero-based pixel coordinates.
    size_px:
        Side length in pixels. Even sizes are recommended because they avoid
        ambiguity when converting the center to array slices.
    """

    center_xy: tuple[float, float]
    size_px: int

    def __post_init__(self) -> None:
        if self.size_px <= 0:
            raise ValueError("ROI size must be positive.")

    def slices(self, image_shape: Sequence[int]) -> tuple[slice, slice]:
        """Return ``(row_slice, column_slice)`` and validate image bounds."""
        if len(image_shape) < 2:
            raise ValueError("image_shape must contain at least rows and columns.")

        rows, cols = int(image_shape[-2]), int(image_shape[-1])
        x, y = self.center_xy
        row_start = normal_round(y - self.size_px / 2)
        col_start = normal_round(x - self.size_px / 2)
        row_stop = row_start + self.size_px
        col_stop = col_start + self.size_px

        if row_start < 0 or col_start < 0 or row_stop > rows or col_stop > cols:
            raise ValueError(
                "ROI extends outside the image: "
                f"center_xy={self.center_xy}, size_px={self.size_px}, "
                f"image_shape={(rows, cols)}."
            )
        return slice(row_start, row_stop), slice(col_start, col_stop)

    def extract(self, image: ArrayLike) -> FloatArray:
        """Extract the ROI from a 2-D image."""
        array = np.asarray(image)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2-D image, received shape {array.shape}.")
        row_slice, col_slice = self.slices(array.shape)
        return np.asarray(array[row_slice, col_slice], dtype=float)


def as_image_stack(images: ArrayLike) -> FloatArray:
    """Validate and return an image stack with shape ``(slices, rows, columns)``."""
    stack = np.asarray(images, dtype=float)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3:
        raise ValueError(
            "Expected a 2-D image or a 3-D stack with shape "
            f"(slices, rows, columns); received {stack.shape}."
        )
    if not np.all(np.isfinite(stack)):
        raise ValueError("Image data contain NaN or infinite values.")
    return stack


def validate_pixel_spacing(pixel_spacing_mm: Sequence[float]) -> tuple[float, float]:
    """Validate DICOM-style ``(row, column)`` pixel spacing in millimetres."""
    if len(pixel_spacing_mm) != 2:
        raise ValueError("pixel_spacing_mm must contain (row_mm, column_mm).")
    row_mm, col_mm = (float(pixel_spacing_mm[0]), float(pixel_spacing_mm[1]))
    if row_mm <= 0 or col_mm <= 0:
        raise ValueError("Pixel spacing values must be positive.")
    return row_mm, col_mm


def first_descending_crossing(
    x: ArrayLike,
    y: ArrayLike,
    level: float,
    *,
    start_index: int = 0,
) -> float:
    """Interpolate the first downward crossing of ``level``.

    ``NaN`` is returned when no crossing occurs. Flat segments are handled by
    returning the right-hand sample location.
    """
    x_values = np.asarray(x, dtype=float).ravel()
    y_values = np.asarray(y, dtype=float).ravel()
    if x_values.size != y_values.size or x_values.size == 0:
        raise ValueError("x and y must be non-empty arrays of equal length.")
    if start_index < 0 or start_index >= x_values.size:
        raise ValueError("start_index is outside the curve.")

    for index in range(max(start_index + 1, 1), y_values.size):
        y_left = y_values[index - 1]
        y_right = y_values[index]
        if y_left > level >= y_right:
            x_left = x_values[index - 1]
            x_right = x_values[index]
            if y_right == y_left:
                return float(x_right)
            fraction = (level - y_left) / (y_right - y_left)
            return float(x_left + fraction * (x_right - x_left))

    exact = np.flatnonzero(y_values[start_index:] == level)
    if exact.size:
        return float(x_values[start_index + int(exact[0])])
    return float("nan")


def radial_average(
    values: ArrayLike,
    y_coordinates: ArrayLike,
    x_coordinates: ArrayLike,
    *,
    bin_width: float | None = None,
    max_radius: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Average a 2-D array in annular bins around coordinate zero.

    The coordinate arrays describe the physical coordinate assigned to each
    row and column. Samples are assigned to the nearest radial bin, so the
    returned radius starts at zero.
    """
    image = np.asarray(values, dtype=float)
    y_axis = np.asarray(y_coordinates, dtype=float).ravel()
    x_axis = np.asarray(x_coordinates, dtype=float).ravel()
    if image.shape != (y_axis.size, x_axis.size):
        raise ValueError(
            f"values shape {image.shape} does not match coordinate lengths "
            f"{(y_axis.size, x_axis.size)}."
        )

    positive_steps = []
    if y_axis.size > 1:
        positive_steps.extend(np.abs(np.diff(y_axis))[np.abs(np.diff(y_axis)) > 0])
    if x_axis.size > 1:
        positive_steps.extend(np.abs(np.diff(x_axis))[np.abs(np.diff(x_axis)) > 0])
    if bin_width is None:
        if not positive_steps:
            raise ValueError("At least one coordinate axis must contain multiple samples.")
        bin_width = float(np.min(positive_steps))
    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    xx, yy = np.meshgrid(x_axis, y_axis)
    radii = np.hypot(xx, yy)
    if max_radius is None:
        max_radius = min(float(np.max(np.abs(x_axis))), float(np.max(np.abs(y_axis))))
    if max_radius < 0:
        raise ValueError("max_radius must be non-negative.")

    bin_index = np.rint(radii / bin_width).astype(int)
    max_bin = int(math.floor(max_radius / bin_width))
    valid = (bin_index <= max_bin) & np.isfinite(image)

    sums = np.bincount(bin_index[valid].ravel(), weights=image[valid].ravel(), minlength=max_bin + 1)
    counts = np.bincount(bin_index[valid].ravel(), minlength=max_bin + 1)
    means = np.divide(
        sums,
        counts,
        out=np.full(max_bin + 1, np.nan, dtype=float),
        where=counts > 0,
    )

    radii_out = np.arange(max_bin + 1, dtype=float) * bin_width
    finite = np.isfinite(means)
    if np.count_nonzero(finite) >= 2 and not np.all(finite):
        means = np.interp(radii_out, radii_out[finite], means[finite])
    return radii_out, means


def centered_coordinate_axis(length: int, spacing: float = 1.0) -> FloatArray:
    """Return a coordinate axis centered on the FFT-shifted zero index."""
    if length <= 0 or spacing <= 0:
        raise ValueError("length and spacing must be positive.")
    return (np.arange(length, dtype=float) - length // 2) * spacing


def ensure_directory(path: str | Path) -> Path:
    """Return an existing directory path or raise a clear error."""
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return directory


def parse_xy(text: str) -> tuple[float, float]:
    """Parse a command-line ``X,Y`` coordinate."""
    try:
        x_text, y_text = text.split(",", maxsplit=1)
        return float(x_text), float(y_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected an X,Y coordinate, received {text!r}.") from exc
