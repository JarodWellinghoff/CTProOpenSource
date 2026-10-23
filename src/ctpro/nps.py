"""Noise Power Spectrum (NPS) analysis for uniform CT phantom images.

The public API separates numerical computation from DICOM loading and plotting.
Image arrays use ``(slices, rows, columns)`` order, while ROI centers use
Cartesian ``(x, y) = (column, row)`` coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._utils import (
    SquareROI,
    as_image_stack,
    first_descending_crossing,
    radial_average,
    validate_pixel_spacing,
)


@dataclass(frozen=True)
class NPSMetrics:
    """Scalar metrics derived from a one-dimensional NPS curve."""

    average_frequency_per_cm: float
    peak_frequency_per_cm: float
    ten_percent_frequency_per_cm: float
    low_frequency_slope: float


@dataclass(frozen=True)
class NPS2DResult:
    """Result of a 2-D NPS analysis."""

    nps_2d: NDArray[np.float64]
    frequency_x_per_cm: NDArray[np.float64]
    frequency_y_per_cm: NDArray[np.float64]
    radial_frequency_per_cm: NDArray[np.float64]
    radial_nps: NDArray[np.float64]
    noise_sd_hu: float
    metrics: NPSMetrics
    roi_centers_xy: NDArray[np.float64]
    roi_size_px: int


@dataclass(frozen=True)
class NPS3DResult:
    """Result of a 3-D NPS analysis."""

    nps_3d: NDArray[np.float64] | None
    planes: Mapping[str, NDArray[np.float64]]
    frequency_x_per_cm: NDArray[np.float64]
    frequency_y_per_cm: NDArray[np.float64]
    frequency_z_per_cm: NDArray[np.float64]
    radial_frequency_per_cm: NDArray[np.float64]
    radial_nps: NDArray[np.float64]
    noise_sd_hu: float
    metrics: NPSMetrics
    roi_centers_xy: NDArray[np.float64]
    voi_size_px: int
    num_volumes: int


def detrend_plane(
    image: ArrayLike,
    pixel_spacing_mm: Sequence[float] = (1.0, 1.0),
) -> NDArray[np.float64]:
    """Fit and subtract a least-squares plane from a 2-D ROI."""
    roi = np.asarray(image, dtype=float)
    if roi.ndim != 2:
        raise ValueError(f"Expected a 2-D ROI, received shape {roi.shape}.")
    row_mm, col_mm = validate_pixel_spacing(pixel_spacing_mm)

    y = (np.arange(roi.shape[0]) - (roi.shape[0] - 1) / 2) * row_mm
    x = (np.arange(roi.shape[1]) - (roi.shape[1] - 1) / 2) * col_mm
    xx, yy = np.meshgrid(x, y)
    design = np.column_stack((xx.ravel(), yy.ravel(), np.ones(roi.size)))
    coefficients, *_ = np.linalg.lstsq(design, roi.ravel(), rcond=None)
    trend = (design @ coefficients).reshape(roi.shape)
    return roi - trend


def get_overlap(x1, x2, y1, y2):
    overlap_w = np.maximum(
        0,
        np.minimum(x2[0], x2[1]) - np.maximum(x1[0], x1[1]),
    )
    overlap_h = np.maximum(
        0,
        np.minimum(y2[0], y2[1]) - np.maximum(y1[0], y1[1]),
    )
    return overlap_w * overlap_h


def get_overlaps(x1, x2, y1, y2, idx):
    if len(x1) == idx:
        return []
    overlap_w = np.maximum(
        0,
        np.minimum(x2[idx], x2[idx + 1 :]) - np.maximum(x1[idx], x1[idx + 1 :]),
    )
    overlap_h = np.maximum(
        0,
        np.minimum(y2[idx], y2[idx + 1 :]) - np.maximum(y1[idx], y1[idx + 1 :]),
    )
    return overlap_w * overlap_h


def any_overlap(r, s, n):
    angles = np.arange(0, n + 1) * (2 * np.pi / n)
    xc = np.sin(angles) * r
    yc = np.cos(angles) * r
    x1 = xc - s / 2
    y1 = yc - s / 2
    x2 = xc + s / 2
    y2 = yc + s / 2
    for i in range(n):
        overlap_a = get_overlap(
            [x1[i], x1[i + 1]],
            [x2[i], x2[i + 1]],
            [y1[i], y1[i + 1]],
            [y2[i], y2[i + 1]],
        )
        if np.any(np.round(overlap_a, 6)):
            return True
    return False


def find_high(r, s):
    low = 2
    high = 4
    overlap_high = False
    while not overlap_high:
        low = high
        high *= 2
        overlap_high = any_overlap(r, s, high)
    return low, high


def binary_search(low, high, r, s):

    while low <= high:
        mid = low + (high - low) // 2
        overlap_mid = any_overlap(r, s, mid)
        overlap_mid_plus = any_overlap(r, s, mid + 1)

        if overlap_mid_plus and not overlap_mid:
            return mid
        if overlap_mid_plus and overlap_mid:
            high = mid - 1
        else:
            low = mid + 1




def normalized_min_separation(n):
    """
    Return the minimum horizontal-or-vertical center separation,
    divided by radius, for n equally spaced axis-aligned squares.

    For a radius r, squares of side s do not overlap when

        s / r <= normalized_min_separation(n)

    Touching is considered non-overlapping.
    """
    n = np.asarray(n, dtype=np.int64)

    if np.any(n < 2):
        raise ValueError("n must be at least 2.")

    # Encodes the distance between the sampled chord directions and
    # the nearest 45-degree direction.
    phase = np.select(
        [
            n % 8 == 0,
            n % 4 == 0,
            n % 2 == 0,
        ],
        [
            4.0,  # divisible by 8
            0.0,  # divisible by 4 but not 8
            2.0,  # even but not divisible by 4
        ],
        default=1.0,  # odd
    )

    angular_factor = np.cos(np.pi / 4.0 - np.pi * phase / (4.0 * n))

    return 2.0 * np.sin(np.pi / n) * angular_factor


def any_overlap_fast(r, s, n, *, rtol=1e-12):
    """
    Test whether any two squares overlap in positive area.

    Assumes:
      - centers are equally spaced on a circle of radius r;
      - all squares are axis-aligned;
      - all squares have side length s;
      - touching does not count as overlap.
    """
    if r <= 0 or s <= 0:
        raise ValueError("r and s must be positive.")

    if n <= 1:
        return False

    normalized_limit = float(normalized_min_separation(np.asarray(n)))

    # The tolerance prevents floating-point roundoff from classifying
    # exact touching as overlap.
    return s > r * normalized_limit * (1.0 + rtol)


def max_square_counts(
    radii,
    side_lengths,
    *,
    n_max=None,
    rtol=1e-12,
):
    """
    Calculate the maximum square count for every side/radius pair.

    Returns
    -------
    result : ndarray, shape (len(side_lengths), len(radii))
        result[i, j] is the maximum count for:
            side length = side_lengths[i]
            radius      = radii[j]

    Notes
    -----
    If n_max is supplied, results are limited to counts <= n_max.
    Otherwise, a mathematically safe upper bound is calculated.
    """
    radii = np.asarray(radii, dtype=np.float64)
    side_lengths = np.asarray(side_lengths, dtype=np.float64)

    if radii.ndim != 1 or side_lengths.ndim != 1:
        raise ValueError("radii and side_lengths must be one-dimensional.")

    if radii.size == 0 or side_lengths.size == 0:
        return np.empty(
            (side_lengths.size, radii.size),
            dtype=np.int64,
        )

    if not np.all(np.isfinite(radii)):
        raise ValueError("All radii must be finite.")

    if not np.all(np.isfinite(side_lengths)):
        raise ValueError("All side lengths must be finite.")

    if np.any(radii <= 0) or np.any(side_lengths <= 0):
        raise ValueError("All radii and side lengths must be positive.")

    if rtol < 0:
        raise ValueError("rtol cannot be negative.")

    # Geometry is scale invariant, so only s / r matters.
    ratios = side_lengths[:, None] / radii[None, :]

    if n_max is None:
        minimum_ratio = float(ratios.min())

        # normalized_min_separation(n) < 2*pi/n, so no count
        # greater than this bound can possibly fit.
        n_max = max(
            2,
            int(np.ceil(2.0 * np.pi / minimum_ratio)),
        )
    elif n_max < 2:
        raise ValueError("n_max must be at least 2.")

    counts = np.arange(2, n_max + 1, dtype=np.int64)

    thresholds = normalized_min_separation(counts)

    # Treat values differing only by roundoff as touching rather than
    # positive-area overlap.
    thresholds *= 1.0 + rtol

    # thresholds itself is not monotonic. Its suffix maximum is.
    #
    # suffix_max[i] answers:
    # "Is there any feasible count >= counts[i]?"
    suffix_max = np.maximum.accumulate(thresholds[::-1])[::-1]

    # -suffix_max is ascending, as required by searchsorted.
    indices = (
        np.searchsorted(
            -suffix_max,
            -ratios.ravel(),
            side="right",
        )
        - 1
    )

    # If even two squares overlap, only one square fits.
    result = np.ones(ratios.size, dtype=np.int64)

    valid = indices >= 0
    result[valid] = counts[indices[valid]]

    return result.reshape(ratios.shape)


def max_squares(
    radius,
    side_length,
    *,
    n_max=None,
    rtol=1e-12,
):
    """
    Return the maximum number of equally spaced, axis-aligned squares
    centered on a circle without positive-area overlap.
    """
    if not np.isfinite(radius):
        raise ValueError("Radius must be finite.")

    if not np.isfinite(side_length):
        raise ValueError("Side length must be finite.")

    if radius <= 0 or side_length <= 0:
        raise ValueError("Radius and side length must be positive.")

    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be finite and nonnegative.")

    ratio = side_length / radius

    if n_max is None:
        n_max = max(
            2,
            int(np.ceil(2.0 * np.pi * (1.0 + rtol) / ratio)),
        )
    else:
        if isinstance(n_max, (bool, np.bool_)) or not isinstance(
            n_max, (int, np.integer)
        ):
            raise TypeError("n_max must be an integer.")

        n_max = int(n_max)

        if n_max < 2:
            raise ValueError("n_max must be at least 2.")

    counts = np.arange(2, n_max + 1, dtype=np.int64)

    thresholds = normalized_min_separation(counts) * (1.0 + rtol)

    feasible = ratio <= thresholds

    return int(
        np.max(
            counts,
            where=feasible,
            initial=1,
        )
    )


def place_rois_on_circle(
    image_shape: Sequence[int],
    radius_px: float,
    roi_size_px: int,
    *,
    count: int | None = None,
    center_xy: tuple[float, float] | None = None,
    phase_rad: float = 0.0,
) -> NDArray[np.float64]:
    """Place evenly spaced, axis-aligned square ROIs on a circle.

    When ``count`` is omitted, the largest count whose adjacent centers are at
    least one ROI width apart is used.
    """
    rows, cols = int(image_shape[-2]), int(image_shape[-1])
    if radius_px <= 0:
        raise ValueError("radius_px must be positive.")
    if roi_size_px <= 0:
        raise ValueError("roi_size_px must be positive.")
    if center_xy is None:
        center_xy = ((cols - 1) / 2, (rows - 1) / 2)

    if count is None:
        if roi_size_px >= 2 * radius_px:
            count = 1
        else:
            count = max_squares(radius_px, roi_size_px)
    if count <= 0:
        raise ValueError("count must be positive.")

    angles = phase_rad + np.arange(count, dtype=float) * 2 * np.pi / count
    centers = np.column_stack(
        (
            center_xy[0] + radius_px * np.cos(angles),
            center_xy[1] + radius_px * np.sin(angles),
        )
    )
    _validate_centers(centers, roi_size_px, (rows, cols))
    return centers


def place_rois_on_matrix(
    image_shape: Sequence[int],
    side_length_px: float,
    roi_size_px: int,
    *,
    center_xy: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Place a centered square matrix of non-overlapping ROIs."""
    rows, cols = int(image_shape[-2]), int(image_shape[-1])
    if side_length_px <= 0 or roi_size_px <= 0:
        raise ValueError("side_length_px and roi_size_px must be positive.")
    per_side = int(math.floor(side_length_px / roi_size_px))
    if per_side < 1:
        raise ValueError("side_length_px is smaller than roi_size_px.")
    if center_xy is None:
        center_xy = ((cols - 1) / 2, (rows - 1) / 2)

    offsets = (np.arange(per_side) - (per_side - 1) / 2) * roi_size_px
    x_offsets, y_offsets = np.meshgrid(offsets, offsets)
    centers = np.column_stack(
        (
            center_xy[0] + x_offsets.ravel(),
            center_xy[1] + y_offsets.ravel(),
        )
    )
    _validate_centers(centers, roi_size_px, (rows, cols))
    return centers


def _validate_centers(
    centers_xy: ArrayLike,
    roi_size_px: int,
    image_shape: Sequence[int],
) -> NDArray[np.float64]:
    centers = np.asarray(centers_xy, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] == 0:
        raise ValueError("centers_xy must have shape (number_of_rois, 2).")
    for center in centers:
        SquareROI((float(center[0]), float(center[1])), roi_size_px).slices(image_shape)
    return centers


def nps_statistics(frequency_per_cm: ArrayLike, nps_curve: ArrayLike) -> NPSMetrics:
    """Derive peak, weighted-average, 10%-of-peak, and low-frequency slope metrics."""
    frequency = np.asarray(frequency_per_cm, dtype=float).ravel()
    curve = np.asarray(nps_curve, dtype=float).ravel()
    if frequency.size != curve.size or frequency.size == 0:
        raise ValueError(
            "frequency_per_cm and nps_curve must have equal non-zero length."
        )
    if np.any(curve < 0) or not np.all(np.isfinite(curve)):
        raise ValueError("nps_curve must contain finite, non-negative values.")

    peak_index = int(np.argmax(curve))
    peak_frequency = float(frequency[peak_index])
    total = float(np.sum(curve))
    average_frequency = (
        float(np.sum(frequency * curve) / total) if total > 0 else float("nan")
    )

    ten_percent_frequency = float("nan")
    peak_value = float(curve[peak_index])
    if peak_value > 0 and peak_index < curve.size - 1:
        ten_percent_frequency = first_descending_crossing(
            frequency,
            curve,
            0.10 * peak_value,
            start_index=peak_index,
        )

    number_of_points = min(10, curve.size)
    if number_of_points >= 2:
        low_frequency_slope = float(
            np.polyfit(frequency[:number_of_points], curve[:number_of_points], 1)[0]
        )
    else:
        low_frequency_slope = float("nan")

    return NPSMetrics(
        average_frequency_per_cm=average_frequency,
        peak_frequency_per_cm=peak_frequency,
        ten_percent_frequency_per_cm=ten_percent_frequency,
        low_frequency_slope=low_frequency_slope,
    )


def compute_nps_2d(
    images_hu: ArrayLike,
    roi_centers_xy: ArrayLike,
    roi_size_px: int,
    pixel_spacing_mm: Sequence[float],
    *,
    remove_trend: bool = True,
    zero_padding_factor: int = 8,
    minimum_fft_size: int = 160,
) -> NPS2DResult:
    """Compute a 2-D NPS averaged over slices and ROIs.

    The normalization yields NPS units of ``HU²·cm²``. Integrating the 2-D NPS
    over spatial frequency approximately recovers the detrended noise variance.
    """
    images = as_image_stack(images_hu)
    row_mm, col_mm = validate_pixel_spacing(pixel_spacing_mm)
    centers = _validate_centers(roi_centers_xy, roi_size_px, images.shape[1:])
    if zero_padding_factor < 1 or minimum_fft_size < roi_size_px:
        raise ValueError("Invalid FFT sizing parameters.")

    nfft = max(minimum_fft_size, zero_padding_factor * roi_size_px)
    nps = np.zeros((nfft, nfft), dtype=np.float64)
    noise_levels: list[float] = []

    row_cm = row_mm / 10.0
    col_cm = col_mm / 10.0
    number_of_samples = images.shape[0] * centers.shape[0]

    for image in images:
        for x, y in centers:
            roi = SquareROI((float(x), float(y)), roi_size_px).extract(image)
            residual = (
                detrend_plane(roi, (row_mm, col_mm))
                if remove_trend
                else roi - np.mean(roi)
            )
            noise_levels.append(float(np.std(residual)))
            spectrum = np.fft.fftshift(np.fft.fft2(residual, s=(nfft, nfft)))
            periodogram = (
                np.abs(spectrum) ** 2 * row_cm * col_cm / (roi_size_px * roi_size_px)
            )
            nps += periodogram / number_of_samples

    frequency_y = np.fft.fftshift(np.fft.fftfreq(nfft, d=row_cm))
    frequency_x = np.fft.fftshift(np.fft.fftfreq(nfft, d=col_cm))
    radial_frequency, radial_nps = radial_average(
        nps,
        frequency_y,
        frequency_x,
        max_radius=min(0.5 / row_cm, 0.5 / col_cm),
    )
    metrics = nps_statistics(radial_frequency, radial_nps)

    return NPS2DResult(
        nps_2d=nps,
        frequency_x_per_cm=frequency_x,
        frequency_y_per_cm=frequency_y,
        radial_frequency_per_cm=radial_frequency,
        radial_nps=radial_nps,
        noise_sd_hu=float(np.mean(noise_levels)),
        metrics=metrics,
        roi_centers_xy=centers,
        roi_size_px=roi_size_px,
    )


def compute_nps_3d(
    images_hu: ArrayLike,
    roi_centers_xy: ArrayLike,
    voi_size_px: int,
    voxel_spacing_mm: Sequence[float],
    *,
    num_volumes: int | None = None,
    remove_trend: bool = True,
    zero_padding_factor: int = 8,
    minimum_fft_size: int = 160,
    return_volume: bool = False,
    max_working_memory_gib: float = 2.0,
) -> NPS3DResult:
    """Compute a 3-D NPS from non-overlapping cubic VOIs.

    ``voxel_spacing_mm`` is ``(row_mm, column_mm, slice_mm)``. The method
    requires at least ``voi_size_px`` consecutive slices per independent
    volume. The NPS units are ``HU²·cm³``.
    """
    images = as_image_stack(images_hu)
    if len(voxel_spacing_mm) != 3:
        raise ValueError("voxel_spacing_mm must contain (row_mm, column_mm, slice_mm).")
    row_mm, col_mm = validate_pixel_spacing(voxel_spacing_mm[:2])
    slice_mm = float(voxel_spacing_mm[2])
    if slice_mm <= 0:
        raise ValueError("slice spacing must be positive.")
    centers = _validate_centers(roi_centers_xy, voi_size_px, images.shape[1:])

    available_volumes = images.shape[0] // voi_size_px
    if available_volumes < 1:
        raise ValueError(
            f"At least {voi_size_px} slices are required for one cubic VOI; "
            f"received {images.shape[0]}."
        )
    if num_volumes is None:
        num_volumes = available_volumes
    if num_volumes < 1 or num_volumes > available_volumes:
        raise ValueError(
            f"num_volumes must be between 1 and {available_volumes}; received {num_volumes}."
        )

    nfft = max(minimum_fft_size, zero_padding_factor * voi_size_px)
    estimated_bytes = nfft**3 * 32
    if estimated_bytes > max_working_memory_gib * 1024**3:
        raise MemoryError(
            "Requested 3-D FFT is expected to exceed the configured memory limit: "
            f"nfft={nfft}, estimated working memory={estimated_bytes / 1024**3:.2f} GiB. "
            "Reduce zero padding/VOI size or raise max_working_memory_gib."
        )

    row_cm = row_mm / 10.0
    col_cm = col_mm / 10.0
    slice_cm = slice_mm / 10.0
    nps = np.zeros((nfft, nfft, nfft), dtype=np.float64)
    noise_levels: list[float] = []
    number_of_samples = num_volumes * centers.shape[0]

    for volume_index in range(num_volumes):
        z_start = volume_index * voi_size_px
        z_stop = z_start + voi_size_px
        slab = images[z_start:z_stop]

        for x, y in centers:
            row_slice, col_slice = SquareROI((float(x), float(y)), voi_size_px).slices(
                images.shape[1:]
            )
            voi = np.asarray(slab[:, row_slice, col_slice], dtype=float)
            if remove_trend:
                voi = np.stack(
                    [
                        detrend_plane(slice_image, (row_mm, col_mm))
                        for slice_image in voi
                    ],
                    axis=0,
                )
            else:
                voi = voi - np.mean(voi, axis=(1, 2), keepdims=True)

            noise_levels.append(float(np.std(voi)))
            spectrum = np.fft.fftshift(
                np.fft.fftn(voi, s=(nfft, nfft, nfft), axes=(0, 1, 2)),
                axes=(0, 1, 2),
            )
            periodogram = (
                np.abs(spectrum) ** 2 * row_cm * col_cm * slice_cm / (voi_size_px**3)
            )
            nps += periodogram / number_of_samples

    center = nfft // 2
    quarter = nfft // 4
    planes: dict[str, NDArray[np.float64]] = {
        "xy_center": nps[center, :, :].copy(),
        "xy_edge": nps[-1, :, :].copy(),
        "xy_average": np.mean(nps, axis=0),
        "yz_center": nps[:, :, center].copy(),
        "yz_edge": nps[:, :, quarter].copy(),
        "yz_average": np.mean(nps, axis=2),
        "xz_center": nps[:, center, :].copy(),
        "xz_edge": nps[:, quarter, :].copy(),
        "xz_average": np.mean(nps, axis=1),
    }

    frequency_z = np.fft.fftshift(np.fft.fftfreq(nfft, d=slice_cm))
    frequency_y = np.fft.fftshift(np.fft.fftfreq(nfft, d=row_cm))
    frequency_x = np.fft.fftshift(np.fft.fftfreq(nfft, d=col_cm))
    radial_frequency, radial_nps = radial_average(
        planes["xy_average"],
        frequency_y,
        frequency_x,
        max_radius=min(0.5 / row_cm, 0.5 / col_cm),
    )
    metrics = nps_statistics(radial_frequency, radial_nps)

    return NPS3DResult(
        nps_3d=nps if return_volume else None,
        planes=planes,
        frequency_x_per_cm=frequency_x,
        frequency_y_per_cm=frequency_y,
        frequency_z_per_cm=frequency_z,
        radial_frequency_per_cm=radial_frequency,
        radial_nps=radial_nps,
        noise_sd_hu=float(np.mean(noise_levels)),
        metrics=metrics,
        roi_centers_xy=centers,
        voi_size_px=voi_size_px,
        num_volumes=num_volumes,
    )


def interactively_adjust_rois(
    image_hu: ArrayLike,
    roi_centers_xy: ArrayLike,
    roi_size_px: int,
    *,
    window_level: float = 40.0,
    window_width: float = 300.0,
) -> NDArray[np.float64]:
    """Display movable ROIs and return their final centers.

    Controls: middle-button drag moves all ROIs; left-button drag changes level;
    right-button drag changes width. Close the window to accept.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = np.asarray(image_hu, dtype=float)
    if image.ndim != 2:
        raise ValueError("image_hu must be a 2-D image.")
    centers = _validate_centers(roi_centers_xy, roi_size_px, image.shape).copy()

    class _Viewer:
        def __init__(self) -> None:
            self.centers = centers
            self.level = float(window_level)
            self.width = float(window_width)
            self.dragging = False
            self.button: int | None = None
            self.previous: tuple[float, float] | None = None
            self.figure, self.axes = plt.subplots()
            self._draw()
            self.figure.canvas.mpl_connect("button_press_event", self._press)
            self.figure.canvas.mpl_connect("button_release_event", self._release)
            self.figure.canvas.mpl_connect("motion_notify_event", self._motion)

        def _draw(self) -> None:
            self.axes.clear()
            self.axes.imshow(
                image,
                cmap="gray",
                vmin=self.level - self.width / 2,
                vmax=self.level + self.width / 2,
            )
            for x, y in self.centers:
                self.axes.add_patch(
                    Rectangle(
                        (x - roi_size_px / 2, y - roi_size_px / 2),
                        roi_size_px,
                        roi_size_px,
                        linewidth=1.5,
                        edgecolor="red",
                        facecolor="none",
                    )
                )
            self.axes.set_title(
                "Middle-drag: move ROIs | Left-drag: level | Right-drag: width"
            )
            self.figure.canvas.draw_idle()

        def _inside(self, x: float, y: float) -> bool:
            half = roi_size_px / 2
            return bool(
                np.any(
                    (np.abs(self.centers[:, 0] - x) <= half)
                    & (np.abs(self.centers[:, 1] - y) <= half)
                )
            )

        def _press(self, event: object) -> None:
            if (
                getattr(event, "inaxes", None) is not self.axes
                or getattr(event, "xdata", None) is None
                or getattr(event, "ydata", None) is None
            ):
                return
            x, y = float(event.xdata), float(event.ydata)
            if self._inside(x, y):
                self.dragging = True
                self.button = int(event.button)
                self.previous = (x, y)

        def _release(self, event: object) -> None:
            self.dragging = False
            self.button = None
            self.previous = None

        def _motion(self, event: object) -> None:
            if (
                not self.dragging
                or self.previous is None
                or getattr(event, "inaxes", None) is not self.axes
                or getattr(event, "xdata", None) is None
                or getattr(event, "ydata", None) is None
            ):
                return
            x, y = float(event.xdata), float(event.ydata)
            dx, dy = x - self.previous[0], y - self.previous[1]
            self.previous = (x, y)

            if self.button == 1:
                self.level += dy * 2
            elif self.button == 3:
                self.width = max(1.0, self.width + dx * 4)
            elif self.button == 2:
                candidate = self.centers + np.array([dx, dy])
                try:
                    _validate_centers(candidate, roi_size_px, image.shape)
                except ValueError:
                    return
                self.centers[:] = candidate
            self._draw()

    viewer = _Viewer()
    plt.show()
    return viewer.centers.copy()


def plot_nps(
    result: NPS2DResult | NPS3DResult, *, show: bool = False
) -> tuple[object, object]:
    """Create 2-D and radial NPS figures without forcing a GUI."""
    import matplotlib.pyplot as plt

    image = (
        result.nps_2d
        if isinstance(result, NPS2DResult)
        else result.planes["xy_average"]
    )
    figure_2d, axes_2d = plt.subplots(figsize=(6, 5))
    axes_2d.imshow(image, cmap="gray", origin="lower")
    axes_2d.set_title(
        "2-D NPS" if isinstance(result, NPS2DResult) else "3-D NPS: XY average"
    )
    axes_2d.set_axis_off()

    figure_1d, axes_1d = plt.subplots(figsize=(6, 4))
    axes_1d.plot(result.radial_frequency_per_cm, result.radial_nps)
    axes_1d.set_xlabel("Spatial frequency (1/cm)")
    axes_1d.set_ylabel(
        "NPS (HU²·cm²)" if isinstance(result, NPS2DResult) else "NPS (HU²·cm³)"
    )
    axes_1d.set_title("Radially averaged NPS")
    axes_1d.grid(True)
    figure_1d.tight_layout()

    if show:
        plt.show()
    return figure_2d, figure_1d
