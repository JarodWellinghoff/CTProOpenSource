"""Contrast-dependent MTF (MTFc) analysis for circular CT phantom inserts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal.windows import hann

from ._utils import as_image_stack, first_descending_crossing, validate_pixel_spacing


@dataclass(frozen=True)
class MTFcMetrics:
    """Key frequencies from a contrast-dependent MTF curve."""

    mtf50_per_cm: float
    mtf10_per_cm: float
    mtf02_per_cm: float


@dataclass(frozen=True)
class MTFcResult:
    """Result of a circular-insert MTFc analysis."""

    frequency_per_cm: NDArray[np.float64]
    mtfc: NDArray[np.float64]
    lower_envelope: NDArray[np.float64]
    upper_envelope: NDArray[np.float64]
    metrics: MTFcMetrics
    cnr: float
    contrast_class: str
    insert_center_xy: tuple[float, float]
    insert_radius_mm: float
    roi_bounds: tuple[int, int, int, int]
    selected_esf_count: int


def close_insert_mask(mask: ArrayLike) -> NDArray[np.bool_]:
    """Apply a 5×5 morphological closing to an insert mask."""
    binary = np.asarray(mask, dtype=bool)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def detect_circular_insert(
    mask: ArrayLike,
    minimum_radius_px: int,
    maximum_radius_px: int,
) -> tuple[tuple[float, float], float]:
    """Detect one circular insert in a binary mask.

    Returns ``((x, y), radius_px)`` relative to the mask.
    """
    if minimum_radius_px <= 0 or maximum_radius_px <= minimum_radius_px:
        raise ValueError("Invalid Hough radius limits.")
    binary = np.asarray(mask, dtype=bool)
    normalized = binary.astype(np.uint8) * 255
    circles = None
    for threshold in range(30, 0, -2):
        circles = cv2.HoughCircles(
            normalized,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=100,
            param2=threshold,
            minRadius=minimum_radius_px,
            maxRadius=maximum_radius_px,
        )
        if circles is not None:
            break
    if circles is None:
        raise ValueError(
            "A circular insert was not detected inside the ROI. Check the ROI, "
            "HU threshold, and expected insert radius."
        )

    x, y, radius = np.asarray(circles[0, 0], dtype=float)
    return (float(x), float(y)), float(radius)


def rebin_radial_profile(
    radius: ArrayLike,
    values: ArrayLike,
    sample_positions: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average values into nearest radial bins and interpolate empty bins."""
    radius_values = np.asarray(radius, dtype=float).ravel()
    data_values = np.asarray(values, dtype=float).ravel()
    positions = np.asarray(sample_positions, dtype=float).ravel()
    if radius_values.size != data_values.size:
        raise ValueError("radius and values must have equal length.")
    if positions.size < 2 or np.any(np.diff(positions) <= 0):
        raise ValueError("sample_positions must be strictly increasing.")

    step = float(np.median(np.diff(positions)))
    indices = np.rint((radius_values - positions[0]) / step).astype(int)
    valid = (indices >= 0) & (indices < positions.size) & np.isfinite(data_values)
    sums = np.bincount(
        indices[valid],
        weights=data_values[valid],
        minlength=positions.size,
    )
    counts = np.bincount(indices[valid], minlength=positions.size)
    profile = np.divide(
        sums,
        counts,
        out=np.full(positions.size, np.nan, dtype=float),
        where=counts > 0,
    )

    finite = np.isfinite(profile)
    if np.count_nonzero(finite) < 2:
        raise ValueError("Too few populated radial bins to form an ESF.")
    profile = np.interp(positions, positions[finite], profile[finite])
    return positions, profile


def find_mtfc_level(
    frequency_per_cm: ArrayLike,
    mtfc_curve: ArrayLike,
    level: float,
) -> float:
    """Return the first linearly interpolated downward MTFc crossing."""
    if not 0 < level < 1:
        raise ValueError("level must be between zero and one.")
    return first_descending_crossing(frequency_per_cm, mtfc_curve, level)


def _logistic(
    x: NDArray[np.float64], amplitude: float, offset: float, slope: float, center: float
) -> NDArray[np.float64]:
    exponent = np.clip(-slope * (x - center), -700, 700)
    return amplitude / (1.0 + np.exp(exponent)) + offset


def _fit_logistic(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    y_range = max(float(np.ptp(y)), np.finfo(float).eps)
    x_range = max(float(np.ptp(x)), np.finfo(float).eps)
    bounds = [
        (-2 * y_range, 2 * y_range),
        (float(np.min(y) - y_range), float(np.max(y) + y_range)),
        (-100 / x_range, 100 / x_range),
        (float(np.min(x)), float(np.max(x))),
    ]

    def objective(parameters: NDArray[np.float64]) -> float:
        residual = y - _logistic(x, *parameters)
        return float(np.sum(residual**2))

    initial = differential_evolution(objective, bounds, seed=3, polish=True).x
    optimized, _ = curve_fit(
        _logistic,
        x,
        y,
        p0=initial,
        maxfev=8000,
        bounds=(
            [item[0] for item in bounds],
            [item[1] for item in bounds],
        ),
    )
    return _logistic(x, *optimized)


def _first_consecutive_above(
    values: NDArray[np.float64], threshold: float, count: int
) -> int:
    run = 0
    for index, value in enumerate(values):
        if value > threshold:
            run += 1
            if run >= count:
                return index - count + 1
        else:
            run = 0
    return 0


def _window_indices(
    distance_mm: ArrayLike,
    left_radius_mm: float,
    right_radius_mm: float,
) -> tuple[int, int]:
    """Map a physical radial edge window onto an ESF sampling axis."""
    distance = np.asarray(distance_mm, dtype=float).ravel()
    if distance.size < 6:
        raise ValueError("At least six ESF samples are required for edge scoring.")
    if np.any(np.diff(distance) <= 0):
        raise ValueError("ESF distance samples must be strictly increasing.")
    if not np.isfinite(left_radius_mm) or not np.isfinite(right_radius_mm):
        raise ValueError("Edge-window radii must be finite.")
    if right_radius_mm <= left_radius_mm:
        raise ValueError("The right edge-window radius must exceed the left radius.")

    left = int(np.searchsorted(distance, left_radius_mm))
    right = int(np.searchsorted(distance, right_radius_mm))
    left = int(np.clip(left, 1, distance.size - 5))
    right = int(np.clip(right, left + 3, distance.size - 1))
    return left, right


def _radial_esf(
    image: NDArray[np.float64],
    center_xy: tuple[float, float],
    pixel_size_mm: float,
    radius_mm: float,
    oversampling: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rows, cols = image.shape
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    distance_map = pixel_size_mm * np.hypot(xx - center_xy[0], yy - center_xy[1])
    mask = distance_map <= radius_mm
    bin_size = pixel_size_mm / oversampling
    sample_positions = np.linspace(
        0,
        radius_mm,
        int(math.floor(radius_mm / bin_size)) + 1,
    )
    distance, esf = rebin_radial_profile(
        distance_map[mask],
        image[mask],
        sample_positions,
    )
    if esf[0] > esf[-1]:
        esf = esf[::-1]

    trim = int(0.05 * distance.size)
    if trim > 0 and distance.size - 2 * trim >= 10:
        distance = distance[trim:-trim]
        esf = esf[trim:-trim]
    return distance, esf


def _esf_to_mtf(
    esf: NDArray[np.float64],
    bin_size_mm: float,
    window_factor: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    maximum = float(np.max(esf))
    minimum = float(np.min(esf))
    midpoint = (maximum + minimum) / 2
    edge_center = int(np.argmin(np.abs(esf - midpoint)))

    high = minimum + 0.85 * (maximum - minimum)
    low = minimum + 0.15 * (maximum - minimum)
    high_indices = np.flatnonzero(esf >= high)
    low_indices = np.flatnonzero(esf <= low)
    if high_indices.size == 0 or low_indices.size == 0:
        raise ValueError("The ESF does not span its 15% and 85% levels.")

    if np.mean(esf[: max(edge_center, 1)]) > np.mean(esf[edge_center:]):
        edge_1 = int(high_indices[-1])
        edge_2 = int(low_indices[0])
    else:
        edge_1 = int(high_indices[0])
        edge_2 = int(low_indices[-1])

    half_window = max(1, int(window_factor * abs(edge_2 - edge_1)))
    lsf = np.diff(esf)
    first = max(edge_center - half_window, 0)
    last = min(edge_center + half_window, lsf.size - 1)
    window = np.zeros_like(lsf)
    length = last - first + 1
    if length > 1:
        window[first : last + 1] = hann(length)
    lsf_windowed = lsf * window

    mtf = np.abs(np.fft.rfft(lsf_windowed))
    if mtf.size == 0 or mtf[0] == 0:
        raise ValueError("The windowed LSF has a zero DC component.")
    mtf = mtf / mtf[0]
    frequency_per_mm = np.fft.rfftfreq(lsf_windowed.size, d=bin_size_mm)
    return frequency_per_mm, lsf_windowed, mtf


def compute_mtfc(
    images_hu: ArrayLike,
    pixel_spacing_mm: Sequence[float],
    roi_bounds: Sequence[int],
    hu_threshold: tuple[float, float],
    *,
    cnr_threshold: float = 15.0,
    hann_window_factor: float = 15.0,
    detection_upsampling: int = 4,
    low_contrast_oversampling: int = 4,
    high_contrast_oversampling: int = 10,
    output_oversampling: int = 4,
    insert_radius_range_mm: tuple[float, float] = (6.0, 20.0),
    center_search_radius_px: float = 4.0,
    center_search_steps: int = 21,
    candidate_esf_count: int = 15,
    selected_esf_count: int = 5,
) -> MTFcResult:
    """Compute MTFc for one circular insert.

    Parameters
    ----------
    roi_bounds:
        ``(row_start, row_stop, column_start, column_stop)`` with exclusive
        stops, matching NumPy slicing.
    hu_threshold:
        Inclusive-lower/exclusive-upper HU range used to segment the insert.
    """
    images = as_image_stack(images_hu)
    row_mm, col_mm = validate_pixel_spacing(pixel_spacing_mm)
    if not np.isclose(row_mm, col_mm, rtol=0, atol=1e-6):
        raise ValueError("Radial MTFc currently requires square pixels.")
    pixel_size_mm = row_mm

    if len(roi_bounds) != 4:
        raise ValueError(
            "roi_bounds must contain row_start, row_stop, col_start, col_stop."
        )
    row_start, row_stop, col_start, col_stop = map(int, roi_bounds)
    if not (0 <= row_start < row_stop <= images.shape[1]):
        raise ValueError("ROI row bounds are outside the image.")
    if not (0 <= col_start < col_stop <= images.shape[2]):
        raise ValueError("ROI column bounds are outside the image.")
    if abs((row_stop - row_start) - (col_stop - col_start)) > 1:
        raise ValueError("MTFc requires an approximately square ROI.")
    if hu_threshold[1] <= hu_threshold[0]:
        raise ValueError("hu_threshold upper value must exceed the lower value.")
    if (
        detection_upsampling < 1
        or low_contrast_oversampling < 1
        or high_contrast_oversampling < 1
        or output_oversampling < 1
    ):
        raise ValueError("Oversampling factors must be positive.")
    if center_search_radius_px < 0:
        raise ValueError("center_search_radius_px must be non-negative.")
    if center_search_steps < 1:
        raise ValueError("center_search_steps must be positive.")
    if candidate_esf_count < 1 or selected_esf_count < 1:
        raise ValueError("ESF candidate and selection counts must be positive.")

    average_image = np.mean(images, axis=0)
    cropped_average = average_image[row_start:row_stop, col_start:col_stop]
    cropped_volume = images[:, row_start:row_stop, col_start:col_stop]

    upsampled = ndimage.zoom(cropped_average, detection_upsampling, order=1)
    binary = (upsampled >= hu_threshold[0]) & (upsampled < hu_threshold[1])
    binary = close_insert_mask(binary)

    min_radius_px = max(
        1,
        int(detection_upsampling * insert_radius_range_mm[0] / pixel_size_mm),
    )
    max_radius_px = max(
        min_radius_px + 1,
        int(detection_upsampling * insert_radius_range_mm[1] / pixel_size_mm),
    )
    center_up, radius_up = detect_circular_insert(binary, min_radius_px, max_radius_px)
    center = np.asarray(center_up, dtype=float) / detection_upsampling
    radius_px = radius_up / detection_upsampling
    insert_radius_mm = radius_px * pixel_size_mm
    analysis_radius_mm = insert_radius_mm * 1.8

    rows, cols = cropped_average.shape
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    distance = pixel_size_mm * np.hypot(xx - center[0], yy - center[1])
    object_mask = distance < insert_radius_mm * 0.9
    background_mask = (distance > insert_radius_mm * 1.1) & (
        distance < analysis_radius_mm
    )
    if not np.any(object_mask) or np.count_nonzero(background_mask) < 2:
        raise ValueError(
            "Detected insert does not leave valid object/background CNR regions."
        )

    middle_slice = cropped_volume[cropped_volume.shape[0] // 2]
    background_sd = float(np.std(middle_slice[background_mask]))
    if background_sd == 0:
        raise ValueError("Background standard deviation is zero; CNR is undefined.")
    cnr = abs(
        float(
            np.mean(middle_slice[object_mask]) - np.mean(middle_slice[background_mask])
        )
        / background_sd
    )

    is_low_contrast = cnr < cnr_threshold
    oversampling = (
        low_contrast_oversampling if is_low_contrast else high_contrast_oversampling
    )

    initial_distance, initial_esf = _radial_esf(
        cropped_average,
        (float(center[0]), float(center[1])),
        pixel_size_mm,
        analysis_radius_mm,
        oversampling,
    )
    centered_esf = initial_esf - np.mean(initial_esf)
    try:
        model = _fit_logistic(initial_distance, centered_esf)
    except Exception:
        model = initial_esf

    lsf = np.diff(model)
    threshold = float(np.max(np.abs(lsf)) * 0.30)
    left = _first_consecutive_above(np.abs(lsf), threshold, 3)
    right = len(lsf) - _first_consecutive_above(np.abs(lsf[::-1]), threshold, 3)
    left = max(1, min(left, len(initial_esf) - 5))
    right = max(left + 3, min(right, len(initial_esf) - 1))
    edge_window_mm = (
        float(initial_distance[left]),
        float(initial_distance[right]),
    )

    candidates: list[
        tuple[
            int,
            float,
            float,
            float,
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ] = []
    moves = np.linspace(
        -center_search_radius_px, center_search_radius_px, center_search_steps
    )
    for dx in moves:
        for dy in moves:
            candidate_distance, candidate_esf = _radial_esf(
                cropped_average,
                (float(center[0] + dx), float(center[1] + dy)),
                pixel_size_mm,
                analysis_radius_mm,
                oversampling,
            )
            candidate_left, candidate_right = _window_indices(
                candidate_distance,
                edge_window_mm[0],
                edge_window_mm[1],
            )
            segment = candidate_esf[candidate_left:candidate_right]
            derivative = np.diff(segment)
            sign_changes = int(
                np.sum(np.sign(derivative[1:]) != np.sign(derivative[:-1]))
            )
            slope = abs(float(np.polyfit(np.arange(segment.size), segment, 1)[0]))
            candidates.append(
                (
                    sign_changes,
                    slope,
                    float(dx),
                    float(dy),
                    candidate_distance,
                    candidate_esf,
                )
            )

    candidates.sort(key=lambda item: (item[0], -item[1]))
    smoothest = candidates[: min(candidate_esf_count, len(candidates))]
    smoothest.sort(key=lambda item: item[1], reverse=True)
    selected = smoothest[: min(selected_esf_count, len(smoothest))]
    if not selected:
        raise RuntimeError("Center refinement did not produce any ESF candidates.")

    target_frequency_per_mm = np.linspace(
        0,
        0.5 / (pixel_size_mm / output_oversampling),
        256,
    )
    curves = []
    for _, _, _, _, _, esf in selected:
        frequency_per_mm, _, mtf = _esf_to_mtf(
            esf,
            pixel_size_mm / oversampling,
            hann_window_factor,
        )
        curves.append(
            interp1d(
                frequency_per_mm,
                mtf,
                bounds_error=False,
                fill_value=0.0,
            )(target_frequency_per_mm)
        )

    curve_array = np.asarray(curves, dtype=float)
    median_curve = np.median(curve_array, axis=0)
    lower = np.min(curve_array, axis=0)
    upper = np.max(curve_array, axis=0)
    frequency_per_cm = target_frequency_per_mm * 10.0

    metrics = MTFcMetrics(
        mtf50_per_cm=find_mtfc_level(frequency_per_cm, median_curve, 0.50),
        mtf10_per_cm=find_mtfc_level(frequency_per_cm, median_curve, 0.10),
        mtf02_per_cm=find_mtfc_level(frequency_per_cm, median_curve, 0.02),
    )
    global_center = (
        float(col_start + center[0]),
        float(row_start + center[1]),
    )
    return MTFcResult(
        frequency_per_cm=frequency_per_cm,
        mtfc=median_curve,
        lower_envelope=lower,
        upper_envelope=upper,
        metrics=metrics,
        cnr=cnr,
        contrast_class="low" if is_low_contrast else "high",
        insert_center_xy=global_center,
        insert_radius_mm=float(insert_radius_mm),
        roi_bounds=(row_start, row_stop, col_start, col_stop),
        selected_esf_count=len(selected),
    )


def select_square_roi(image_hu: ArrayLike) -> tuple[int, int, int, int]:
    """Interactively draw a square ROI and return NumPy-style bounds."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    image = np.asarray(image_hu, dtype=float)
    if image.ndim != 2:
        raise ValueError("image_hu must be 2-D.")
    selection: list[tuple[int, int, int, int]] = []

    figure, axes = plt.subplots(figsize=(8, 8))
    axes.imshow(image, cmap="gray")
    axes.set_title("Draw a square ROI around the insert, then close the window")

    def on_select(click: object, release: object) -> None:
        x1, y1 = float(click.xdata), float(click.ydata)
        x2, y2 = float(release.xdata), float(release.ydata)
        size = max(abs(x2 - x1), abs(y2 - y1))
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        col_start = max(0, int(round(center_x - size / 2)))
        row_start = max(0, int(round(center_y - size / 2)))
        side = max(2, int(round(size)))
        side -= side % 2
        col_stop = min(image.shape[1], col_start + side)
        row_stop = min(image.shape[0], row_start + side)
        side = min(col_stop - col_start, row_stop - row_start)
        side -= side % 2
        selection.clear()
        selection.append((row_start, row_start + side, col_start, col_start + side))

    selector = RectangleSelector(
        axes,
        on_select,
        useblit=True,
        button=[1],
        minspanx=20,
        minspany=20,
        spancoords="data",
        interactive=True,
    )
    # Keep the selector alive until the window closes.
    _ = selector
    plt.show()
    if not selection:
        raise RuntimeError("No ROI was selected.")
    return selection[-1]


def plot_mtfc(result: MTFcResult, *, show: bool = False) -> object:
    """Create an MTFc curve with its candidate envelope."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7, 4))
    axes.plot(result.frequency_per_cm, result.mtfc, label="Median MTFc")
    axes.fill_between(
        result.frequency_per_cm,
        result.lower_envelope,
        result.upper_envelope,
        alpha=0.25,
        label="Selected-ESF range",
    )
    for level, frequency, label in (
        (0.50, result.metrics.mtf50_per_cm, "MTFc50"),
        (0.10, result.metrics.mtf10_per_cm, "MTFc10"),
        (0.02, result.metrics.mtf02_per_cm, "MTFc2"),
    ):
        axes.axhline(level, linestyle="--", alpha=0.5)
        if np.isfinite(frequency):
            axes.axvline(
                frequency,
                linestyle="--",
                alpha=0.5,
                label=f"{label}: {frequency:.2f} lp/cm",
            )
    axes.set_ylim(0, 1.1)
    axes.set_xlabel("Spatial frequency (lp/cm)")
    axes.set_ylabel("MTFc")
    axes.set_title("Contrast-dependent MTF")
    axes.grid(True)
    axes.legend()
    figure.tight_layout()
    if show:
        plt.show()
    return figure
