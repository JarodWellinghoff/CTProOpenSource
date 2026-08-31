"""Thin-wire Point Spread Function and Modulation Transfer Function analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import warnings

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator

from ._utils import (
    SquareROI,
    as_image_stack,
    first_descending_crossing,
    radial_average,
    validate_pixel_spacing,
)


@dataclass(frozen=True)
class MTFMetrics:
    """Key metrics from a thin-wire MTF curve."""

    fwhm_cm: float
    mtf50_per_cm: float
    mtf10_per_cm: float
    mtf02_per_cm: float


@dataclass(frozen=True)
class SliceMTFResult:
    """MTF result for one image."""

    psf_2d: NDArray[np.float64]
    psf_radius_cm: NDArray[np.float64]
    psf_radial: NDArray[np.float64]
    mtf_2d: NDArray[np.float64]
    frequency_per_cm: NDArray[np.float64]
    mtf_radial: NDArray[np.float64]
    metrics: MTFMetrics


@dataclass(frozen=True)
class MTFResult:
    """Average thin-wire MTF result across one or more slices."""

    psf_2d: NDArray[np.float64]
    psf_radius_cm: NDArray[np.float64]
    psf_radial: NDArray[np.float64]
    mtf_2d: NDArray[np.float64]
    frequency_per_cm: NDArray[np.float64]
    mtf_radial: NDArray[np.float64]
    metrics: MTFMetrics
    per_slice_metrics: tuple[MTFMetrics, ...]
    wire_center_xy: tuple[float, float]
    roi_size_px: int
    oversampling_factor: int


def disk_kernel(radius: int) -> NDArray[np.uint8]:
    """Create a disk-shaped morphology kernel."""
    if radius < 2:
        raise ValueError("radius must be at least 2.")
    y, x = np.ogrid[-radius + 1 : radius, -radius + 1 : radius]
    kernel = (x**2 + y**2 <= (radius - 1) ** 2).astype(np.uint8)
    return kernel


def find_mtf_level(
    frequency_per_cm: ArrayLike,
    mtf_curve: ArrayLike,
    level: float,
) -> float:
    """Return the first linearly interpolated downward MTF crossing."""
    if not 0 < level < 1:
        raise ValueError("level must be between zero and one.")
    return first_descending_crossing(frequency_per_cm, mtf_curve, level)


def detect_wire_center(
    image_hu: ArrayLike,
    roi_size_px: int,
    *,
    dilation_radius_px: int = 9,
    hough_min_radius_px: int = 7,
    hough_max_radius_px: int | None = None,
) -> tuple[float, float]:
    """Detect a bright thin wire and return ``(x, y)`` pixel coordinates.

    A Hough-circle candidate is refined to the local maximum. If Hough
    detection fails, the global maximum is used with a warning.
    """
    image = np.asarray(image_hu, dtype=float)
    if image.ndim != 2:
        raise ValueError("image_hu must be a 2-D image.")
    if roi_size_px <= 0:
        raise ValueError("roi_size_px must be positive.")
    if hough_max_radius_px is None:
        hough_max_radius_px = roi_size_px

    dilated = cv2.dilate(image.astype(np.float32), disk_kernel(dilation_radius_px))
    normalized = cv2.normalize(
        dilated,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    circles = None
    for threshold in range(30, 0, -2):
        circles = cv2.HoughCircles(
            normalized,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=100,
            param2=threshold,
            minRadius=hough_min_radius_px,
            maxRadius=hough_max_radius_px,
        )
        if circles is not None:
            break

    if circles is None:
        warnings.warn(
            "Hough detection did not find a wire; using the brightest image pixel.",
            RuntimeWarning,
            stacklevel=2,
        )
        row, col = np.unravel_index(np.argmax(image), image.shape)
        center = (float(col), float(row))
    else:
        candidates = np.asarray(circles[0, :, :2], dtype=float)
        candidate_values = []
        for col, row in candidates:
            row_i = int(np.clip(round(row), 0, image.shape[0] - 1))
            col_i = int(np.clip(round(col), 0, image.shape[1] - 1))
            candidate_values.append(image[row_i, col_i])
        col, row = candidates[int(np.argmax(candidate_values))]

        half = roi_size_px // 2
        row_start = max(0, int(round(row)) - half)
        row_stop = min(image.shape[0], int(round(row)) + half + 1)
        col_start = max(0, int(round(col)) - half)
        col_stop = min(image.shape[1], int(round(col)) + half + 1)
        local = image[row_start:row_stop, col_start:col_stop]
        local_row, local_col = np.unravel_index(np.argmax(local), local.shape)
        center = (float(col_start + local_col), float(row_start + local_row))

    SquareROI(center, roi_size_px).slices(image.shape)
    return center


def _oversample_roi(roi: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    if factor < 1:
        raise ValueError("factor must be positive.")
    rows, cols = roi.shape
    output_rows = rows * factor - 3
    output_cols = cols * factor - 3
    if output_rows < 2 or output_cols < 2:
        raise ValueError("ROI is too small for oversampling.")

    row_axis = np.arange(rows, dtype=float)
    col_axis = np.arange(cols, dtype=float)
    interpolator = RegularGridInterpolator(
        (row_axis, col_axis),
        roi,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    rows_new = np.linspace(0, rows - 1, output_rows)
    cols_new = np.linspace(0, cols - 1, output_cols)
    yy, xx = np.meshgrid(rows_new, cols_new, indexing="ij")
    samples = np.column_stack((yy.ravel(), xx.ravel()))
    return interpolator(samples).reshape(output_rows, output_cols)


def _metrics_from_curves(
    psf_radius_cm: NDArray[np.float64],
    psf_radial: NDArray[np.float64],
    frequency_per_cm: NDArray[np.float64],
    mtf_radial: NDArray[np.float64],
) -> MTFMetrics:
    psf_half_level = 0.5 * (
        float(np.nanmin(psf_radial)) + float(np.nanmax(psf_radial))
    )
    half_radius = first_descending_crossing(psf_radius_cm, psf_radial, psf_half_level)
    fwhm = 2 * half_radius if np.isfinite(half_radius) else float("nan")
    return MTFMetrics(
        fwhm_cm=float(fwhm),
        mtf50_per_cm=find_mtf_level(frequency_per_cm, mtf_radial, 0.50),
        mtf10_per_cm=find_mtf_level(frequency_per_cm, mtf_radial, 0.10),
        mtf02_per_cm=find_mtf_level(frequency_per_cm, mtf_radial, 0.02),
    )


def compute_slice_mtf(
    image_hu: ArrayLike,
    pixel_spacing_mm: Sequence[float],
    wire_center_xy: tuple[float, float],
    roi_size_px: int,
    *,
    oversampling_factor: int | None = None,
    zero_padding_factor: int = 16,
    max_frequency_per_cm: float = 100.0,
    background_hu: float | None = None,
) -> SliceMTFResult:
    """Compute the PSF and MTF for one thin-wire CT image."""
    image = np.asarray(image_hu, dtype=float)
    if image.ndim != 2:
        raise ValueError("image_hu must be a 2-D image.")
    row_mm, col_mm = validate_pixel_spacing(pixel_spacing_mm)
    if not np.isclose(row_mm, col_mm, rtol=0, atol=1e-6):
        raise ValueError("Radial thin-wire MTF currently requires square pixels.")
    if roi_size_px < 4 or roi_size_px % 2:
        raise ValueError("roi_size_px must be an even integer of at least 4.")
    if zero_padding_factor < 1 or max_frequency_per_cm <= 0:
        raise ValueError("Invalid FFT parameters.")

    if oversampling_factor is None:
        oversampling_factor = max(4, int(160 / roi_size_px))
    if oversampling_factor < 1:
        raise ValueError("oversampling_factor must be positive.")

    roi = SquareROI(wire_center_xy, roi_size_px).extract(image)
    if background_hu is None:
        background_hu = float(np.median(image))
    psf = _oversample_roi(roi - background_hu, oversampling_factor)

    peak_locations = np.argwhere(psf == np.max(psf))
    peak_row, peak_col = np.median(peak_locations, axis=0)
    peak_row_i, peak_col_i = int(round(peak_row)), int(round(peak_col))
    peak_value = float(psf[peak_row_i, peak_col_i])
    if peak_value == 0:
        raise ValueError("The oversampled PSF has a zero peak and cannot be normalized.")

    sample_spacing_cm = row_mm / 10.0 / oversampling_factor
    y_axis = (np.arange(psf.shape[0]) - peak_row_i) * sample_spacing_cm
    x_axis = (np.arange(psf.shape[1]) - peak_col_i) * sample_spacing_cm
    max_psf_radius = (
        min(
            peak_row_i,
            psf.shape[0] - peak_row_i - 1,
            peak_col_i,
            psf.shape[1] - peak_col_i - 1,
        )
        * sample_spacing_cm
    )
    psf_radius, psf_radial = radial_average(
        psf,
        y_axis,
        x_axis,
        bin_width=sample_spacing_cm,
        max_radius=max_psf_radius,
    )
    psf_radial = psf_radial / peak_value

    nfft_row = psf.shape[0] * zero_padding_factor
    nfft_col = psf.shape[1] * zero_padding_factor
    spectrum = np.fft.fftshift(np.fft.fft2(psf, s=(nfft_row, nfft_col)))
    mtf_2d = np.abs(spectrum)
    frequency_y = np.fft.fftshift(np.fft.fftfreq(nfft_row, d=sample_spacing_cm))
    frequency_x = np.fft.fftshift(np.fft.fftfreq(nfft_col, d=sample_spacing_cm))
    frequency, mtf_radial = radial_average(
        mtf_2d,
        frequency_y,
        frequency_x,
        max_radius=min(
            max_frequency_per_cm,
            0.5 / sample_spacing_cm,
        ),
    )
    if mtf_radial[0] == 0:
        raise ValueError("The MTF DC component is zero and cannot be normalized.")
    mtf_radial = mtf_radial / mtf_radial[0]
    mtf_2d = mtf_2d / mtf_2d[nfft_row // 2, nfft_col // 2]

    metrics = _metrics_from_curves(psf_radius, psf_radial, frequency, mtf_radial)
    return SliceMTFResult(
        psf_2d=psf,
        psf_radius_cm=psf_radius,
        psf_radial=psf_radial,
        mtf_2d=mtf_2d,
        frequency_per_cm=frequency,
        mtf_radial=mtf_radial,
        metrics=metrics,
    )


def compute_wire_mtf(
    images_hu: ArrayLike,
    pixel_spacing_mm: Sequence[float],
    wire_center_xy: tuple[float, float],
    roi_size_px: int,
    *,
    oversampling_factor: int | None = None,
    zero_padding_factor: int = 16,
    max_frequency_per_cm: float = 100.0,
) -> MTFResult:
    """Compute and average thin-wire MTF results across slices."""
    images = as_image_stack(images_hu)
    results = [
        compute_slice_mtf(
            image,
            pixel_spacing_mm,
            wire_center_xy,
            roi_size_px,
            oversampling_factor=oversampling_factor,
            zero_padding_factor=zero_padding_factor,
            max_frequency_per_cm=max_frequency_per_cm,
        )
        for image in images
    ]

    reference_shape = results[0].psf_2d.shape
    reference_mtf_shape = results[0].mtf_2d.shape
    if any(result.psf_2d.shape != reference_shape for result in results):
        raise RuntimeError("Per-slice PSF arrays have inconsistent shapes.")
    if any(result.mtf_2d.shape != reference_mtf_shape for result in results):
        raise RuntimeError("Per-slice MTF arrays have inconsistent shapes.")

    average_psf_2d = np.mean([result.psf_2d for result in results], axis=0)
    average_psf_radial = np.mean([result.psf_radial for result in results], axis=0)
    average_mtf_2d = np.mean([result.mtf_2d for result in results], axis=0)
    average_mtf_radial = np.mean([result.mtf_radial for result in results], axis=0)
    frequency = results[0].frequency_per_cm
    psf_radius = results[0].psf_radius_cm
    metrics = _metrics_from_curves(
        psf_radius,
        average_psf_radial,
        frequency,
        average_mtf_radial,
    )

    factor = oversampling_factor or max(4, int(160 / roi_size_px))
    return MTFResult(
        psf_2d=average_psf_2d,
        psf_radius_cm=psf_radius,
        psf_radial=average_psf_radial,
        mtf_2d=average_mtf_2d,
        frequency_per_cm=frequency,
        mtf_radial=average_mtf_radial,
        metrics=metrics,
        per_slice_metrics=tuple(result.metrics for result in results),
        wire_center_xy=(float(wire_center_xy[0]), float(wire_center_xy[1])),
        roi_size_px=roi_size_px,
        oversampling_factor=factor,
    )


def adjust_wire_roi(
    image_hu: ArrayLike,
    wire_center_xy: tuple[float, float],
    roi_size_px: int,
) -> tuple[tuple[float, float], int]:
    """Interactively move and resize a wire ROI."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = np.asarray(image_hu, dtype=float)
    if image.ndim != 2:
        raise ValueError("image_hu must be 2-D.")
    SquareROI(wire_center_xy, roi_size_px).slices(image.shape)

    class _Viewer:
        def __init__(self) -> None:
            self.center = np.asarray(wire_center_xy, dtype=float)
            self.size = int(roi_size_px)
            self.dragging = False
            self.previous: tuple[float, float] | None = None
            self.figure, self.axes = plt.subplots()
            self._draw()
            self.figure.canvas.mpl_connect("button_press_event", self._press)
            self.figure.canvas.mpl_connect("button_release_event", self._release)
            self.figure.canvas.mpl_connect("motion_notify_event", self._motion)
            self.figure.canvas.mpl_connect("scroll_event", self._scroll)

        def _draw(self) -> None:
            self.axes.clear()
            self.axes.imshow(image, cmap="gray")
            self.axes.add_patch(
                Rectangle(
                    (
                        self.center[0] - self.size / 2,
                        self.center[1] - self.size / 2,
                    ),
                    self.size,
                    self.size,
                    edgecolor="red",
                    facecolor="none",
                    linewidth=2,
                )
            )
            self.axes.set_title("Middle-drag to move; scroll to resize; close to accept")
            self.axes.set_axis_off()
            self.figure.canvas.draw_idle()

        def _press(self, event: object) -> None:
            if (
                getattr(event, "button", None) != 2
                or getattr(event, "inaxes", None) is not self.axes
                or getattr(event, "xdata", None) is None
                or getattr(event, "ydata", None) is None
            ):
                return
            x, y = float(event.xdata), float(event.ydata)
            if abs(x - self.center[0]) <= self.size / 2 and abs(y - self.center[1]) <= self.size / 2:
                self.dragging = True
                self.previous = (x, y)

        def _release(self, event: object) -> None:
            self.dragging = False
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
            candidate = self.center + np.array([x - self.previous[0], y - self.previous[1]])
            try:
                SquareROI((float(candidate[0]), float(candidate[1])), self.size).slices(
                    image.shape
                )
            except ValueError:
                return
            self.center = candidate
            self.previous = (x, y)
            self._draw()

        def _scroll(self, event: object) -> None:
            if getattr(event, "inaxes", None) is not self.axes:
                return
            delta = 2 if getattr(event, "button", None) == "up" else -2
            candidate = max(4, self.size + delta)
            candidate += candidate % 2
            try:
                SquareROI((float(self.center[0]), float(self.center[1])), candidate).slices(
                    image.shape
                )
            except ValueError:
                return
            self.size = candidate
            self._draw()

    viewer = _Viewer()
    plt.show()
    return (float(viewer.center[0]), float(viewer.center[1])), viewer.size


def plot_mtf(result: MTFResult, *, show: bool = False) -> tuple[object, object, object]:
    """Create PSF, 2-D MTF, and radial MTF figures."""
    import matplotlib.pyplot as plt

    psf_figure, psf_axes = plt.subplots(figsize=(5, 4))
    psf_axes.imshow(result.psf_2d, cmap="gray")
    psf_axes.set_title("Average oversampled PSF")
    psf_axes.set_axis_off()

    mtf_2d_figure, mtf_2d_axes = plt.subplots(figsize=(5, 4))
    mtf_2d_axes.imshow(result.mtf_2d, cmap="gray")
    mtf_2d_axes.set_title("Average 2-D MTF")
    mtf_2d_axes.set_axis_off()

    curve_figure, curve_axes = plt.subplots(figsize=(6, 4))
    curve_axes.plot(result.frequency_per_cm, result.mtf_radial, label="Radial average")
    for level, frequency, label in (
        (0.50, result.metrics.mtf50_per_cm, "MTF50"),
        (0.10, result.metrics.mtf10_per_cm, "MTF10"),
        (0.02, result.metrics.mtf02_per_cm, "MTF2"),
    ):
        curve_axes.axhline(level, linestyle="--", alpha=0.5)
        if np.isfinite(frequency):
            curve_axes.axvline(frequency, linestyle="--", alpha=0.5, label=f"{label}: {frequency:.2f} 1/cm")
    curve_axes.set_ylim(0, 1.1)
    curve_axes.set_xlabel("Spatial frequency (1/cm)")
    curve_axes.set_ylabel("MTF")
    curve_axes.set_title("Thin-wire MTF")
    curve_axes.grid(True)
    curve_axes.legend()
    curve_figure.tight_layout()

    if show:
        plt.show()
    return psf_figure, mtf_2d_figure, curve_figure
