"""Channelized Hotelling Observer (CHO) analysis for SKE/BKE tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import eval_laguerre
from scipy.stats import rankdata

from ._utils import SquareROI, as_image_stack

ChannelKind = Literal["gabor", "laguerre-gauss"]
ResamplingMethod = Literal["bootstrap", "shuffle", "none"]


class ChannelConfiguration(Protocol):
    """Protocol implemented by channel configuration dataclasses."""

    def matrix(self, roi_size_px: int) -> NDArray[np.float64]:
        """Build a pixel-by-channel matrix."""


@dataclass(frozen=True)
class GaborChannels:
    """Gabor channel configuration.

    Frequencies are in cycles per pixel, matching the original CTPro scripts.
    """

    passbands: tuple[tuple[float, float], ...] = (
        (1 / 64, 1 / 32),
        (1 / 32, 1 / 16),
    )
    orientations_rad: tuple[float, ...] = (0.0, math.pi / 2)
    phases_rad: tuple[float, ...] = (0.0,)

    def __post_init__(self) -> None:
        if not self.passbands or not self.orientations_rad or not self.phases_rad:
            raise ValueError("At least one passband, orientation, and phase are required.")
        for low, high in self.passbands:
            if low < 0 or high <= low:
                raise ValueError(f"Invalid Gabor passband {(low, high)}.")

    def matrix(self, roi_size_px: int) -> NDArray[np.float64]:
        x, y = _vectorized_roi_coordinates(roi_size_px)
        filters: list[NDArray[np.float64]] = []
        for low, high in self.passbands:
            center_frequency = (low + high) / 2
            width = 4 * np.log(2) / (np.pi * (high - low))
            for orientation in self.orientations_rad:
                for phase in self.phases_rad:
                    envelope = np.exp(
                        -4 * np.log(2) * (x**2 + y**2) / (width**2)
                    )
                    carrier = np.cos(
                        2
                        * np.pi
                        * center_frequency
                        * (x * np.cos(orientation) + y * np.sin(orientation))
                        + phase
                    )
                    filters.append(envelope * carrier)
        return np.column_stack(filters)


@dataclass(frozen=True)
class LaguerreGaussChannels:
    """Laguerre-Gauss channel configuration compatible with the legacy script."""

    order_count: int = 6
    orientation_count: int = 3

    def __post_init__(self) -> None:
        if self.order_count <= 0 or self.orientation_count <= 0:
            raise ValueError("order_count and orientation_count must be positive.")

    def matrix(self, roi_size_px: int) -> NDArray[np.float64]:
        x, y = _vectorized_roi_coordinates(roi_size_px)
        filters: list[NDArray[np.float64]] = []

        for orientation_index in range(1, self.orientation_count + 1):
            if orientation_index == 1:
                a, b = 8.0, 8.0
            elif orientation_index == 2:
                a, b = 14.0, 5.0
            else:
                a, b = 5.0, 14.0

            ga = 2 * np.pi * ((x / a) ** 2 + (y / b) ** 2)
            for order in range(1, self.order_count + 1):
                filters.append(np.exp(-ga / 2) * eval_laguerre(order, ga))

        return np.column_stack(filters)


@dataclass(frozen=True)
class CHOResult:
    """Distributions and summary template from a CHO analysis."""

    dprime: NDArray[np.float64]
    auc: NDArray[np.float64]
    percent_correct: NDArray[np.float64]
    mean_channel_template: NDArray[np.float64]
    channel_matrix: NDArray[np.float64]
    resampling_method: ResamplingMethod
    internal_noise: float
    covariance_regularization: float
    seed: int | None

    @property
    def mean_dprime(self) -> float:
        return float(np.nanmean(self.dprime))

    @property
    def mean_auc(self) -> float:
        return float(np.nanmean(self.auc))

    @property
    def mean_percent_correct(self) -> float:
        if np.all(np.isnan(self.percent_correct)):
            return float("nan")
        return float(np.nanmean(self.percent_correct))


def _vectorized_roi_coordinates(roi_size_px: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if roi_size_px <= 0:
        raise ValueError("roi_size_px must be positive.")
    coordinate = np.arange(roi_size_px, dtype=float) - (roi_size_px - 1) / 2
    xx, yy = np.meshgrid(coordinate, coordinate)
    # Fortran-order vectorization preserves the channel/image alignment used by
    # the original implementation.
    return xx.reshape(-1, order="F"), yy.reshape(-1, order="F")


def extract_cho_rois(
    images_hu: ArrayLike,
    signal_center_xy: tuple[float, float],
    background_center_xy: tuple[float, float],
    roi_size_px: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract paired signal and background ROI ensembles.

    Returns two arrays with shape ``(samples, rows, columns)``.
    """
    images = as_image_stack(images_hu)
    signal_roi = SquareROI(signal_center_xy, roi_size_px)
    background_roi = SquareROI(background_center_xy, roi_size_px)
    signal_slices = signal_roi.slices(images.shape[1:])
    background_slices = background_roi.slices(images.shape[1:])

    signal = images[:, signal_slices[0], signal_slices[1]]
    background = images[:, background_slices[0], background_slices[1]]
    return np.asarray(signal, dtype=float), np.asarray(background, dtype=float)


def _flatten_ensemble(rois: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(rois, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            "ROI ensembles must have shape (samples, rows, columns); "
            f"received {array.shape}."
        )
    if array.shape[0] < 2:
        raise ValueError("At least two signal and background samples are required.")
    return np.vstack([sample.reshape(-1, order="F") for sample in array])


def _auc_from_scores(signal_scores: NDArray[np.float64], background_scores: NDArray[np.float64]) -> float:
    """Compute empirical AUC using the Mann-Whitney rank statistic."""
    combined = np.concatenate((background_scores, signal_scores))
    ranks = rankdata(combined, method="average")
    n_background = background_scores.size
    n_signal = signal_scores.size
    rank_sum_signal = float(np.sum(ranks[n_background:]))
    u_statistic = rank_sum_signal - n_signal * (n_signal + 1) / 2
    return float(u_statistic / (n_signal * n_background))


def compute_cho(
    signal_rois: ArrayLike,
    background_rois: ArrayLike,
    channels: ChannelConfiguration | ArrayLike,
    *,
    num_resamples: int = 500,
    resampling_method: ResamplingMethod = "bootstrap",
    internal_noise: float = 0.0,
    covariance_regularization: float = 1e-6,
    seed: int | None = None,
) -> CHOResult:
    """Compute CHO d′, empirical AUC, and paired percent correct.

    The calculation is a resubstitution estimate, matching the scope of the
    original scripts. ``covariance_regularization`` is a dimensionless fraction
    of the mean channel variance added to the covariance diagonal. A small
    non-zero value makes the channel covariance inversion reproducible for
    limited ensembles; set it to zero to request an unregularized pseudoinverse.

    ``internal_noise`` is an observer-model parameter and must be calibrated for
    the intended use. The historical example value is not treated as a default.
    """
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive.")
    if resampling_method not in {"bootstrap", "shuffle", "none"}:
        raise ValueError(f"Unsupported resampling_method: {resampling_method!r}.")
    if internal_noise < 0 or covariance_regularization < 0:
        raise ValueError("internal_noise and covariance_regularization must be non-negative.")

    signal = _flatten_ensemble(signal_rois)
    background = _flatten_ensemble(background_rois)
    if signal.shape[1] != background.shape[1]:
        raise ValueError("Signal and background ROIs must have equal pixel counts.")

    roi_side = int(round(math.sqrt(signal.shape[1])))
    if roi_side * roi_side != signal.shape[1]:
        raise ValueError("ROI pixel count is not a square.")

    if hasattr(channels, "matrix"):
        channel_matrix = np.asarray(channels.matrix(roi_side), dtype=float)
    else:
        channel_matrix = np.asarray(channels, dtype=float)
    if channel_matrix.ndim != 2 or channel_matrix.shape[0] != signal.shape[1]:
        raise ValueError(
            "Channel matrix must have shape (pixels, channels); "
            f"expected first dimension {signal.shape[1]}, got {channel_matrix.shape}."
        )
    if channel_matrix.shape[1] == 0:
        raise ValueError("Channel matrix contains no channels.")

    rng = np.random.default_rng(seed)
    dprime = np.empty(num_resamples, dtype=float)
    auc = np.empty(num_resamples, dtype=float)
    percent_correct = np.full(num_resamples, np.nan, dtype=float)
    templates = np.empty((num_resamples, channel_matrix.shape[1]), dtype=float)

    for iteration in range(num_resamples):
        if resampling_method == "bootstrap":
            signal_indices = rng.integers(0, signal.shape[0], size=signal.shape[0])
            background_indices = rng.integers(
                0, background.shape[0], size=background.shape[0]
            )
        elif resampling_method == "shuffle":
            signal_indices = rng.permutation(signal.shape[0])
            background_indices = rng.permutation(background.shape[0])
        else:
            signal_indices = np.arange(signal.shape[0])
            background_indices = np.arange(background.shape[0])

        signal_sample = signal[signal_indices]
        background_sample = background[background_indices]
        signal_channels = channel_matrix.T @ signal_sample.T
        background_channels = channel_matrix.T @ background_sample.T

        mean_difference = np.mean(signal_sample, axis=0) - np.mean(background_sample, axis=0)
        signal_covariance = np.atleast_2d(np.cov(signal_channels, rowvar=True))
        background_covariance = np.atleast_2d(np.cov(background_channels, rowvar=True))
        covariance = 0.5 * (signal_covariance + background_covariance)

        mean_variance = float(np.trace(covariance) / covariance.shape[0])
        regularizer = covariance_regularization * max(mean_variance, np.finfo(float).eps)
        covariance = covariance + regularizer * np.eye(covariance.shape[0])

        template = np.linalg.pinv(covariance, hermitian=True) @ channel_matrix.T @ mean_difference
        templates[iteration] = template

        signal_scores = np.asarray(template @ signal_channels, dtype=float).ravel()
        background_scores = np.asarray(template @ background_channels, dtype=float).ravel()

        if internal_noise > 0:
            noise_scale = internal_noise * float(np.std(background_scores))
            signal_scores = signal_scores + rng.normal(0.0, noise_scale, signal_scores.size)
            background_scores = background_scores + rng.normal(
                0.0, noise_scale, background_scores.size
            )

        pooled_variance = 0.5 * (
            float(np.var(signal_scores)) + float(np.var(background_scores))
        )
        mean_score_difference = float(np.mean(signal_scores) - np.mean(background_scores))
        dprime[iteration] = (
            abs(mean_score_difference) / math.sqrt(pooled_variance)
            if pooled_variance > 0
            else float("inf") if mean_score_difference != 0 else 0.0
        )
        auc[iteration] = _auc_from_scores(signal_scores, background_scores)

        if signal_scores.size == background_scores.size:
            percent_correct[iteration] = float(
                np.mean(signal_scores > background_scores)
                + 0.5 * np.mean(signal_scores == background_scores)
            )

    return CHOResult(
        dprime=dprime,
        auc=auc,
        percent_correct=percent_correct,
        mean_channel_template=np.mean(templates, axis=0),
        channel_matrix=channel_matrix,
        resampling_method=resampling_method,
        internal_noise=float(internal_noise),
        covariance_regularization=float(covariance_regularization),
        seed=seed,
    )


def analyze_cho(
    images_hu: ArrayLike,
    signal_center_xy: tuple[float, float],
    background_center_xy: tuple[float, float],
    roi_size_px: int,
    channels: ChannelConfiguration | ArrayLike | None = None,
    **kwargs: object,
) -> CHOResult:
    """Extract signal/background ensembles and run :func:`compute_cho`."""
    if channels is None:
        channels = GaborChannels()
    signal, background = extract_cho_rois(
        images_hu,
        signal_center_xy,
        background_center_xy,
        roi_size_px,
    )
    return compute_cho(signal, background, channels, **kwargs)


def select_cho_rois(
    ensemble_average_hu: ArrayLike,
    roi_size_px: int,
    *,
    window_min: float = 50.0,
    window_max: float = 150.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Select signal then background ROI centers with two mouse clicks."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = np.asarray(ensemble_average_hu, dtype=float)
    if image.ndim != 2:
        raise ValueError("ensemble_average_hu must be a 2-D image.")

    figure, axes = plt.subplots(figsize=(8, 8))
    axes.imshow(image, cmap="gray", vmin=window_min, vmax=window_max)
    axes.set_title("Click signal center, then background center")
    selected = plt.ginput(2, timeout=-1)
    if len(selected) != 2:
        plt.close(figure)
        raise RuntimeError("Two ROI centers were not selected.")

    signal_center = (float(selected[0][0]), float(selected[0][1]))
    background_center = (float(selected[1][0]), float(selected[1][1]))
    SquareROI(signal_center, roi_size_px).slices(image.shape)
    SquareROI(background_center, roi_size_px).slices(image.shape)

    for center, color, label in (
        (signal_center, "red", "Signal"),
        (background_center, "blue", "Background"),
    ):
        axes.add_patch(
            Rectangle(
                (center[0] - roi_size_px / 2, center[1] - roi_size_px / 2),
                roi_size_px,
                roi_size_px,
                edgecolor=color,
                facecolor="none",
                linewidth=2,
                label=label,
            )
        )
    axes.legend()
    figure.canvas.draw_idle()
    plt.show()
    return signal_center, background_center
