"""Command-line interface for CTPro image-quality analyses."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ._utils import parse_xy
from .cho import (
    GaborChannels,
    LaguerreGaussChannels,
    analyze_cho,
    select_cho_rois,
)
from .dicom import load_ct_series, load_repeated_ct_series
from .mtf import adjust_wire_roi, compute_wire_mtf, detect_wire_center, plot_mtf
from .mtfc import compute_mtfc, plot_mtfc, select_square_roi
from .nps import (
    compute_nps_2d,
    compute_nps_3d,
    interactively_adjust_rois,
    place_rois_on_circle,
    place_rois_on_matrix,
    plot_nps,
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def _slice_range(arguments: argparse.Namespace) -> tuple[int, int | None]:
    start = int(arguments.start) - 1
    stop = int(arguments.end) if arguments.end is not None else None
    if start < 0:
        raise ValueError("--start is one-based and must be at least 1.")
    if stop is not None and stop <= start:
        raise ValueError("--end must be greater than or equal to --start.")
    return start, stop


def _write_json(summary: dict[str, Any], destination: str | None) -> None:
    text = json.dumps(summary, indent=2, allow_nan=True)
    print(text)
    if destination:
        Path(destination).write_text(text + "\n", encoding="utf-8")


def _add_common_series_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("directory", help="Directory containing one DICOM series.")
    parser.add_argument("--start", type=_positive_int, default=1, help="First slice (one-based).")
    parser.add_argument("--end", type=_positive_int, help="Last slice (one-based, inclusive).")
    parser.add_argument("--json", dest="json_path", help="Write the numerical summary to JSON.")
    parser.add_argument("--npz", dest="npz_path", help="Write numerical arrays to a compressed NPZ.")
    parser.add_argument("--plot", action="store_true", help="Display analysis figures.")


def _nps_command(arguments: argparse.Namespace) -> int:
    start, stop = _slice_range(arguments)
    series = load_ct_series(arguments.directory, start=start, stop=stop)
    row_mm, col_mm = series.pixel_spacing_mm
    if not np.isclose(row_mm, col_mm, atol=1e-6):
        raise ValueError("Automatic ROI layouts currently require square pixels.")

    center = (
        parse_xy(arguments.center)
        if arguments.center
        else ((series.images_hu.shape[2] - 1) / 2, (series.images_hu.shape[1] - 1) / 2)
    )
    if arguments.layout == "circle":
        radius_px = arguments.radius_mm / col_mm
        centers = place_rois_on_circle(
            series.images_hu.shape[1:],
            radius_px,
            arguments.roi_size,
            count=arguments.count,
            center_xy=center,
        )
    else:
        side_px = arguments.side_mm / col_mm
        centers = place_rois_on_matrix(
            series.images_hu.shape[1:],
            side_px,
            arguments.roi_size,
            center_xy=center,
        )

    if arguments.interactive:
        centers = interactively_adjust_rois(
            series.images_hu[0],
            centers,
            arguments.roi_size,
        )

    if arguments.dimension == "2d":
        result = compute_nps_2d(
            series.images_hu,
            centers,
            arguments.roi_size,
            series.pixel_spacing_mm,
            zero_padding_factor=arguments.zero_padding_factor,
            minimum_fft_size=arguments.minimum_fft_size,
        )
        summary = {
            "metric": "NPS",
            "dimension": "2D",
            "slices": int(series.images_hu.shape[0]),
            "rois": int(centers.shape[0]),
            "roi_size_px": arguments.roi_size,
            "noise_sd_hu": result.noise_sd_hu,
            **asdict(result.metrics),
        }
        if arguments.npz_path:
            np.savez_compressed(
                arguments.npz_path,
                nps_2d=result.nps_2d,
                frequency_x_per_cm=result.frequency_x_per_cm,
                frequency_y_per_cm=result.frequency_y_per_cm,
                radial_frequency_per_cm=result.radial_frequency_per_cm,
                radial_nps=result.radial_nps,
                roi_centers_xy=result.roi_centers_xy,
            )
    else:
        if series.slice_spacing_mm is None:
            raise ValueError("Slice spacing could not be determined for 3-D NPS.")
        result = compute_nps_3d(
            series.images_hu,
            centers,
            arguments.roi_size,
            (*series.pixel_spacing_mm, series.slice_spacing_mm),
            num_volumes=arguments.volumes,
            zero_padding_factor=arguments.zero_padding_factor,
            minimum_fft_size=arguments.minimum_fft_size,
            return_volume=bool(arguments.npz_path and arguments.save_full_volume),
            max_working_memory_gib=arguments.max_memory_gib,
        )
        summary = {
            "metric": "NPS",
            "dimension": "3D",
            "slices": int(series.images_hu.shape[0]),
            "rois": int(centers.shape[0]),
            "voi_size_px": arguments.roi_size,
            "num_volumes": result.num_volumes,
            "noise_sd_hu": result.noise_sd_hu,
            **asdict(result.metrics),
        }
        if arguments.npz_path:
            arrays: dict[str, Any] = {
                "frequency_x_per_cm": result.frequency_x_per_cm,
                "frequency_y_per_cm": result.frequency_y_per_cm,
                "frequency_z_per_cm": result.frequency_z_per_cm,
                "radial_frequency_per_cm": result.radial_frequency_per_cm,
                "radial_nps": result.radial_nps,
                "roi_centers_xy": result.roi_centers_xy,
            }
            arrays.update({f"plane_{name}": value for name, value in result.planes.items()})
            if result.nps_3d is not None:
                arrays["nps_3d"] = result.nps_3d
            np.savez_compressed(arguments.npz_path, **arrays)

    _write_json(summary, arguments.json_path)
    if arguments.plot:
        plot_nps(result, show=True)
    return 0


def _cho_command(arguments: argparse.Namespace) -> int:
    start, stop = _slice_range(arguments)
    series = load_repeated_ct_series(arguments.directories, start=start, stop=stop)
    average_image = np.mean(series.images_hu, axis=0)

    if arguments.signal_center and arguments.background_center:
        signal_center = parse_xy(arguments.signal_center)
        background_center = parse_xy(arguments.background_center)
    elif arguments.interactive:
        signal_center, background_center = select_cho_rois(
            average_image,
            arguments.roi_size,
        )
    else:
        raise ValueError(
            "Provide both --signal-center and --background-center, or use --interactive."
        )

    if arguments.channels == "gabor":
        channels = GaborChannels(
            orientations_rad=tuple(math.radians(value) for value in arguments.orientations),
            phases_rad=tuple(math.radians(value) for value in arguments.phases),
        )
    else:
        channels = LaguerreGaussChannels(
            order_count=arguments.lg_orders,
            orientation_count=arguments.lg_orientations,
        )

    result = analyze_cho(
        series.images_hu,
        signal_center,
        background_center,
        arguments.roi_size,
        channels,
        num_resamples=arguments.resamples,
        resampling_method=arguments.resampling,
        internal_noise=arguments.internal_noise,
        covariance_regularization=arguments.regularization,
        seed=arguments.seed,
    )
    summary = {
        "metric": "CHO",
        "samples_per_class": int(series.images_hu.shape[0]),
        "roi_size_px": arguments.roi_size,
        "channel_type": arguments.channels,
        "num_channels": int(result.channel_matrix.shape[1]),
        "resamples": arguments.resamples,
        "resampling_method": result.resampling_method,
        "internal_noise": result.internal_noise,
        "covariance_regularization": result.covariance_regularization,
        "seed": result.seed,
        "mean_dprime": result.mean_dprime,
        "mean_auc": result.mean_auc,
        "mean_percent_correct": result.mean_percent_correct,
    }
    if arguments.npz_path:
        np.savez_compressed(
            arguments.npz_path,
            dprime=result.dprime,
            auc=result.auc,
            percent_correct=result.percent_correct,
            mean_channel_template=result.mean_channel_template,
            channel_matrix=result.channel_matrix,
        )
    _write_json(summary, arguments.json_path)
    return 0


def _mtf_command(arguments: argparse.Namespace) -> int:
    start, stop = _slice_range(arguments)
    series = load_ct_series(arguments.directory, start=start, stop=stop)
    reference = np.mean(series.images_hu, axis=0)

    if arguments.center:
        center = parse_xy(arguments.center)
    else:
        center = detect_wire_center(reference, arguments.roi_size)
    roi_size = arguments.roi_size
    if arguments.interactive:
        center, roi_size = adjust_wire_roi(reference, center, roi_size)

    result = compute_wire_mtf(
        series.images_hu,
        series.pixel_spacing_mm,
        center,
        roi_size,
        oversampling_factor=arguments.oversampling,
        zero_padding_factor=arguments.zero_padding_factor,
        max_frequency_per_cm=arguments.max_frequency,
    )
    summary = {
        "metric": "thin-wire MTF",
        "slices": int(series.images_hu.shape[0]),
        "wire_center_xy": list(result.wire_center_xy),
        "roi_size_px": result.roi_size_px,
        "oversampling_factor": result.oversampling_factor,
        **asdict(result.metrics),
    }
    if arguments.npz_path:
        np.savez_compressed(
            arguments.npz_path,
            psf_2d=result.psf_2d,
            psf_radius_cm=result.psf_radius_cm,
            psf_radial=result.psf_radial,
            mtf_2d=result.mtf_2d,
            frequency_per_cm=result.frequency_per_cm,
            mtf_radial=result.mtf_radial,
        )
    _write_json(summary, arguments.json_path)
    if arguments.plot:
        plot_mtf(result, show=True)
    return 0


def _parse_roi_center_size(text: str, image_shape: Sequence[int]) -> tuple[int, int, int, int]:
    try:
        x_text, y_text, size_text = text.split(",", maxsplit=2)
        x, y, size = float(x_text), float(y_text), int(size_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("--roi must use X,Y,SIZE format.") from exc
    if size <= 0:
        raise ValueError("ROI size must be positive.")
    row_start = int(round(y - size / 2))
    col_start = int(round(x - size / 2))
    row_stop = row_start + size
    col_stop = col_start + size
    if not (0 <= row_start < row_stop <= image_shape[0]):
        raise ValueError("ROI row bounds are outside the image.")
    if not (0 <= col_start < col_stop <= image_shape[1]):
        raise ValueError("ROI column bounds are outside the image.")
    return row_start, row_stop, col_start, col_stop


def _mtfc_command(arguments: argparse.Namespace) -> int:
    start, stop = _slice_range(arguments)
    series = load_ct_series(arguments.directory, start=start, stop=stop)
    average_image = np.mean(series.images_hu, axis=0)

    if arguments.roi:
        roi = _parse_roi_center_size(arguments.roi, average_image.shape)
    elif arguments.interactive:
        roi = select_square_roi(average_image)
    else:
        raise ValueError("Provide --roi X,Y,SIZE or use --interactive.")

    result = compute_mtfc(
        series.images_hu,
        series.pixel_spacing_mm,
        roi,
        (arguments.hu_range[0], arguments.hu_range[1]),
        cnr_threshold=arguments.cnr_threshold,
        hann_window_factor=arguments.window_factor,
        detection_upsampling=arguments.detection_upsampling,
    )
    summary = {
        "metric": "contrast-dependent MTF",
        "slices": int(series.images_hu.shape[0]),
        "roi_bounds": list(result.roi_bounds),
        "insert_center_xy": list(result.insert_center_xy),
        "insert_radius_mm": result.insert_radius_mm,
        "cnr": result.cnr,
        "contrast_class": result.contrast_class,
        "selected_esf_count": result.selected_esf_count,
        **asdict(result.metrics),
    }
    if arguments.npz_path:
        np.savez_compressed(
            arguments.npz_path,
            frequency_per_cm=result.frequency_per_cm,
            mtfc=result.mtfc,
            lower_envelope=result.lower_envelope,
            upper_envelope=result.upper_envelope,
        )
    _write_json(summary, arguments.json_path)
    if arguments.plot:
        plot_mtfc(result, show=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="ctpro",
        description="Research-oriented CT image-quality analysis tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    nps_parser = subparsers.add_parser("nps", help="Compute 2-D or 3-D NPS.")
    _add_common_series_arguments(nps_parser)
    nps_parser.add_argument("--dimension", choices=("2d", "3d"), default="2d")
    nps_parser.add_argument("--layout", choices=("circle", "matrix"), default="circle")
    nps_parser.add_argument("--roi-size", type=_positive_int, default=40)
    nps_parser.add_argument("--radius-mm", type=float, default=120.0)
    nps_parser.add_argument("--side-mm", type=float, default=300.0)
    nps_parser.add_argument("--count", type=_positive_int)
    nps_parser.add_argument("--center", help="Pattern center as X,Y pixels.")
    nps_parser.add_argument("--interactive", action="store_true")
    nps_parser.add_argument("--volumes", type=_positive_int)
    nps_parser.add_argument("--zero-padding-factor", type=_positive_int, default=8)
    nps_parser.add_argument("--minimum-fft-size", type=_positive_int, default=160)
    nps_parser.add_argument("--max-memory-gib", type=float, default=2.0)
    nps_parser.add_argument(
        "--save-full-volume",
        action="store_true",
        help="Include the full 3-D NPS array in --npz output.",
    )
    nps_parser.set_defaults(function=_nps_command)

    cho_parser = subparsers.add_parser("cho", help="Compute CHO figures of merit.")
    cho_parser.add_argument(
        "directories",
        nargs="+",
        help="One or more repeated-scan DICOM directories.",
    )
    cho_parser.add_argument("--start", type=_positive_int, default=1)
    cho_parser.add_argument("--end", type=_positive_int)
    cho_parser.add_argument("--signal-center", help="Signal center as X,Y pixels.")
    cho_parser.add_argument("--background-center", help="Background center as X,Y pixels.")
    cho_parser.add_argument("--interactive", action="store_true")
    cho_parser.add_argument("--roi-size", type=_positive_int, required=True)
    cho_parser.add_argument("--channels", choices=("gabor", "laguerre-gauss"), default="gabor")
    cho_parser.add_argument("--orientations", type=float, nargs="+", default=[0.0, 90.0])
    cho_parser.add_argument("--phases", type=float, nargs="+", default=[0.0])
    cho_parser.add_argument("--lg-orders", type=_positive_int, default=6)
    cho_parser.add_argument("--lg-orientations", type=_positive_int, default=3)
    cho_parser.add_argument("--resamples", type=_positive_int, default=500)
    cho_parser.add_argument(
        "--resampling",
        choices=("bootstrap", "shuffle", "none"),
        default="bootstrap",
    )
    cho_parser.add_argument("--internal-noise", type=float, default=0.0)
    cho_parser.add_argument("--regularization", type=float, default=1e-6)
    cho_parser.add_argument("--seed", type=int)
    cho_parser.add_argument("--json", dest="json_path")
    cho_parser.add_argument("--npz", dest="npz_path")
    cho_parser.set_defaults(function=_cho_command)

    mtf_parser = subparsers.add_parser("mtf", help="Compute thin-wire MTF.")
    _add_common_series_arguments(mtf_parser)
    mtf_parser.add_argument("--center", help="Wire center as X,Y pixels.")
    mtf_parser.add_argument("--roi-size", type=_positive_int, default=20)
    mtf_parser.add_argument("--interactive", action="store_true")
    mtf_parser.add_argument("--oversampling", type=_positive_int)
    mtf_parser.add_argument("--zero-padding-factor", type=_positive_int, default=16)
    mtf_parser.add_argument("--max-frequency", type=float, default=100.0)
    mtf_parser.set_defaults(function=_mtf_command)

    mtfc_parser = subparsers.add_parser("mtfc", help="Compute circular-insert MTFc.")
    _add_common_series_arguments(mtfc_parser)
    mtfc_parser.add_argument("--roi", help="Square ROI as center X,Y,SIZE pixels.")
    mtfc_parser.add_argument("--interactive", action="store_true")
    mtfc_parser.add_argument("--hu-range", type=float, nargs=2, required=True, metavar=("LOW", "HIGH"))
    mtfc_parser.add_argument("--cnr-threshold", type=float, default=15.0)
    mtfc_parser.add_argument("--window-factor", type=float, default=15.0)
    mtfc_parser.add_argument("--detection-upsampling", type=_positive_int, default=4)
    mtfc_parser.set_defaults(function=_mtfc_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.function(arguments))
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, MemoryError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
