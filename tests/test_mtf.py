import numpy as np
import pytest

from ctpro.mtf import compute_slice_mtf, compute_wire_mtf, find_mtf_level


def test_find_mtf_level_interpolates() -> None:
    assert find_mtf_level([0, 1, 2], [1.0, 0.6, 0.2], 0.5) == pytest.approx(1.25)


def test_compute_thin_wire_mtf_from_gaussian_psf() -> None:
    size = 64
    center = (32.0, 32.0)
    y, x = np.indices((size, size))
    sigma_px = 1.5
    image = 1000 * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma_px**2))

    result = compute_slice_mtf(
        image,
        pixel_spacing_mm=(0.2, 0.2),
        wire_center_xy=center,
        roi_size_px=20,
        oversampling_factor=4,
        zero_padding_factor=4,
        max_frequency_per_cm=50,
    )

    assert result.mtf_radial[0] == pytest.approx(1.0)
    assert np.isfinite(result.metrics.mtf50_per_cm)
    assert np.isfinite(result.metrics.fwhm_cm)
    assert 0.04 < result.metrics.fwhm_cm < 0.12


def test_compute_wire_mtf_averages_slices() -> None:
    size = 48
    center = (24.0, 24.0)
    y, x = np.indices((size, size))
    image = 500 * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / 4)
    result = compute_wire_mtf(
        np.stack([image, image]),
        (0.3, 0.3),
        center,
        20,
        oversampling_factor=4,
        zero_padding_factor=2,
        max_frequency_per_cm=30,
    )
    assert len(result.per_slice_metrics) == 2
    assert result.mtf_radial[0] == pytest.approx(1)
