import numpy as np
import pytest

from ctpro.mtfc import find_mtfc_level, rebin_radial_profile


def test_mtfc_level_interpolates() -> None:
    value = find_mtfc_level([0, 2, 4], [1.0, 0.4, 0.0], 0.5)
    assert value == pytest.approx(5 / 3)


def test_radial_rebin_does_not_wrap_last_bin_to_zero() -> None:
    positions = np.array([0.0, 1.0, 2.0])
    radius = np.array([0.0, 1.0, 2.0])
    values = np.array([10.0, 20.0, 30.0])
    output_positions, profile = rebin_radial_profile(radius, values, positions)
    np.testing.assert_allclose(output_positions, positions)
    np.testing.assert_allclose(profile, values)


def test_compute_mtfc_on_synthetic_circular_insert() -> None:
    from scipy.ndimage import gaussian_filter

    from ctpro.mtfc import compute_mtfc

    rng = np.random.default_rng(123)
    size = 128
    yy, xx = np.indices((size, size))
    insert = (xx - 64) ** 2 + (yy - 64) ** 2 <= 10**2
    image = np.zeros((size, size), dtype=float)
    image[insert] = -95.0
    image = gaussian_filter(image, sigma=0.8)
    volume = np.stack(
        [image + rng.normal(0, 2, image.shape) for _ in range(20)],
        axis=0,
    )

    result = compute_mtfc(
        volume,
        (1.0, 1.0),
        (34, 94, 34, 94),
        (-110.0, -80.0),
        detection_upsampling=2,
        insert_radius_range_mm=(7.0, 13.0),
        center_search_steps=5,
        candidate_esf_count=5,
        selected_esf_count=3,
        hann_window_factor=5,
    )

    assert result.insert_center_xy[0] == pytest.approx(64, abs=1)
    assert result.insert_center_xy[1] == pytest.approx(64, abs=1)
    assert result.insert_radius_mm == pytest.approx(10, abs=2)
    assert np.isfinite(result.metrics.mtf50_per_cm)
    assert result.metrics.mtf50_per_cm < result.metrics.mtf10_per_cm
