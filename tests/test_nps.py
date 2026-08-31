import numpy as np
import pytest

from ctpro.nps import compute_nps_2d, compute_nps_3d, nps_statistics


def test_nps_statistics_accepts_column_vectors() -> None:
    metrics = nps_statistics(
        np.arange(5, dtype=float),
        np.array([[0.0], [2.0], [1.0], [0.1], [0.0]]),
    )
    assert metrics.peak_frequency_per_cm == pytest.approx(1.0)
    assert np.isfinite(metrics.average_frequency_per_cm)


def test_compute_nps_2d_parseval_sanity() -> None:
    rng = np.random.default_rng(4)
    images = rng.normal(0, 10, size=(12, 64, 64))
    result = compute_nps_2d(
        images,
        [[32, 32]],
        roi_size_px=16,
        pixel_spacing_mm=(0.5, 0.5),
        zero_padding_factor=2,
        minimum_fft_size=32,
    )

    dfx = abs(result.frequency_x_per_cm[1] - result.frequency_x_per_cm[0])
    dfy = abs(result.frequency_y_per_cm[1] - result.frequency_y_per_cm[0])
    recovered_sd = np.sqrt(np.sum(result.nps_2d) * dfx * dfy)

    assert recovered_sd == pytest.approx(result.noise_sd_hu, rel=0.05)
    assert np.all(result.nps_2d >= 0)
    assert result.radial_frequency_per_cm[0] == pytest.approx(0)


def test_compute_nps_3d_returns_expected_planes() -> None:
    rng = np.random.default_rng(8)
    images = rng.normal(size=(16, 32, 32))
    result = compute_nps_3d(
        images,
        [[16, 16]],
        voi_size_px=8,
        voxel_spacing_mm=(1, 1, 1),
        num_volumes=2,
        zero_padding_factor=2,
        minimum_fft_size=16,
        max_working_memory_gib=0.1,
    )
    assert result.nps_3d is None
    assert result.num_volumes == 2
    assert set(result.planes) == {
        "xy_center",
        "xy_edge",
        "xy_average",
        "yz_center",
        "yz_edge",
        "yz_average",
        "xz_center",
        "xz_edge",
        "xz_average",
    }
    assert all(plane.shape == (16, 16) for plane in result.planes.values())
