import numpy as np
import pytest

from ctpro.cho import GaborChannels, compute_cho


def test_gabor_channel_matrix_shape() -> None:
    channels = GaborChannels(
        passbands=((1 / 32, 1 / 16),),
        orientations_rad=(0.0, np.pi / 2),
        phases_rad=(0.0,),
    )
    matrix = channels.matrix(12)
    assert matrix.shape == (12 * 12, 2)
    assert np.all(np.isfinite(matrix))


def test_cho_detects_simple_signal_and_is_reproducible() -> None:
    rng = np.random.default_rng(10)
    sample_count = 80
    size = 12
    background = rng.normal(0, 1, size=(sample_count, size, size))

    axis = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(axis, axis)
    signal_template = np.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
    signal = background + 1.2 * signal_template

    channel_matrix = signal_template.reshape(-1, order="F")[:, None]
    first = compute_cho(
        signal,
        background,
        channel_matrix,
        num_resamples=20,
        seed=42,
    )
    second = compute_cho(
        signal,
        background,
        channel_matrix,
        num_resamples=20,
        seed=42,
    )

    np.testing.assert_allclose(first.dprime, second.dprime)
    np.testing.assert_allclose(first.auc, second.auc)
    assert first.mean_dprime > 1
    assert first.mean_auc > 0.7
    assert 0 <= first.mean_percent_correct <= 1


def test_cho_rejects_unknown_resampling() -> None:
    data = np.zeros((3, 4, 4))
    with pytest.raises(ValueError, match="Unsupported"):
        compute_cho(data, data, np.ones((16, 1)), resampling_method="bad")  # type: ignore[arg-type]
