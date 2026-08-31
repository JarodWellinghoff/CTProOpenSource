import numpy as np
import pytest

from ctpro._utils import SquareROI, first_descending_crossing, radial_average


def test_square_roi_uses_xy_coordinates() -> None:
    image = np.arange(100).reshape(10, 10)
    roi = SquareROI((6, 3), 2).extract(image)
    np.testing.assert_array_equal(roi, image[2:4, 5:7])


def test_square_roi_rejects_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="outside"):
        SquareROI((0, 0), 4).slices((10, 10))


def test_first_descending_crossing_interpolates() -> None:
    value = first_descending_crossing([0, 1, 2], [1.0, 0.6, 0.2], 0.5)
    assert value == pytest.approx(1.25)


def test_radial_average_starts_at_origin() -> None:
    axis = np.arange(-2, 3, dtype=float)
    xx, yy = np.meshgrid(axis, axis)
    radius = np.hypot(xx, yy)
    output_radius, profile = radial_average(radius, axis, axis, bin_width=1.0, max_radius=2)
    np.testing.assert_allclose(output_radius, [0, 1, 2])
    assert profile[0] == pytest.approx(0)
