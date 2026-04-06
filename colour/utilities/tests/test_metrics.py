"""Define the unit tests for the :mod:`colour.utilities.metrics` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType


from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_ndarray,
    metric_mse,
    metric_psnr,
    xp_as_array,
    xp_assert_close,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestMetricMse",
    "TestMetricPsnr",
]


class TestMetricMse:
    """
    Define :func:`colour.utilities.metrics.metric_mse` definition unit tests
    methods.
    """

    def test_metric_mse(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.metrics.metric_mse` definition."""

        a = xp_as_array([0.48222001, 0.31654775, 0.22070353], xp=xp)
        assert as_ndarray(metric_mse(a, a)) == 0

        b = a * 0.9
        xp_assert_close(
            metric_mse(a, b),
            0.0012714955474297446,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        b = a * 1.1
        xp_assert_close(
            metric_mse(a, b),
            0.0012714955474297446,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestMetricPsnr:
    """
    Define :func:`colour.utilities.metrics.metric_psnr` definition unit tests
    methods.
    """

    def test_metric_psnr(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.metrics.metric_psnr` definition."""

        a = xp_as_array([0.48222001, 0.31654775, 0.22070353], xp=xp)
        assert as_ndarray(metric_psnr(a, a)) == 0

        b = a * 0.9
        xp_assert_close(
            metric_psnr(a, b),
            28.956851563141299,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        b = a * 1.1
        xp_assert_close(
            metric_psnr(a, b),
            28.956851563141296,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
