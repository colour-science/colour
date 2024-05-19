# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.utilities.metrics` module."""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import metric_mse, metric_psnr

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

    def test_metric_mse(self):
        """Test :func:`colour.utilities.metrics.metric_mse` definition."""

        a = np.array([0.48222001, 0.31654775, 0.22070353])
        assert metric_mse(a, a) == 0

        b = a * 0.9
        np.testing.assert_allclose(
            metric_mse(a, b),
            0.0012714955474297446,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        b = a * 1.1
        np.testing.assert_allclose(
            metric_mse(a, b),
            0.0012714955474297446,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestMetricPsnr:
    """
    Define :func:`colour.utilities.metrics.metric_psnr` definition unit tests
    methods.
    """

    def test_metric_psnr(self):
        """Test :func:`colour.utilities.metrics.metric_psnr` definition."""

        a = np.array([0.48222001, 0.31654775, 0.22070353])
        assert metric_psnr(a, a) == 0

        b = a * 0.9
        np.testing.assert_allclose(
            metric_psnr(a, b),
            28.956851563141299,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        b = a * 1.1
        np.testing.assert_allclose(
            metric_psnr(a, b),
            28.956851563141296,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
