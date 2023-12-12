# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.colorimetry.uniformity` module."""

from __future__ import annotations

import unittest

import numpy as np

from colour.colorimetry import spectral_uniformity
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.hints import NDArrayFloat

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DATA_UNIFORMITY_FIRST_ORDER_DERIVATIVES",
    "TestSpectralUniformity",
]

DATA_UNIFORMITY_FIRST_ORDER_DERIVATIVES: NDArrayFloat = np.array(
    [
        9.55142857e-06,
        1.14821429e-05,
        1.87842857e-05,
        2.87114286e-05,
        3.19714286e-05,
        3.23428571e-05,
        3.38500000e-05,
        3.99257143e-05,
        4.13335714e-05,
        2.40021429e-05,
        5.76214286e-06,
        1.47571429e-06,
        9.79285714e-07,
        2.00571429e-06,
        3.71571429e-06,
        5.76785714e-06,
        7.55571429e-06,
        7.46357143e-06,
        5.74928571e-06,
        3.86928571e-06,
        3.54071429e-06,
        4.47428571e-06,
        5.64357143e-06,
        7.63714286e-06,
        1.01714286e-05,
        1.22542857e-05,
        1.48100000e-05,
        1.65171429e-05,
        1.54307143e-05,
        1.45364286e-05,
        1.40378571e-05,
        1.15878571e-05,
        1.07435714e-05,
        1.09792857e-05,
        1.03985714e-05,
        8.29714286e-06,
        6.30571429e-06,
        5.09428571e-06,
        4.85000000e-06,
        5.53714286e-06,
        6.41285714e-06,
        7.25928571e-06,
        7.77500000e-06,
        7.16071429e-06,
        6.66357143e-06,
        6.73285714e-06,
        7.53071429e-06,
        1.07335714e-05,
        1.62342857e-05,
        2.25707143e-05,
        2.70564286e-05,
        2.77814286e-05,
        2.50257143e-05,
        1.79664286e-05,
        1.05050000e-05,
        5.96571429e-06,
        3.64214286e-06,
        2.16642857e-06,
        1.29357143e-06,
        8.36428571e-07,
        7.25000000e-07,
        6.39285714e-07,
        6.62857143e-07,
        8.55714286e-07,
        1.45071429e-06,
        2.25428571e-06,
        3.41428571e-06,
        4.98642857e-06,
        6.49071429e-06,
        7.89285714e-06,
        9.16642857e-06,
        9.95214286e-06,
        9.76642857e-06,
        9.31500000e-06,
        8.90928571e-06,
        8.15785714e-06,
        6.89357143e-06,
        5.57214286e-06,
        4.45928571e-06,
        3.47785714e-06,
        2.76500000e-06,
        2.31142857e-06,
        1.70928571e-06,
        1.17714286e-06,
        9.84285714e-07,
        8.82857143e-07,
        7.41428571e-07,
        7.01428571e-07,
        7.08571429e-07,
        6.66428571e-07,
        7.59285714e-07,
        8.70000000e-07,
        8.27142857e-07,
        7.17142857e-07,
        6.60000000e-07,
    ]
)

DATA_UNIFORMITY_SECOND_ORDER_DERIVATIVES: NDArrayFloat = np.array(
    [
        7.97142857e-09,
        3.69285714e-08,
        9.21500000e-08,
        6.66714286e-08,
        6.75428571e-08,
        1.30571429e-07,
        1.83300000e-07,
        8.26071429e-08,
        4.10357143e-08,
        1.64628571e-07,
        1.47007143e-07,
        4.51000000e-08,
        2.17285714e-08,
        1.48071429e-08,
        1.10071429e-08,
        7.44285714e-09,
        2.18571429e-09,
        6.13571429e-09,
        1.97571429e-08,
        2.90714286e-08,
        2.38642857e-08,
        1.47428571e-08,
        1.36000000e-08,
        1.38214286e-08,
        1.03000000e-08,
        1.15142857e-08,
        1.10000000e-08,
        1.16500000e-08,
        2.53071429e-08,
        2.91571429e-08,
        1.73857143e-08,
        1.17571429e-08,
        1.42714286e-08,
        1.35642857e-08,
        6.82142857e-09,
        1.81571429e-08,
        2.31285714e-08,
        1.85857143e-08,
        1.46142857e-08,
        9.45714286e-09,
        5.62142857e-09,
        4.75000000e-09,
        4.68571429e-09,
        8.71428571e-09,
        1.58500000e-08,
        2.12142857e-08,
        3.20785714e-08,
        4.23357143e-08,
        3.73000000e-08,
        2.26357143e-08,
        6.57857143e-09,
        3.39285714e-09,
        1.18071429e-08,
        3.26500000e-08,
        3.33500000e-08,
        1.87428571e-08,
        1.10214286e-08,
        7.07142857e-09,
        4.00000000e-09,
        2.15714286e-09,
        1.74285714e-09,
        1.77857143e-09,
        2.10714286e-09,
        3.57857143e-09,
        4.47142857e-09,
        5.59285714e-09,
        6.45000000e-09,
        6.65000000e-09,
        8.52142857e-09,
        7.58571429e-09,
        4.87857143e-09,
        2.58571429e-09,
        2.62857143e-09,
        1.84285714e-09,
        1.41428571e-09,
        2.24285714e-09,
        3.65714286e-09,
        3.47142857e-09,
        3.28571429e-09,
        2.60000000e-09,
        2.27857143e-09,
        1.84285714e-09,
        2.57142857e-09,
        1.32142857e-09,
        5.57142857e-10,
        6.85714286e-10,
        1.10000000e-09,
        6.42857143e-10,
        2.21428571e-10,
        4.50000000e-10,
        7.07142857e-10,
        2.50000000e-10,
        1.85714286e-10,
        2.14285714e-10,
        2.28571429e-10,
    ]
)


class TestSpectralUniformity(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.uniformity.spectral_uniformity`
    definition unit tests methods.
    """

    def test_spectral_uniformity(self):
        """
        Test :func:`colour.colorimetry.uniformity.spectral_uniformity`
        definition.
        """

        from colour.quality.datasets import SDS_TCS

        np.testing.assert_allclose(
            spectral_uniformity(SDS_TCS.values()),
            DATA_UNIFORMITY_FIRST_ORDER_DERIVATIVES,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            spectral_uniformity(
                SDS_TCS.values(), use_second_order_derivatives=True
            ),
            DATA_UNIFORMITY_SECOND_ORDER_DERIVATIVES,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


if __name__ == "__main__":
    unittest.main()
