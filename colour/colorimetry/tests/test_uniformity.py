# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.colorimetry.uniformity` module.
"""

import numpy as np
import unittest

from colour.colorimetry import spectral_uniformity

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['DATA_UNIFORMITY', 'TestSpectralUniformity']

DATA_UNIFORMITY = np.array([
    9.55142857e-06, 1.14821429e-05, 1.87842857e-05, 2.87114286e-05,
    3.19714286e-05, 3.23428571e-05, 3.38500000e-05, 3.99257143e-05,
    4.13335714e-05, 2.40021429e-05, 5.76214286e-06, 1.47571429e-06,
    9.79285714e-07, 2.00571429e-06, 3.71571429e-06, 5.76785714e-06,
    7.55571429e-06, 7.46357143e-06, 5.74928571e-06, 3.86928571e-06,
    3.54071429e-06, 4.47428571e-06, 5.64357143e-06, 7.63714286e-06,
    1.01714286e-05, 1.22542857e-05, 1.48100000e-05, 1.65171429e-05,
    1.54307143e-05, 1.45364286e-05, 1.40378571e-05, 1.15878571e-05,
    1.07435714e-05, 1.09792857e-05, 1.03985714e-05, 8.29714286e-06,
    6.30571429e-06, 5.09428571e-06, 4.85000000e-06, 5.53714286e-06,
    6.41285714e-06, 7.25928571e-06, 7.77500000e-06, 7.16071429e-06,
    6.66357143e-06, 6.73285714e-06, 7.53071429e-06, 1.07335714e-05,
    1.62342857e-05, 2.25707143e-05, 2.70564286e-05, 2.77814286e-05,
    2.50257143e-05, 1.79664286e-05, 1.05050000e-05, 5.96571429e-06,
    3.64214286e-06, 2.16642857e-06, 1.29357143e-06, 8.36428571e-07,
    7.25000000e-07, 6.39285714e-07, 6.62857143e-07, 8.55714286e-07,
    1.45071429e-06, 2.25428571e-06, 3.41428571e-06, 4.98642857e-06,
    6.49071429e-06, 7.89285714e-06, 9.16642857e-06, 9.95214286e-06,
    9.76642857e-06, 9.31500000e-06, 8.90928571e-06, 8.15785714e-06,
    6.89357143e-06, 5.57214286e-06, 4.45928571e-06, 3.47785714e-06,
    2.76500000e-06, 2.31142857e-06, 1.70928571e-06, 1.17714286e-06,
    9.84285714e-07, 8.82857143e-07, 7.41428571e-07, 7.01428571e-07,
    7.08571429e-07, 6.66428571e-07, 7.59285714e-07, 8.70000000e-07,
    8.27142857e-07, 7.17142857e-07, 6.60000000e-07
])


class TestSpectralUniformity(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.uniformity.spectral_uniformity`
    definition unit tests methods.
    """

    def test_spectral_uniformity(self):
        """
        Tests :func:`colour.colorimetry.uniformity.spectral_uniformity`
        definition.
        """

        from colour.quality.datasets import SDS_TCS

        np.testing.assert_almost_equal(
            spectral_uniformity(SDS_TCS.values()), DATA_UNIFORMITY, decimal=7)


if __name__ == '__main__':
    unittest.main()
