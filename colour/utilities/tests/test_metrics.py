# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.utilities.metrics` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.utilities import metric_mse, metric_psnr

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestMetricMse', 'TestMetricPsnr']


class TestMetricMse(unittest.TestCase):
    """
    Defines :func:`colour.utilities.metrics.metric_mse` definition unit tests
    methods.
    """

    def test_metric_mse(self):
        """
        Tests :func:`colour.utilities.metrics.metric_mse` definition.
        """

        a = np.array([0.48222001, 0.31654775, 0.22070353])
        self.assertEqual(metric_mse(a, a), 0)

        b = a * 0.9
        self.assertAlmostEqual(
            metric_mse(a, b), 0.0012714955474297446, places=7)

        b = a * 1.1
        self.assertAlmostEqual(
            metric_mse(a, b), 0.0012714955474297446, places=7)


class TestMetricPsnr(unittest.TestCase):
    """
    Defines :func:`colour.utilities.metrics.metric_psnr` definition unit tests
    methods.
    """

    def test_metric_psnr(self):
        """
        Tests :func:`colour.utilities.metrics.metric_psnr` definition.
        """

        a = np.array([0.48222001, 0.31654775, 0.22070353])
        self.assertEqual(metric_psnr(a, a), np.inf)

        b = a * 0.9
        self.assertAlmostEqual(metric_psnr(a, b), 28.956851563141299, places=7)

        b = a * 1.1
        self.assertAlmostEqual(metric_psnr(a, b), 28.956851563141296, places=7)


if __name__ == '__main__':
    unittest.main()
