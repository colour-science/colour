# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.difference` module.
"""

import numpy as np
import unittest

from colour.difference import delta_E

from colour.utilities import domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestDelta_E',
]


class TestDelta_E(unittest.TestCase):
    """
    Defines :func:`colour.difference.delta_E` definition unit tests methods.
    """

    def test_domain_range_scale_delta_E(self):
        """
        Tests :func:`colour.difference.delta_E` definition domain and range
        scale support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])

        m = ('CIE 1976', 'CIE 1994', 'CIE 2000', 'CMC', 'DIN99')
        v = [delta_E(Lab_1, Lab_2, method) for method in m]

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for method, value in zip(m, v):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_almost_equal(
                        delta_E(Lab_1 * factor, Lab_2 * factor, method),
                        value,
                        decimal=7)


if __name__ == '__main__':
    unittest.main()
