# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.recovery` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from six.moves import zip

from colour.colorimetry import spectral_to_XYZ_integration
from colour.recovery import XYZ_to_spectral
from colour.utilities import domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_spectral']


class TestXYZ_to_spectral(unittest.TestCase):
    """
    Defines :func:`colour.recovery.XYZ_to_spectral` definition unit tests
    methods.
    """

    def test_domain_range_scale_XYZ_to_spectral(self):
        """
        Tests :func:`colour.recovery.XYZ_to_spectral` definition domain
        and range scale support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        m = ('Smits 1999', 'Meng 2015')
        v = [
            spectral_to_XYZ_integration(XYZ_to_spectral(XYZ, method))
            for method in m
        ]

        d_r = (('reference', 1, 1), (1, 1, 0.01), (100, 100, 1))
        for method, value in zip(m, v):
            for scale, factor_a, factor_b in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_almost_equal(
                        spectral_to_XYZ_integration(
                            XYZ_to_spectral(XYZ * factor_a, method=method)),
                        value * factor_b,
                        decimal=7)


if __name__ == '__main__':
    unittest.main()
