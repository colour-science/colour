# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.recovery` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from six.moves import zip

from colour.colorimetry import sd_to_XYZ_integration
from colour.recovery import XYZ_to_sd
from colour.utilities import domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_sd']


class TestXYZ_to_sd(unittest.TestCase):
    """
    Defines :func:`colour.recovery.XYZ_to_sd` definition unit tests
    methods.
    """

    def test_domain_range_scale_XYZ_to_sd(self):
        """
        Tests :func:`colour.recovery.XYZ_to_sd` definition domain
        and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        m = ('Smits 1999', 'Meng 2015')
        v = [sd_to_XYZ_integration(XYZ_to_sd(XYZ, method)) for method in m]

        d_r = (('reference', 1, 1), (1, 1, 0.01), (100, 100, 1))
        for method, value in zip(m, v):
            for scale, factor_a, factor_b in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_almost_equal(
                        sd_to_XYZ_integration(
                            XYZ_to_sd(XYZ * factor_a, method=method)),
                        value * factor_b,
                        decimal=7)


if __name__ == '__main__':
    unittest.main()
