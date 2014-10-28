# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.adaptation.cie1994` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.adaptation import chromatic_adaptation_CIE1994

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestChromaticAdaptationCIE1994']


class TestChromaticAdaptationCIE1994(unittest.TestCase):
    """
    Defines
    :func:`colour.adaptation.cie1994.chromatic_adaptation_CIE1994` definition
    unit tests methods.
    """

    def test_chromatic_adaptation_CIE1994(self):
        """
        Tests :func:`colour.adaptation.cie1994.chromatic_adaptation_CIE1994`
        definition.
        """

        np.testing.assert_almost_equal(
            chromatic_adaptation_CIE1994(
                XYZ_1=np.array([28.0, 21.26, 5.27]),
                xy_o1=(0.4476, 0.4074),
                xy_o2=(0.3127, 0.3290),
                Y_o=20,
                E_o1=1000,
                E_o2=1000),
            np.array([24.03379521, 21.15621214, 17.64301199]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_CIE1994(
                XYZ_1=np.array([21.77, 19.18, 16.73]),
                xy_o1=(0.3127, 0.3290),
                xy_o2=(0.3127, 0.3290),
                Y_o=50,
                E_o1=100,
                E_o2=1000),
            np.array([21.12891746, 19.42980532, 19.49577765]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_CIE1994(
                XYZ_1=np.array([0.4709771, 0.3495, 0.11301649]) * 100,
                xy_o1=(0.3127, 0.3290),
                xy_o2=(0.4476, 0.4074),
                Y_o=20,
                E_o1=100,
                E_o2=1000),
            np.array([40.55293261, 28.95161939, 4.09480293]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
