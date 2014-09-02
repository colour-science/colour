#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.common` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import XYZ_to_sRGB

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_sRGB']


class TestXYZ_to_sRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.common.XYZ_to_sRGB` definition unit tests
    methods.
    """

    def test_XYZ_to_RGB(self):
        """
        Tests :func:`colour.models.common.XYZ_to_sRGB` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_sRGB(np.array([0.1180583421, 0.1034, 0.0515089229])),
            np.array([0.48224885, 0.31651974, 0.22070513]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_sRGB(np.array([0.21341854, 0.19387471, 0.16653476])),
            np.array([0.59313312, 0.44141487, 0.42141429]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_sRGB(np.array([0, 0, 0]),
                        (0.34567, 0.35850)),
            np.array([0., 0., 0.]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_sRGB(np.array([0.1180583421, 0.1034, 0.0515089229]),
                        illuminant=(0.32168, 0.33767)),
            np.array([0.47572655, 0.31766531, 0.23306098]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_sRGB(np.array([0.1180583421, 0.1034, 0.0515089229]),
                        chromatic_adaptation_method='Bradford'),
            np.array([0.48224885, 0.31651974, 0.22070513]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_sRGB(np.array([0.1180583421, 0.1034, 0.0515089229]),
                        transfer_function=False),
            np.array([0.19797725, 0.08168657, 0.03992654]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
