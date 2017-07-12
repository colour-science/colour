#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.recovery.meng2015` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry import (STANDARD_OBSERVERS_CMFS, SpectralShape,
                                spectral_to_XYZ_integration)
from colour.recovery import XYZ_to_spectral_Meng2015

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_spectral_Meng2015']


class TestXYZ_to_spectral_Meng2015(unittest.TestCase):
    """
    Defines :func:`colour.recovery.meng2015.XYZ_to_spectral_Meng2015`
    definition unit tests methods.
    """

    def test_XYZ_to_spectral_Meng2015(self):
        """
        Tests :func:`colour.recovery.meng2015.XYZ_to_spectral_Meng2015`
        definition.
        """

        cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        shape = SpectralShape(cmfs.shape.start, cmfs.shape.end, 5)
        cmfs_c = cmfs.clone().align(shape)

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        np.testing.assert_almost_equal(
            spectral_to_XYZ_integration(
                XYZ_to_spectral_Meng2015(XYZ), cmfs=cmfs_c),
            XYZ,
            decimal=7)

        shape = SpectralShape(cmfs.shape.start, cmfs.shape.end, 10)
        cmfs_c = cmfs.clone().align(shape)

        np.testing.assert_almost_equal(
            spectral_to_XYZ_integration(
                XYZ_to_spectral_Meng2015(XYZ, interval=10), cmfs=cmfs_c),
            XYZ,
            decimal=7)

        np.testing.assert_almost_equal(
            spectral_to_XYZ_integration(
                XYZ_to_spectral_Meng2015(XYZ, interval=10, tolerance=1e-3),
                cmfs=cmfs_c),
            XYZ,
            decimal=7)

        shape = SpectralShape(400, 700, 5)
        cmfs_c = cmfs.clone().align(shape)
        np.testing.assert_almost_equal(
            spectral_to_XYZ_integration(
                XYZ_to_spectral_Meng2015(XYZ, cmfs=cmfs_c), cmfs=cmfs_c),
            XYZ,
            decimal=7)


if __name__ == '__main__':
    unittest.main()
