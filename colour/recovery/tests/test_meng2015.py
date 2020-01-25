# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.recovery.meng2015` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry import (DEFAULT_SPECTRAL_SHAPE,
                                STANDARD_OBSERVERS_CMFS, SpectralShape,
                                ILLUMINANTS_SDS, sd_to_XYZ_integration)
from colour.recovery import XYZ_to_sd_Meng2015
from colour.utilities import domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestXYZ_to_sd_Meng2015']


class TestXYZ_to_sd_Meng2015(unittest.TestCase):
    """
    Defines :func:`colour.recovery.meng2015.XYZ_to_sd_Meng2015`
    definition unit tests methods.
    """

    def test_XYZ_to_sd_Meng2015(self):
        """
        Tests :func:`colour.recovery.meng2015.XYZ_to_sd_Meng2015`
        definition.
        """

        cmfs = (STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
                .copy().trim(DEFAULT_SPECTRAL_SHAPE))
        shape = SpectralShape(cmfs.shape.start, cmfs.shape.end, 5)
        cmfs_c = cmfs.copy().align(shape)

        XYZ = np.array([0.21781186, 0.12541048, 0.04697113])
        np.testing.assert_almost_equal(
            sd_to_XYZ_integration(XYZ_to_sd_Meng2015(XYZ, cmfs_c), cmfs_c) /
            100,
            XYZ,
            decimal=7)

        shape = SpectralShape(cmfs.shape.start, cmfs.shape.end, 10)
        cmfs_c = cmfs.copy().align(shape)

        np.testing.assert_almost_equal(
            sd_to_XYZ_integration(XYZ_to_sd_Meng2015(XYZ, cmfs_c), cmfs_c) /
            100,
            XYZ,
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_integration(
                XYZ_to_sd_Meng2015(XYZ, cmfs_c, ILLUMINANTS_SDS['D65']),
                cmfs_c, ILLUMINANTS_SDS['D65']) / 100,
            XYZ,
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_integration(
                XYZ_to_sd_Meng2015(
                    XYZ,
                    cmfs_c,
                    optimisation_parameters={'options': {
                        'ftol': 1e-10,
                    }}), cmfs_c) / 100,
            XYZ,
            decimal=7)

        shape = SpectralShape(400, 700, 5)
        cmfs_c = cmfs.copy().align(shape)
        np.testing.assert_almost_equal(
            sd_to_XYZ_integration(XYZ_to_sd_Meng2015(XYZ, cmfs_c), cmfs_c) /
            100,
            XYZ,
            decimal=7)

    def test_raise_exception_XYZ_to_sd_Meng2015(self):
        """
        Tests :func:`colour.recovery.meng2015.XYZ_to_sd_Meng2015`
        definition raised exception.
        """

        self.assertRaises(
            RuntimeError,
            XYZ_to_sd_Meng2015,
            np.array([0.0, 0.0, 1.0]),
            optimisation_parameters={
                'options': {
                    'maxiter': 10
                },
            })

    def test_domain_range_scale_XYZ_to_sd_Meng2015(self):
        """
        Tests :func:`colour.recovery.meng2015.XYZ_to_sd_Meng2015`
        definition domain and range scale support.
        """

        XYZ_i = np.array([0.21781186, 0.12541048, 0.04697113])
        XYZ_o = sd_to_XYZ_integration(XYZ_to_sd_Meng2015(XYZ_i))

        d_r = (('reference', 1, 1), (1, 1, 0.01), (100, 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    sd_to_XYZ_integration(
                        XYZ_to_sd_Meng2015(XYZ_i * factor_a)),
                    XYZ_o * factor_b,
                    decimal=7)


if __name__ == '__main__':
    unittest.main()
