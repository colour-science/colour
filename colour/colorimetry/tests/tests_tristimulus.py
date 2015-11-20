#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.colorimetry.tristimulus` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry import (
    CMFS,
    ILLUMINANTS_RELATIVE_SPDS,
    SpectralPowerDistribution,
    spectral_to_XYZ,
    wavelength_to_XYZ)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RELATIVE_SPD_DATA',
           'TestSpectral_to_XYZ',
           'TestWavelength_to_XYZ']

RELATIVE_SPD_DATA = SpectralPowerDistribution(
    'Custom', {
        340: 0.0000,
        345: 0.0000,
        350: 0.0000,
        355: 0.0000,
        360: 0.0000,
        365: 0.0000,
        370: 0.0000,
        375: 0.0000,
        380: 0.0000,
        385: 0.0000,
        390: 0.0000,
        395: 0.0000,
        400: 0.0641,
        405: 0.0650,
        410: 0.0654,
        415: 0.0652,
        420: 0.0645,
        425: 0.0629,
        430: 0.0605,
        435: 0.0581,
        440: 0.0562,
        445: 0.0551,
        450: 0.0543,
        455: 0.0539,
        460: 0.0537,
        465: 0.0538,
        470: 0.0541,
        475: 0.0547,
        480: 0.0559,
        485: 0.0578,
        490: 0.0603,
        495: 0.0629,
        500: 0.0651,
        505: 0.0667,
        510: 0.0680,
        515: 0.0691,
        520: 0.0705,
        525: 0.0720,
        530: 0.0736,
        535: 0.0753,
        540: 0.0772,
        545: 0.0791,
        550: 0.0809,
        555: 0.0833,
        560: 0.0870,
        565: 0.0924,
        570: 0.0990,
        575: 0.1061,
        580: 0.1128,
        585: 0.1190,
        590: 0.1251,
        595: 0.1308,
        600: 0.1360,
        605: 0.1403,
        610: 0.1439,
        615: 0.1473,
        620: 0.1511,
        625: 0.1550,
        630: 0.1590,
        635: 0.1634,
        640: 0.1688,
        645: 0.1753,
        650: 0.1828,
        655: 0.1909,
        660: 0.1996,
        665: 0.2088,
        670: 0.2187,
        675: 0.2291,
        680: 0.2397,
        685: 0.2505,
        690: 0.2618,
        695: 0.2733,
        700: 0.2852,
        705: 0.0000,
        710: 0.0000,
        715: 0.0000,
        720: 0.0000,
        725: 0.0000,
        730: 0.0000,
        735: 0.0000,
        740: 0.0000,
        745: 0.0000,
        750: 0.0000,
        755: 0.0000,
        760: 0.0000,
        765: 0.0000,
        770: 0.0000,
        775: 0.0000,
        780: 0.0000,
        785: 0.0000,
        790: 0.0000,
        795: 0.0000,
        800: 0.0000,
        805: 0.0000,
        810: 0.0000,
        815: 0.0000,
        820: 0.0000,
        825: 0.0000,
        830: 0.0000})


class TestSpectral_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.spectral_to_XYZ` definition
    unit tests methods.
    """

    def test_spectral_to_XYZ(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.spectral_to_XYZ`
        definition.
        """

        cmfs = CMFS.get('CIE 1931 2 Degree Standard Observer')
        np.testing.assert_almost_equal(
            spectral_to_XYZ(
                RELATIVE_SPD_DATA.zeros(cmfs.shape),
                cmfs,
                ILLUMINANTS_RELATIVE_SPDS.get('A').clone().zeros(cmfs.shape)),
            np.array([14.46371626, 10.85832347, 2.04664796]),
            decimal=7)

        cmfs = CMFS.get('CIE 1964 10 Degree Standard Observer')
        np.testing.assert_almost_equal(
            spectral_to_XYZ(
                RELATIVE_SPD_DATA.zeros(cmfs.shape),
                cmfs,
                ILLUMINANTS_RELATIVE_SPDS.get('C').clone().zeros(cmfs.shape)),
            np.array([10.7704252, 9.44870313, 6.62742289]),
            decimal=7)

        np.testing.assert_almost_equal(
            spectral_to_XYZ(
                RELATIVE_SPD_DATA.zeros(cmfs.shape),
                cmfs,
                ILLUMINANTS_RELATIVE_SPDS.get('F2').clone().zeros(cmfs.shape)),
            np.array([11.57830745, 9.98744967, 3.95396539]),
            decimal=7)


class TestWavelength_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.wavelength_to_XYZ` definition
    unit tests methods.
    """

    def test_wavelength_to_XYZ(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.wavelength_to_XYZ`
        definition.
        """

        np.testing.assert_almost_equal(
            wavelength_to_XYZ(
                480,
                CMFS.get('CIE 1931 2 Degree Standard Observer')),
            np.array([0.09564, 0.13902, 0.81295]),
            decimal=7)

        np.testing.assert_almost_equal(
            wavelength_to_XYZ(
                480,
                CMFS.get('CIE 2012 2 Degree Standard Observer')),
            np.array([0.08182895, 0.1788048, 0.7552379]),
            decimal=7)

        np.testing.assert_almost_equal(
            wavelength_to_XYZ(
                641.5,
                CMFS.get('CIE 2012 2 Degree Standard Observer')),
            np.array([0.44575583, 0.18184213, 0.]),
            decimal=7)

        np.testing.assert_almost_equal(
            wavelength_to_XYZ(
                480.5,
                CMFS.get('CIE 2012 2 Degree Standard Observer'),
                'Cubic Spline'),
            np.array([0.07773422, 0.18148028, 0.7337162]),
            decimal=7)

        np.testing.assert_almost_equal(
            wavelength_to_XYZ(
                480.5,
                CMFS.get('CIE 2012 2 Degree Standard Observer'),
                'Linear'),
            np.array([0.07779856, 0.18149335, 0.7340129]),
            decimal=7)

        np.testing.assert_almost_equal(
            wavelength_to_XYZ(
                480.5,
                CMFS.get('CIE 2012 2 Degree Standard Observer'),
                'Pchip'),
            np.array([0.07773515, 0.18148048, 0.73372294]),
            decimal=7)

    def test_n_dimensional_wavelength_to_XYZ(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.wavelength_to_XYZ`
        definition n-dimensional arrays support.
        """

        cmfs = CMFS.get('CIE 1931 2 Degree Standard Observer')
        wl = 480
        XYZ = np.array([0.09564, 0.13902, 0.81295])
        np.testing.assert_almost_equal(
            wavelength_to_XYZ(wl, cmfs),
            XYZ,
            decimal=7)

        wl = np.tile(wl, 6)
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            wavelength_to_XYZ(wl, cmfs),
            XYZ,
            decimal=7)

        wl = np.reshape(wl, (2, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            wavelength_to_XYZ(wl, cmfs),
            XYZ,
            decimal=7)

        wl = np.reshape(wl, (2, 3, 1))
        XYZ = np.reshape(XYZ, (2, 3, 1, 3))
        np.testing.assert_almost_equal(
            wavelength_to_XYZ(wl, cmfs),
            XYZ,
            decimal=7)


if __name__ == '__main__':
    unittest.main()
