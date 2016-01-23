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
    CIE_standard_illuminant_A,
    ILLUMINANTS_RELATIVE_SPDS,
    SpectralPowerDistribution,
    SpectralShape)
from colour.colorimetry import (
    lagrange_coefficients_ASTME202211,
    tristimulus_weighting_factors_ASTME202211,
    adjust_tristimulus_weighting_factors_ASTME30815,
    spectral_to_XYZ,
    wavelength_to_XYZ)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RELATIVE_SPD_DATA',
           'LAGRANGE_COEFFICIENTS_A',
           'LAGRANGE_COEFFICIENTS_B',
           'A_CIE_1964_10_10_TWF',
           'A_CIE_1964_10_20_TWF',
           'D65_CIE_1931_2_20_TWF',
           'D65_CIE_1931_2_20_ATWF',
           'TestLagrangeCoefficientsASTME202211',
           'TestTristimulusWeightingFactorsASTME202211',
           'TestAdjustTristimulusWeightingFactorsASTME30815',
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

LAGRANGE_COEFFICIENTS_A = np.array(
    [[-0.0285, 0.9405, 0.1045, -0.0165],
     [-0.0480, 0.8640, 0.2160, -0.0320],
     [-0.0595, 0.7735, 0.3315, -0.0455],
     [-0.0640, 0.6720, 0.4480, -0.0560],
     [-0.0625, 0.5625, 0.5625, -0.0625],
     [-0.0560, 0.4480, 0.6720, -0.0640],
     [-0.0455, 0.3315, 0.7735, -0.0595],
     [-0.0320, 0.2160, 0.8640, -0.0480],
     [-0.0165, 0.1045, 0.9405, -0.0285]])

LAGRANGE_COEFFICIENTS_B = np.array(
    [[0.8550, 0.1900, -0.0450],
     [0.7200, 0.3600, -0.0800],
     [0.5950, 0.5100, -0.1050],
     [0.4800, 0.6400, -0.1200],
     [0.3750, 0.7500, -0.1250],
     [0.2800, 0.8400, -0.1200],
     [0.1950, 0.9100, -0.1050],
     [0.1200, 0.9600, -0.0800],
     [0.0550, 0.9900, -0.0450]])

A_CIE_1964_10_10_TWF = np.array(
    [[-0.000, -0.000, -0.000],
     [-0.000, -0.000, -0.000],
     [-0.000, -0.000, -0.000],
     [0.002, 0.000, 0.008],
     [0.025, 0.003, 0.110],
     [0.134, 0.014, 0.615],
     [0.377, 0.039, 1.792],
     [0.686, 0.084, 3.386],
     [0.964, 0.156, 4.944],
     [1.080, 0.259, 5.806],
     [1.006, 0.424, 5.812],
     [0.731, 0.696, 4.919],
     [0.343, 1.082, 3.300],
     [0.078, 1.616, 1.973],
     [0.022, 2.422, 1.152],
     [0.218, 3.529, 0.658],
     [0.750, 4.840, 0.382],
     [1.642, 6.100, 0.211],
     [2.842, 7.250, 0.102],
     [4.336, 8.114, 0.032],
     [6.200, 8.758, 0.001],
     [8.262, 8.988, -0.000],
     [10.227, 8.760, 0.000],
     [11.945, 8.304, 0.000],
     [12.746, 7.468, 0.000],
     [12.337, 6.323, 0.000],
     [10.817, 5.033, 0.000],
     [8.560, 3.744, 0.000],
     [6.014, 2.506, 0.000],
     [3.887, 1.560, 0.000],
     [2.309, 0.911, 0.000],
     [1.276, 0.499, 0.000],
     [0.666, 0.259, 0.000],
     [0.336, 0.130, 0.000],
     [0.166, 0.065, 0.000],
     [0.082, 0.032, 0.000],
     [0.040, 0.016, 0.000],
     [0.020, 0.008, 0.000],
     [0.010, 0.004, 0.000],
     [0.005, 0.002, 0.000],
     [0.003, 0.001, 0.000],
     [0.001, 0.001, 0.000],
     [0.001, 0.000, 0.000],
     [0.000, 0.000, 0.000],
     [0.000, 0.000, 0.000],
     [0.000, 0.000, 0.000],
     [0.000, 0.000, 0.000],
     [0.000, 0.000, 0.000]])

A_CIE_1964_10_20_TWF = np.array(
    [[-0.000, -0.000, -0.001],
     [-0.009, -0.001, -0.041],
     [0.060, 0.005, 0.257],
     [0.773, 0.078, 3.697],
     [1.900, 0.304, 9.755],
     [1.971, 0.855, 11.487],
     [0.718, 2.146, 6.785],
     [0.043, 4.899, 2.321],
     [1.522, 9.647, 0.743],
     [5.677, 14.461, 0.196],
     [12.445, 17.474, 0.005],
     [20.554, 17.584, -0.003],
     [25.332, 14.896, 0.000],
     [21.571, 10.080, 0.000],
     [12.179, 5.068, 0.000],
     [4.668, 1.830, 0.000],
     [1.324, 0.513, 0.000],
     [0.318, 0.123, 0.000],
     [0.075, 0.029, 0.000],
     [0.018, 0.007, 0.000],
     [0.005, 0.002, 0.000],
     [0.001, 0.001, 0.000],
     [0.000, 0.000, 0.000],
     [0.000, 0.000, 0.000]])

D65_CIE_1931_2_20_TWF = np.array(
    [[-0.001, -0.000, -0.005],
     [-0.008, -0.000, -0.039],
     [0.179, 0.002, 0.829],
     [2.542, 0.071, 12.203],
     [6.670, 0.453, 33.637],
     [6.333, 1.316, 36.334],
     [2.213, 2.933, 18.278],
     [0.052, 6.866, 5.543],
     [1.348, 14.106, 1.611],
     [5.767, 18.981, 0.382],
     [11.301, 18.863, 0.068],
     [16.256, 15.455, 0.025],
     [17.933, 10.699, 0.013],
     [14.020, 6.277, 0.003],
     [7.057, 2.743, 0.000],
     [2.527, 0.927, -0.000],
     [0.670, 0.242, -0.000],
     [0.140, 0.050, 0.000],
     [0.035, 0.013, 0.000],
     [0.008, 0.003, 0.000],
     [0.002, 0.001, 0.000],
     [0.000, 0.000, 0.000],
     [0.000, 0.000, 0.000],
     [0.000, 0.000, 0.000]])

D65_CIE_1931_2_20_ATWF = np.array(
    [[0.170, 0.002, 0.785],
     [2.542, 0.071, 12.203],
     [6.670, 0.453, 33.637],
     [6.333, 1.316, 36.334],
     [2.213, 2.933, 18.278],
     [0.052, 6.866, 5.543],
     [1.348, 14.106, 1.611],
     [5.767, 18.981, 0.382],
     [11.301, 18.863, 0.068],
     [16.256, 15.455, 0.025],
     [17.933, 10.699, 0.013],
     [14.020, 6.277, 0.003],
     [7.057, 2.743, 0.000],
     [2.527, 0.927, -0.000],
     [0.670, 0.242, -0.000],
     [0.185, 0.067, 0.000]])


class TestLagrangeCoefficientsASTME202211(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.\
lagrange_coefficients_ASTME202211` definition unit tests methods.
    """

    def test_lagrange_coefficients_ASTME202211(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
    lagrange_coefficients_ASTME202211` definition.
        """

        np.testing.assert_almost_equal(
            lagrange_coefficients_ASTME202211(10, 'inner'),
            LAGRANGE_COEFFICIENTS_A,
            decimal=7)

        np.testing.assert_almost_equal(
            lagrange_coefficients_ASTME202211(10, 'boundary'),
            LAGRANGE_COEFFICIENTS_B,
            decimal=7)


class TestTristimulusWeightingFactorsASTME202211(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.\
tristimulus_weighting_factors_ASTME202211` definition unit tests methods.
    """

    def test_tristimulus_weighting_factors_ASTME202211(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
tristimulus_weighting_factors_ASTME202211` definition.

        Notes
        -----
        :attr:`A_CIE_1964_10_10_TWF`, :attr:`A_CIE_1964_10_20_TWF` and
        :attr:`D65_CIE_1931_2_20_TWF` attributes data is matching [1]_.

        References
        ----------
        .. [1]  ASTM International. (2015). ASTM E308–15 - Standard Practice
                for Computing the Colors of Objects by Using the CIE System,
                1–47. doi:10.1520/E0308-15
        """

        cmfs = CMFS.get('CIE 1964 10 Degree Standard Observer')
        wl = cmfs.shape.range()
        A = SpectralPowerDistribution(
            'A', dict(zip(wl, CIE_standard_illuminant_A(wl))))

        twf = tristimulus_weighting_factors_ASTME202211(
            cmfs, A, SpectralShape(360, 830, 10))
        np.testing.assert_almost_equal(
            np.round(twf, 3),
            A_CIE_1964_10_10_TWF,
            decimal=3)

        twf = tristimulus_weighting_factors_ASTME202211(
            cmfs, A, SpectralShape(360, 830, 20))
        np.testing.assert_almost_equal(
            np.round(twf, 3),
            A_CIE_1964_10_20_TWF,
            decimal=3)

        cmfs = CMFS.get('CIE 1931 2 Degree Standard Observer')
        D65 = ILLUMINANTS_RELATIVE_SPDS['D65'].clone().align(
            cmfs.shape, interpolation_method='Linear')
        twf = tristimulus_weighting_factors_ASTME202211(
            cmfs, D65, SpectralShape(360, 830, 20))
        np.testing.assert_almost_equal(
            np.round(twf, 3),
            D65_CIE_1931_2_20_TWF,
            decimal=3)


class TestAdjustTristimulusWeightingFactorsASTME30815(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.\
adjust_tristimulus_weighting_factors_ASTME30815` definition unit tests methods.
    """

    def test_adjust_tristimulus_weighting_factors_ASTME30815(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
adjust_tristimulus_weighting_factors_ASTME30815` definition.
        """

        np.testing.assert_almost_equal(
            adjust_tristimulus_weighting_factors_ASTME30815(
                D65_CIE_1931_2_20_TWF, SpectralShape(360, 830, 20),
                SpectralShape(400, 700, 20)),
            D65_CIE_1931_2_20_ATWF,
            decimal=3)


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
