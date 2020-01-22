# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.colorimetry.tristimulus` module.

References
----------
-   :cite:`ASTMInternational2015b` : ASTM International. (2015). ASTM E308-15 -
    Standard Practice for Computing the Colors of Objects by Using the CIE
    System. doi:10.1520/E0308-15
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.algebra import LinearInterpolator
from colour.colorimetry import (CMFS, sd_CIE_standard_illuminant_A,
                                ILLUMINANTS_SDS, MultiSpectralDistributions,
                                SpectralDistribution, SpectralShape)
from colour.colorimetry import (
    lagrange_coefficients_ASTME2022, tristimulus_weighting_factors_ASTME2022,
    adjust_tristimulus_weighting_factors_ASTME308, sd_to_XYZ_integration,
    sd_to_XYZ_tristimulus_weighting_factors_ASTME308, sd_to_XYZ_ASTME308,
    multi_sds_to_XYZ_integration, multi_sds_to_XYZ_ASTME308, wavelength_to_XYZ)
from colour.utilities import domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'SAMPLE_SD', 'LAGRANGE_COEFFICIENTS_A', 'LAGRANGE_COEFFICIENTS_B',
    'A_CIE_1964_10_10_TWF', 'A_CIE_1964_10_20_TWF', 'D65_CIE_1931_2_20_TWF',
    'D65_CIE_1931_2_20_TWF_K1', 'D65_CIE_1931_2_20_ATWF', 'MSDS_ARRAY', 'MSDS',
    'XYZ_D65_INTEGRATION_MSDS', 'XYZ_D65_ARRAY_INTEGRATION',
    'XYZ_D65_ARRAY_K1_INTEGRATION', 'XYZ_D65_ASTME308_MSDS',
    'XYZ_D65_ASTME308_K1_MSDS', 'TestLagrangeCoefficientsASTME2022',
    'TestTristimulusWeightingFactorsASTME2022',
    'TestAdjustTristimulusWeightingFactorsASTME308',
    'TestSd_to_XYZ_integration', 'TestSd_to_XYZ_ASTME308',
    'TestMultiSds_to_XYZ_integration', 'TestMultiSds_to_XYZ_ASTME308',
    'TestWavelength_to_XYZ'
]

SAMPLE_SD = SpectralDistribution({
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
    830: 0.0000
})

LAGRANGE_COEFFICIENTS_A = np.array([
    [-0.0285, 0.9405, 0.1045, -0.0165],
    [-0.0480, 0.8640, 0.2160, -0.0320],
    [-0.0595, 0.7735, 0.3315, -0.0455],
    [-0.0640, 0.6720, 0.4480, -0.0560],
    [-0.0625, 0.5625, 0.5625, -0.0625],
    [-0.0560, 0.4480, 0.6720, -0.0640],
    [-0.0455, 0.3315, 0.7735, -0.0595],
    [-0.0320, 0.2160, 0.8640, -0.0480],
    [-0.0165, 0.1045, 0.9405, -0.0285],
])

LAGRANGE_COEFFICIENTS_B = np.array([
    [0.8550, 0.1900, -0.0450],
    [0.7200, 0.3600, -0.0800],
    [0.5950, 0.5100, -0.1050],
    [0.4800, 0.6400, -0.1200],
    [0.3750, 0.7500, -0.1250],
    [0.2800, 0.8400, -0.1200],
    [0.1950, 0.9100, -0.1050],
    [0.1200, 0.9600, -0.0800],
    [0.0550, 0.9900, -0.0450],
])

A_CIE_1964_10_10_TWF = np.array([
    [-0.000, -0.000, -0.000],
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
    [0.000, 0.000, 0.000],
])

A_CIE_1964_10_20_TWF = np.array([
    [-0.000, -0.000, -0.001],
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
    [0.000, 0.000, 0.000],
])

D65_CIE_1931_2_20_TWF = np.array([
    [-0.001, -0.000, -0.005],
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
    [0.000, 0.000, 0.000],
])

D65_CIE_1931_2_20_TWF_K1 = np.array([
    [-0.10095678, -0.00265636, -0.48295051],
    [-0.83484763, -0.02190274, -4.11563004],
    [18.94315946, 0.22803520, 87.62930101],
    [268.66426663, 7.45156533, 1289.46785306],
    [704.84093814, 47.85119956, 3554.47729494],
    [669.16372619, 139.09334752, 3839.47028848],
    [233.89418975, 309.95075927, 1931.47489098],
    [5.51347204, 725.55154566, 585.77542998],
    [142.48116090, 1490.55009270, 170.27819443],
    [609.43424752, 2005.73581058, 40.34233506],
    [1194.21293134, 1993.32004423, 7.15981395],
    [1717.79835378, 1633.12477710, 2.59651081],
    [1895.00791740, 1130.54333854, 1.34461357],
    [1481.55235852, 663.25632432, 0.29999368],
    [745.76471129, 289.85683288, 0.01943154],
    [267.01875994, 97.97358872, -0.00261658],
    [70.75239887, 25.56445574, -0.00019929],
    [14.78862574, 5.31713332, 0.00000000],
    [3.67620064, 1.32650433, 0.00000000],
    [0.89699648, 0.32392186, 0.00000000],
    [0.16623785, 0.06003153, 0.00000000],
    [0.04824448, 0.01742197, 0.00000000],
    [0.01310759, 0.00473339, 0.00000000],
    [0.00223616, 0.00080752, 0.00000000],
])

D65_CIE_1931_2_20_ATWF = np.array([
    [0.170, 0.002, 0.785],
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
    [0.185, 0.067, 0.000],
])

MSDS_ARRAY = np.array([
    [[0.01367208, 0.09127947, 0.01524376, 0.02810712, 0.19176012, 0.04299992],
     [0.01591516, 0.31454948, 0.08416876, 0.09071489, 0.71026170, 0.04374762],
     [0.00959792, 0.25822842, 0.41388571, 0.22275120, 0.00407416, 0.37439537],
     [0.01106279, 0.07090867, 0.02204929, 0.12487984, 0.18168917, 0.00202945],
     [0.01791409, 0.29707789, 0.56295109, 0.23752193, 0.00236515, 0.58190280],
     [0.10565346, 0.46204320, 0.19180590, 0.56250858, 0.42085907, 0.00270085]],
    [[0.04325933, 0.26825359, 0.23732357, 0.05175860, 0.01181048, 0.08233768],
     [0.02577249, 0.08305486, 0.04303044, 0.32298771, 0.23022813, 0.00813306],
     [0.02484169, 0.12027161, 0.00541695, 0.00654612, 0.18603799, 0.36247808],
     [0.01861601, 0.12924391, 0.00785840, 0.40062562, 0.94044405, 0.32133976],
     [0.03102159, 0.16815442, 0.37186235, 0.08610666, 0.00413520, 0.78492409],
     [0.04727245, 0.32210270, 0.22679484, 0.31613642, 0.11242847, 0.00244144]],
])

MSDS = MultiSpectralDistributions(
    np.transpose(np.reshape(MSDS_ARRAY, [-1, 6])),
    SpectralShape(400, 700, 60).range())

XYZ_D65_INTEGRATION_MSDS = np.array([
    [7.50233492, 3.95038753, 8.40338691],
    [26.92687731, 15.07139447, 28.71690972],
    [16.70142165, 28.21458593, 25.65389977],
    [11.57556582, 8.64089371, 6.57886800],
    [18.73219899, 35.07426727, 30.15041858],
    [45.16536627, 39.61360393, 43.69092793],
    [8.17423105, 13.09267724, 25.94320661],
    [22.46636804, 19.31025987, 7.96118575],
    [6.58131705, 2.52857864, 11.09362054],
    [43.91622393, 27.97952698, 11.73575112],
    [8.53774314, 19.70241814, 17.70479540],
    [23.91149458, 26.21467120, 30.68294836],
])

XYZ_D65_ARRAY_INTEGRATION = np.array([
    [
        [7.19510558, 3.86227393, 10.09950719],
        [25.57464912, 14.71934603, 34.84931928],
        [17.58300551, 28.56388139, 30.18370150],
        [11.32631694, 8.46087304, 7.90263107],
        [19.65793587, 35.59047030, 35.14042633],
        [45.82162927, 39.26057155, 51.79537877],
    ],
    [
        [8.82617380, 13.38600040, 30.56510531],
        [22.33167167, 18.95683859, 9.39034481],
        [6.69130415, 2.57592352, 13.25898396],
        [41.81950400, 27.11920225, 14.26746010],
        [9.24148668, 20.20448258, 20.19416075],
        [24.78545992, 26.22388193, 36.44325237],
    ],
])

XYZ_D65_ARRAY_K1_INTEGRATION = np.array([
    [
        [776.11755347, 416.61356647, 1089.40789347],
        [2758.67169575, 1587.73804035, 3759.10653835],
        [1896.63363103, 3081.11250154, 3255.83833518],
        [1221.74071019, 912.65263825, 852.43651028],
        [2120.45103874, 3839.05259546, 3790.50750799],
        [4942.66142767, 4234.93698798, 5587.03443988],
    ],
    [
        [952.05669281, 1443.91347280, 3296.97938475],
        [2408.86005021, 2044.82547598, 1012.91236929],
        [721.77378835, 277.85825207, 1430.21253594],
        [4510.96245706, 2925.27867434, 1538.99426670],
        [996.85542597, 2179.40562871, 2178.29223889],
        [2673.54388456, 2828.70277157, 3931.04000571],
    ],
])

XYZ_D65_ASTME308_MSDS = np.array([
    [7.50462879, 3.95733605, 8.38922164],
    [26.94169070, 15.09763521, 28.67425696],
    [16.70295581, 28.20634330, 25.66397540],
    [11.57004370, 8.64530316, 6.56080272],
    [18.74361790, 35.06186242, 30.18256374],
    [45.12217114, 39.62387120, 43.59379506],
    [8.17736556, 13.09427036, 25.93377871],
    [22.44504068, 19.31189958, 7.92783105],
    [6.57962398, 2.53363268, 11.07307595],
    [43.91277836, 27.99946326, 11.69170298],
    [8.55079137, 19.69075442, 17.73974503],
    [23.88934466, 26.21652383, 30.63634245],
])

XYZ_D65_ASTME308_K1_MSDS = np.array([
    [793.03459123, 418.18249296, 886.51193069],
    [2847.00193341, 1595.40828847, 3030.08693538],
    [1765.04689430, 2980.64122346, 2711.98227327],
    [1222.63807218, 913.57276319, 693.29791668],
    [1980.68922284, 3705.08262551, 3189.47382721],
    [4768.18288568, 4187.16253479, 4606.67521602],
    [864.12452076, 1383.70726024, 2740.49312500],
    [2371.82866258, 2040.74109793, 837.75552835],
    [695.28681092, 267.73587534, 1170.12213512],
    [4640.38305142, 2958.77964509, 1235.49414014],
    [903.58544429, 2080.77572128, 1874.60723907],
    [2524.45220328, 2770.37157298, 3237.42586015],
])


class TestLagrangeCoefficientsASTME2022(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.\
lagrange_coefficients_ASTME2022` definition unit tests methods.
    """

    def test_lagrange_coefficients_ASTME2022(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
lagrange_coefficients_ASTME2022` definition.
        """

        np.testing.assert_almost_equal(
            lagrange_coefficients_ASTME2022(10, 'inner'),
            LAGRANGE_COEFFICIENTS_A,
            decimal=7)

        np.testing.assert_almost_equal(
            lagrange_coefficients_ASTME2022(10, 'boundary'),
            LAGRANGE_COEFFICIENTS_B,
            decimal=7)


class TestTristimulusWeightingFactorsASTME2022(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.\
tristimulus_weighting_factors_ASTME2022` definition unit tests methods.
    """

    def test_tristimulus_weighting_factors_ASTME2022(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
tristimulus_weighting_factors_ASTME2022` definition.

        Notes
        -----
        -   :attr:`A_CIE_1964_10_10_TWF`, :attr:`A_CIE_1964_10_20_TWF` and
            :attr:`D65_CIE_1931_2_20_TWF` attributes data is matching
            :cite:`ASTMInternational2015b`.

        References
        ----------
        :cite:`ASTMInternational2015b`
        """

        cmfs = CMFS['CIE 1964 10 Degree Standard Observer']
        A = sd_CIE_standard_illuminant_A(cmfs.shape)

        twf = tristimulus_weighting_factors_ASTME2022(
            cmfs, A, SpectralShape(360, 830, 10))
        np.testing.assert_almost_equal(
            np.round(twf, 3), A_CIE_1964_10_10_TWF, decimal=3)

        twf = tristimulus_weighting_factors_ASTME2022(
            cmfs, A, SpectralShape(360, 830, 20))
        np.testing.assert_almost_equal(
            np.round(twf, 3), A_CIE_1964_10_20_TWF, decimal=3)

        cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
        D65 = ILLUMINANTS_SDS['D65'].copy().align(
            cmfs.shape, interpolator=LinearInterpolator)
        twf = tristimulus_weighting_factors_ASTME2022(
            cmfs, D65, SpectralShape(360, 830, 20))
        np.testing.assert_almost_equal(
            np.round(twf, 3), D65_CIE_1931_2_20_TWF, decimal=3)

        twf = tristimulus_weighting_factors_ASTME2022(
            cmfs, D65, SpectralShape(360, 830, 20), k=1)
        np.testing.assert_almost_equal(
            twf, D65_CIE_1931_2_20_TWF_K1, decimal=7)

    def test_raise_exception_tristimulus_weighting_factors_ASTME2022(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
tristimulus_weighting_factors_ASTME2022` definition raised exception.
        """

        shape = SpectralShape(360, 830, 10)
        cmfs_1 = CMFS['CIE 1964 10 Degree Standard Observer']
        cmfs_2 = cmfs_1.copy().align(shape)
        A_1 = sd_CIE_standard_illuminant_A(cmfs_1.shape)
        A_2 = sd_CIE_standard_illuminant_A(cmfs_2.shape)

        self.assertRaises(ValueError, tristimulus_weighting_factors_ASTME2022,
                          cmfs_1, A_2, shape)

        self.assertRaises(ValueError, tristimulus_weighting_factors_ASTME2022,
                          cmfs_2, A_1, shape)


class TestAdjustTristimulusWeightingFactorsASTME308(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.\
adjust_tristimulus_weighting_factors_ASTME308` definition unit tests methods.
    """

    def test_adjust_tristimulus_weighting_factors_ASTME308(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
adjust_tristimulus_weighting_factors_ASTME308` definition.
        """

        np.testing.assert_almost_equal(
            adjust_tristimulus_weighting_factors_ASTME308(
                D65_CIE_1931_2_20_TWF, SpectralShape(360, 830, 20),
                SpectralShape(400, 700, 20)),
            D65_CIE_1931_2_20_ATWF,
            decimal=3)


class TestSd_to_XYZ_integration(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.sd_to_XYZ_integration`
    definition unit tests methods.
    """

    def test_sd_to_XYZ_integration(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
sd_to_XYZ_integration`
        definition.
        """

        cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
        np.testing.assert_almost_equal(
            sd_to_XYZ_integration(SAMPLE_SD, cmfs, ILLUMINANTS_SDS['A']),
            np.array([14.46365624, 10.85827910, 2.04662343]),
            decimal=7)

        cmfs = CMFS['CIE 1964 10 Degree Standard Observer']
        np.testing.assert_almost_equal(
            sd_to_XYZ_integration(SAMPLE_SD, cmfs, ILLUMINANTS_SDS['C']),
            np.array([10.77031004, 9.44863775, 6.62745989]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_integration(SAMPLE_SD, cmfs, ILLUMINANTS_SDS['FL2']),
            np.array([11.57834054, 9.98738373, 3.95462625]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_integration(
                SAMPLE_SD, cmfs, ILLUMINANTS_SDS['FL2'], k=683),
            np.array([122441.23450378, 105616.82732832, 41820.26940409]),
            decimal=7)

    def test_domain_range_scale_sd_to_XYZ_integration(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
sd_to_XYZ_integration` definition domain and range scale support.
        """

        cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
        XYZ = sd_to_XYZ_integration(SAMPLE_SD, cmfs, ILLUMINANTS_SDS['A'])

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    sd_to_XYZ_integration(SAMPLE_SD, cmfs,
                                          ILLUMINANTS_SDS['A']),
                    XYZ * factor,
                    decimal=7)


class TestSd_to_XYZ_tristimulus_weighting_factors_ASTME308(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.\
sd_to_XYZ_tristimulus_weighting_factors_ASTME308`
    definition unit tests methods.
    """

    def test_sd_to_XYZ_tristimulus_weighting_factors_ASTME308(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
sd_to_XYZ_tristimulus_weighting_factors_ASTME308`
        definition.
        """

        cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
        np.testing.assert_almost_equal(
            sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
                SAMPLE_SD, cmfs, ILLUMINANTS_SDS['A']),
            np.array([14.46366344, 10.85828513, 2.04663792]),
            decimal=7)

        cmfs = CMFS['CIE 1964 10 Degree Standard Observer']
        np.testing.assert_almost_equal(
            sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
                SAMPLE_SD, cmfs, ILLUMINANTS_SDS['C']),
            np.array([10.77033881, 9.44864632, 6.62758924]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
                SAMPLE_SD, cmfs, ILLUMINANTS_SDS['FL2']),
            np.array([11.57837130, 9.98734511, 3.95499522]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
                SAMPLE_SD.copy().trim(SpectralShape(400, 700, 5)), cmfs,
                ILLUMINANTS_SDS['A']),
            np.array([14.38180830, 10.74512906, 2.01579131]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
                SAMPLE_SD.copy().interpolate(SpectralShape(400, 700, 10)),
                cmfs, ILLUMINANTS_SDS['A']),
            np.array([14.38284399, 10.74577954, 2.01553721]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
                SAMPLE_SD.copy().interpolate(SpectralShape(400, 700, 20)),
                cmfs, ILLUMINANTS_SDS['A']),
            np.array([14.38356848, 10.74613294, 2.01526418]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
                SAMPLE_SD.copy().interpolate(SpectralShape(400, 700, 20)),
                cmfs,
                ILLUMINANTS_SDS['A'],
                k=1),
            np.array([1636.74286798, 1222.82981953, 229.32204062]),
            decimal=7)

    def test_domain_range_scale_sd_to_XYZ_twf_ASTME308(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
sd_to_XYZ_tristimulus_weighting_factors_ASTME308` definition domain and
range scale support.
        """

        cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
        XYZ = sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
            SAMPLE_SD, cmfs, ILLUMINANTS_SDS['A'])

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
                        SAMPLE_SD, cmfs, ILLUMINANTS_SDS['A']),
                    XYZ * factor,
                    decimal=7)


class TestSd_to_XYZ_ASTME308(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.sd_to_XYZ_ASTME308`
    definition unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._sd = SAMPLE_SD.copy()
        self._cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
        self._A = sd_CIE_standard_illuminant_A(self._cmfs.shape)

    def test_sd_to_XYZ_ASTME308_mi_1nm(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.sd_to_XYZ_ASTME308`
        definition for 1 nm measurement intervals.
        """

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(self._sd.copy().align(self._cmfs.shape),
                               self._cmfs, self._A),
            np.array([14.46372680, 10.85832950, 2.04663200]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(self._cmfs.shape),
                self._cmfs,
                self._A,
                use_practice_range=False),
            np.array([14.46366018, 10.85827949, 2.04662258]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 1)), self._cmfs,
                self._A),
            np.array([14.54173397, 10.88628632, 2.04965822]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 1)),
                self._cmfs,
                self._A,
                use_practice_range=False),
            np.array([14.54203076, 10.88636754, 2.04964877]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 1)),
                self._cmfs,
                self._A,
                k=1),
            np.array([1568.98152997, 1174.57671769, 221.14803420]),
            decimal=7)

    def test_sd_to_XYZ_ASTME308_mi_5nm(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.sd_to_XYZ_ASTME308`
        definition for 5 nm measurement intervals.
        """

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(360, 830, 5)), self._cmfs,
                self._A),
            np.array([14.46372173, 10.85832502, 2.04664734]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(360, 830, 5)),
                self._cmfs,
                self._A,
                use_practice_range=False),
            np.array([14.46366388, 10.85828159, 2.04663915]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(360, 830, 5)),
                self._cmfs,
                self._A,
                mi_5nm_omission_method=False),
            np.array([14.46373399, 10.85833553, 2.0466465]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 5)), self._cmfs,
                self._A),
            np.array([14.54025742, 10.88576251, 2.04950226]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 5)),
                self._cmfs,
                self._A,
                use_practice_range=False),
            np.array([14.54051517, 10.88583304, 2.04949406]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 5)),
                self._cmfs,
                self._A,
                mi_5nm_omission_method=False),
            np.array([14.54022093, 10.88575468, 2.04951057]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(360, 830, 5)),
                self._cmfs,
                self._A,
                use_practice_range=False,
                mi_5nm_omission_method=False),
            np.array([14.46366737, 10.85828552, 2.04663707]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 5)),
                self._cmfs,
                self._A,
                use_practice_range=False,
                mi_5nm_omission_method=False),
            np.array([14.54051772, 10.88583590, 2.04950113]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 5)),
                self._cmfs,
                self._A,
                k=1),
            np.array([1568.82479013, 1174.52212708, 221.13156963]),
            decimal=7)

    def test_sd_to_XYZ_ASTME308_mi_10nm(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.sd_to_XYZ_ASTME308`
        definition for 10 nm measurement intervals.
        """

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(360, 830, 10)), self._cmfs,
                self._A),
            np.array([14.47779980, 10.86358645, 2.04751388]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(360, 830, 10)),
                self._cmfs,
                self._A,
                use_practice_range=False),
            np.array([14.47773312, 10.86353641, 2.04750445]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 10)), self._cmfs,
                self._A),
            np.array([14.54137532, 10.88641727, 2.04931318]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 10)),
                self._cmfs,
                self._A,
                use_practice_range=False),
            np.array([14.54167211, 10.88649849, 2.04930374]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 10)),
                self._cmfs,
                self._A,
                k=1),
            np.array([1568.94283333, 1174.59084705, 221.11080639]),
            decimal=7)

    def test_sd_to_XYZ_ASTME308_mi_20nm(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.sd_to_XYZ_ASTME308`
        definition for 20 nm measurement intervals.
        """

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(360, 820, 20)), self._cmfs,
                self._A),
            np.array([14.50187464, 10.87217124, 2.04918305]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(360, 820, 20)),
                self._cmfs,
                self._A,
                use_practice_range=False),
            np.array([14.50180785, 10.87212116, 2.04917361]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(360, 820, 20)),
                self._cmfs,
                self._A,
                mi_20nm_interpolation_method=False),
            np.array([14.50216194, 10.87236873, 2.04977256]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 20)), self._cmfs,
                self._A),
            np.array([14.54114025, 10.88634755, 2.04916445]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 20)),
                self._cmfs,
                self._A,
                use_practice_range=False),
            np.array([14.54143704, 10.88642877, 2.04915501]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 20)),
                self._cmfs,
                self._A,
                mi_20nm_interpolation_method=False),
            np.array([14.54242562, 10.88694088, 2.04919645]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(360, 820, 20)),
                self._cmfs,
                self._A,
                use_practice_range=False,
                mi_20nm_interpolation_method=False),
            np.array([14.50209515, 10.87231865, 2.04976312]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 20)),
                self._cmfs,
                self._A,
                use_practice_range=False,
                mi_20nm_interpolation_method=False),
            np.array([14.54272240, 10.88702210, 2.04918701]),
            decimal=7)

        np.testing.assert_almost_equal(
            sd_to_XYZ_ASTME308(
                self._sd.copy().align(SpectralShape(400, 700, 20)),
                self._cmfs,
                self._A,
                k=1),
            np.array([1568.91747040, 1174.58332427, 221.09475945]),
            decimal=7)

    def test_raise_exception_sd_to_XYZ_ASTME308(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.sd_to_XYZ_ASTME308`
        definition raised exception.
        """

        self.assertRaises(ValueError, sd_to_XYZ_ASTME308,
                          self._sd.copy().align(SpectralShape(360, 820, 2)))


class TestMultiSds_to_XYZ_integration(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.\
multi_sds_to_XYZ_integration` definition unit tests methods.
    """

    def test_multi_sds_to_XYZ_integration(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
multi_sds_to_XYZ_integration`
        definition.
        """

        cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
        np.testing.assert_almost_equal(
            multi_sds_to_XYZ_integration(MSDS, cmfs, ILLUMINANTS_SDS['D65']),
            XYZ_D65_INTEGRATION_MSDS,
            decimal=7)

        np.testing.assert_almost_equal(
            multi_sds_to_XYZ_integration(
                MSDS_ARRAY,
                cmfs,
                ILLUMINANTS_SDS['D65'],
                shape=SpectralShape(400, 700, 60)),
            XYZ_D65_ARRAY_INTEGRATION,
            decimal=7)

        np.testing.assert_almost_equal(
            multi_sds_to_XYZ_integration(
                MSDS_ARRAY,
                cmfs,
                ILLUMINANTS_SDS['D65'],
                1,
                shape=SpectralShape(400, 700, 60)),
            XYZ_D65_ARRAY_K1_INTEGRATION,
            decimal=7)

    def test_domain_range_scale_multi_sds_to_XYZ_integration(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
multi_sds_to_XYZ_integration` definition domain and range scale support.
        """

        cmfs = CMFS['CIE 1931 2 Degree Standard Observer']

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    multi_sds_to_XYZ_integration(
                        MSDS_ARRAY,
                        cmfs,
                        ILLUMINANTS_SDS['D65'],
                        shape=SpectralShape(400, 700, 60)),
                    XYZ_D65_ARRAY_INTEGRATION * factor,
                    decimal=7)


class TestMultiSds_to_XYZ_ASTME308(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.tristimulus.\
multi_sds_to_XYZ_ASTME308` definition unit tests methods.
    """

    def test_multi_sds_to_XYZ_ASTME308(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
multi_sds_to_XYZ_ASTME308`
        definition.
        """

        cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
        msds = MSDS.copy().align(SpectralShape(400, 700, 20))
        np.testing.assert_almost_equal(
            multi_sds_to_XYZ_ASTME308(msds, cmfs, ILLUMINANTS_SDS['D65']),
            XYZ_D65_ASTME308_MSDS,
            decimal=7)

        np.testing.assert_almost_equal(
            multi_sds_to_XYZ_ASTME308(msds, cmfs, ILLUMINANTS_SDS['D65'], k=1),
            XYZ_D65_ASTME308_K1_MSDS,
            decimal=7)

    def test_domain_range_scale_multi_sds_to_XYZ_ASTME308(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
multi_sds_to_XYZ_ASTME308` definition domain and range scale support.
        """

        cmfs = CMFS['CIE 1931 2 Degree Standard Observer']

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    multi_sds_to_XYZ_ASTME308(
                        MSDS.copy().align(SpectralShape(400, 700, 20)), cmfs,
                        ILLUMINANTS_SDS['D65']),
                    XYZ_D65_ASTME308_MSDS * factor,
                    decimal=7)

    def test_raise_exception_multi_sds_to_XYZ_ASTME308(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.\
multi_sds_to_XYZ_ASTME308` definition raise exception.
        """

        self.assertRaises(ValueError, multi_sds_to_XYZ_ASTME308, MSDS_ARRAY)


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
            wavelength_to_XYZ(480,
                              CMFS['CIE 1931 2 Degree Standard Observer']),
            np.array([0.09564, 0.13902, 0.81295]),
            decimal=7)

        np.testing.assert_almost_equal(
            wavelength_to_XYZ(480,
                              CMFS['CIE 2012 2 Degree Standard Observer']),
            np.array([0.08182895, 0.17880480, 0.75523790]),
            decimal=7)

        np.testing.assert_almost_equal(
            wavelength_to_XYZ(641.5,
                              CMFS['CIE 2012 2 Degree Standard Observer']),
            np.array([0.44575583, 0.18184213, 0.00000000]),
            decimal=7)

    def test_raise_exception_wavelength_to_XYZ(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.wavelength_to_XYZ`
        definition raised exception.
        """

        self.assertRaises(ValueError, wavelength_to_XYZ, 1)

        self.assertRaises(ValueError, wavelength_to_XYZ, 1000)

    def test_n_dimensional_wavelength_to_XYZ(self):
        """
        Tests :func:`colour.colorimetry.tristimulus.wavelength_to_XYZ`
        definition n-dimensional arrays support.
        """

        cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
        wl = 480
        XYZ = wavelength_to_XYZ(wl, cmfs)

        wl = np.tile(wl, 6)
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            wavelength_to_XYZ(wl, cmfs), XYZ, decimal=7)

        wl = np.reshape(wl, (2, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            wavelength_to_XYZ(wl, cmfs), XYZ, decimal=7)

        wl = np.reshape(wl, (2, 3, 1))
        XYZ = np.reshape(XYZ, (2, 3, 1, 3))
        np.testing.assert_almost_equal(
            wavelength_to_XYZ(wl, cmfs), XYZ, decimal=7)


if __name__ == '__main__':
    unittest.main()
