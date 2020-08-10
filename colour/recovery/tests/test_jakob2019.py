# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.recovery.jakob2019` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import shutil
import tempfile
import unittest

from colour.characterisation import SDS_COLOURCHECKERS
from colour.colorimetry import (CCS_ILLUMINANTS, SDS_ILLUMINANTS,
                                MSDS_CMFS_STANDARD_OBSERVER, sd_to_XYZ)
from colour.difference import JND_CIE1976, delta_E_CIE1976
from colour.models import RGB_COLOURSPACES, RGB_to_XYZ, XYZ_to_Lab
from colour.recovery.jakob2019 import (
    XYZ_to_sd_Jakob2019, sd_Jakob2019, error_function,
    dimensionalise_coefficients, JAKOB2019_SPECTRAL_SHAPE,
    Jakob2019Interpolator)
from colour.utilities import domain_range_scale, full, ones, zeros

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestErrorFunction', 'TestXYZ_to_sd_Jakob2019', 'TestJakob2019Interpolator'
]

MSDS_CMFS = MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
RGB_COLOURSPACE_sRGB = RGB_COLOURSPACES['sRGB']
RGB_COLOURSPACE_PROPHOTO_RGB = RGB_COLOURSPACES['ProPhoto RGB']
SD_D65 = SDS_ILLUMINANTS['D65']
CCS_D65 = CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']


class TestErrorFunction(unittest.TestCase):
    """
    Defines :func:`colour.recovery.jakob2019.error_function` definition unit
    tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._Lab_e = np.array([72, -20, 61])

    def test_compare_intermediates(self):
        """
        Compares intermediate results of
        :func:`colour.recovery.jakob2019.error_function` with
        :func:`colour.sd_to_XYZ`, :func:`colour.XYZ_to_Lab` and checks if the
        error is computed correctly by comparing it with
        :func:`colour.difference.delta_E_CIE1976`.
        """

        # Quoted names refer to colours from ColorChecker N Ohta (using D65).
        coefficient_list = [
            np.array([0, 0, 0]),  # 50% gray
            np.array([0, 0, -1e+9]),  # Pure black
            np.array([0, 0, +1e+9]),  # Pure white
            np.array([1e+9, -1e+9, 2.1e+8]),  # A pathological example
            np.array([2.2667394, -7.6313081, 1.03185]),  # 'blue'
            np.array([-31.377077, 26.810094, -6.1139927]),  # 'green'
            np.array([25.064246, -16.072039, 0.10431365]),  # 'red'
            np.array([-19.325667, 22.242319, -5.8144924]),  # 'yellow'
            np.array([21.909902, -17.227963, 2.142351]),  # 'magenta'
            np.array([-15.864009, 8.6735071, -1.4012552]),  # 'cyan'
        ]

        # error_function will not align these for us.
        shape = JAKOB2019_SPECTRAL_SHAPE
        aligned_cmfs = MSDS_CMFS.copy().align(shape)
        illuminant = SD_D65.copy().align(shape)
        XYZ_n = sd_to_XYZ(SD_D65)
        XYZ_n /= XYZ_n[1]

        for coefficients in coefficient_list:
            error, _derror, R, XYZ, Lab = error_function(
                coefficients,
                self._Lab_e,
                aligned_cmfs,
                illuminant,
                additional_data=True)

            sd = sd_Jakob2019(
                dimensionalise_coefficients(coefficients, shape), shape)

            sd_XYZ = sd_to_XYZ(sd, illuminant=illuminant) / 100
            sd_Lab = XYZ_to_Lab(XYZ, CCS_D65)
            error_reference = delta_E_CIE1976(self._Lab_e, Lab)

            np.testing.assert_allclose(sd.values, R, atol=1e-14)
            np.testing.assert_allclose(XYZ, sd_XYZ, atol=1e-14)
            self.assertLess(abs(error_reference - error), JND_CIE1976)
            self.assertLess(delta_E_CIE1976(Lab, sd_Lab), JND_CIE1976)

    def test_derivatives(self):
        """
        Compares gradients computed using closed-form expressions of
        derivatives with finite difference approximations.
        """

        shape = JAKOB2019_SPECTRAL_SHAPE
        aligned_cmfs = MSDS_CMFS.copy().align(shape)
        illuminant = SD_D65.copy().align(shape)
        XYZ_n = sd_to_XYZ(SD_D65)
        XYZ_n /= XYZ_n[1]

        var_range = np.linspace(-10, 10, 1000)
        h = var_range[1] - var_range[0]

        # Vary one coefficient at a time, keeping the others fixed to 1.
        for coefficient_i in range(3):
            errors = np.empty_like(var_range)
            derrors = np.empty_like(var_range)

            for i, var in enumerate(var_range):
                coefficients = ones(3)
                coefficients[coefficient_i] = var

                error, derror = error_function(coefficients, self._Lab_e,
                                               aligned_cmfs, illuminant)

                errors[i] = error
                derrors[i] = derror[coefficient_i]

            staggered_derrors = (derrors[:-1] + derrors[1:]) / 2
            approximate_derrors = np.diff(errors) / h

            # The approximated derivatives aren't too accurate, so tolerances
            # have to be rather loose.
            np.testing.assert_allclose(
                staggered_derrors, approximate_derrors, atol=1e-3, rtol=1e-2)


class TestXYZ_to_sd_Jakob2019(unittest.TestCase):
    """
    Defines :func:`colour.recovery.jakob2019.XYZ_to_sd_Jakob2019` definition
    unit tests methods.
    """

    def test_XYZ_to_sd_Jakob2019(self):
        """
        Tests :func:`colour.recovery.jakob2019.XYZ_to_sd_Jakob2019` definition.
        """

        # Tests the round-trip with values of a colour checker.
        for name, sd in SDS_COLOURCHECKERS['ColorChecker N Ohta'].items():
            XYZ = sd_to_XYZ(sd, illuminant=SD_D65) / 100

            _recovered_sd, error = XYZ_to_sd_Jakob2019(
                XYZ, illuminant=SD_D65, additional_data=True)

            if error > JND_CIE1976:
                self.fail('Delta E for \'{0}\' is {1}!'.format(name, error))

    def test_domain_range_scale_XYZ_to_sd_Jakob2019(self):
        """
        Tests :func:`colour.recovery.jakob2019.XYZ_to_sd_Jakob2019` definition
        domain and range scale support.
        """

        XYZ_i = np.array([0.21781186, 0.12541048, 0.04697113])
        XYZ_o = sd_to_XYZ(XYZ_to_sd_Jakob2019(XYZ_i))

        d_r = (('reference', 1, 1), (1, 1, 0.01), (100, 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    sd_to_XYZ(XYZ_to_sd_Jakob2019(XYZ_i * factor_a)),
                    XYZ_o * factor_b,
                    decimal=7)


class TestJakob2019Interpolator(unittest.TestCase):
    """
    Defines :class:`colour.recovery.jakob2019.Jakob2019Interpolator`
    definition unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_Jakob2019Interpolator(self):
        """
        Tests the entirety of the
        :class:`colour.recovery.jakob2019.Jakob2019Interpolator`class.
        """

        interpolator = Jakob2019Interpolator()
        interpolator.generate(
            RGB_COLOURSPACE_sRGB, MSDS_CMFS, SD_D65, 4, verbose=True)

        path = os.path.join(self._temporary_directory, 'Jakob2019_Test.coeff')
        interpolator.write(path)
        interpolator.read(path)

        for RGB in [
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1]),
                zeros(3),
                full(3, 0.5),
                ones(3),
        ]:
            XYZ = RGB_to_XYZ(RGB, RGB_COLOURSPACE_sRGB.whitepoint, CCS_D65,
                             RGB_COLOURSPACE_sRGB.RGB_to_XYZ_matrix)
            Lab = XYZ_to_Lab(XYZ, CCS_D65)

            recovered_sd = interpolator.RGB_to_sd(RGB)
            recovered_XYZ = sd_to_XYZ(recovered_sd, illuminant=SD_D65) / 100
            recovered_Lab = XYZ_to_Lab(recovered_XYZ, CCS_D65)

            error = delta_E_CIE1976(Lab, recovered_Lab)
            if error > 2 * JND_CIE1976:
                self.fail('Delta E for RGB={0} in colourspace {1} is {2}!'
                          .format(RGB, RGB_COLOURSPACE_sRGB.name, error))


if __name__ == '__main__':
    unittest.main()
