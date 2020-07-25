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

from colour.characterisation import COLOURCHECKER_SDS
from colour.colorimetry import (ILLUMINANTS, ILLUMINANT_SDS,
                                STANDARD_OBSERVER_CMFS, SpectralDistribution,
                                sd_to_XYZ)
from colour.difference import delta_E_CIE1976
from colour.models import RGB_COLOURSPACES, RGB_to_XYZ, XYZ_to_Lab
from colour.recovery.jakob2019 import (
    XYZ_to_sd_Jakob2019, sd_Jakob2019, error_function,
    dimensionalise_coefficients, DEFAULT_SPECTRAL_SHAPE_JAKOB_2019,
    ACCEPTABLE_DELTA_E, Jakob2019Interpolator)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_sd_Jakob2019', 'TestErrorFunction', 'TestJakob2019Interpolator'
]

CMFS = STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer']
sRGB = RGB_COLOURSPACES['sRGB']
PROPHOTO_RGB = RGB_COLOURSPACES['ProPhoto RGB']
D65 = SpectralDistribution(ILLUMINANT_SDS['D65'])
D65_XY = ILLUMINANTS['CIE 1931 2 Degree Standard Observer']["D65"]


class TestXYZ_to_sd_Jakob2019(unittest.TestCase):
    """
    Defines :func:`colour.recovery.jakob2019.XYZ_to_sd_Jakob2019`
    definition unit tests methods.
    """

    def test_roundtrip_colourchecker(self):
        """
        Tests :func:`colour.recovery.jakob2019.XYZ_to_sd_Jakob2019` definition
        round-trip errors using a color checker.
        """

        for name, sd in COLOURCHECKER_SDS['ColorChecker N Ohta'].items():
            # The colours aren't too saturated and the tests should pass with
            # or without feedback.
            for use_feedback in [None, 'adaptive-from-grey']:
                XYZ = sd_to_XYZ(sd, illuminant=D65) / 100

                _recovered_sd, error = XYZ_to_sd_Jakob2019(
                    XYZ, return_error=True, use_feedback=use_feedback)

                if error > ACCEPTABLE_DELTA_E:
                    self.fail('Delta E for \'{0}\' with use_feedback={1}'
                              ' is {2}'.format(name, use_feedback, error))


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
        shape = DEFAULT_SPECTRAL_SHAPE_JAKOB_2019
        aligned_cmfs = CMFS.copy().align(shape)
        illuminant = D65.copy().align(shape)
        illuminant_XYZ = sd_to_XYZ(D65)
        illuminant_XYZ /= illuminant_XYZ[1]

        for coefficients in coefficient_list:
            error, _derror, R, XYZ, Lab = error_function(
                coefficients,
                self._Lab_e,
                DEFAULT_SPECTRAL_SHAPE_JAKOB_2019,
                aligned_cmfs,
                illuminant,
                illuminant_XYZ,
                return_intermediates=True)

            sd = sd_Jakob2019(
                dimensionalise_coefficients(coefficients, shape), shape)

            sd_XYZ = sd_to_XYZ(sd, illuminant=illuminant) / 100
            sd_Lab = XYZ_to_Lab(XYZ, D65_XY)
            error_reference = delta_E_CIE1976(self._Lab_e, Lab)

            np.testing.assert_allclose(sd.values, R, atol=1e-14)
            np.testing.assert_allclose(XYZ, sd_XYZ, atol=1e-14)
            self.assertLess(abs(error_reference - error), ACCEPTABLE_DELTA_E)
            self.assertLess(delta_E_CIE1976(Lab, sd_Lab), ACCEPTABLE_DELTA_E)

    def test_derivatives(self):
        """
        Compares gradients computed using closed-form expressions of
        derivatives with finite difference approximations.
        """

        shape = DEFAULT_SPECTRAL_SHAPE_JAKOB_2019
        aligned_cmfs = CMFS.copy().align(shape)
        illuminant = D65.copy().align(shape)
        illuminant_XYZ = sd_to_XYZ(D65)
        illuminant_XYZ /= illuminant_XYZ[1]

        var_range = np.linspace(-10, 10, 1000)
        h = var_range[1] - var_range[0]

        # Vary one coefficient at a time, keeping the others fixed to 1.
        for coeff_index in range(3):
            errors = np.empty_like(var_range)
            derrors = np.empty_like(var_range)

            for i, var in enumerate(var_range):
                coefficients = np.array([1.0, 1, 1])
                coefficients[coeff_index] = var

                error, derror = error_function(coefficients, self._Lab_e,
                                               shape, aligned_cmfs, illuminant,
                                               illuminant_XYZ)

                errors[i] = error
                derrors[i] = derror[coeff_index]

            staggered_derrors = (derrors[:-1] + derrors[1:]) / 2
            approximate_derrors = np.diff(errors) / h

            # The approximated derivatives aren't too accurate, so tolerances
            # have to be rather loose.
            np.testing.assert_allclose(
                staggered_derrors, approximate_derrors, atol=1e-3, rtol=1e-2)


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

    def test_interpolator(self):
        """
        Tests the entirety of the
        :class:`colour.recovery.jakob2019.Jakob2019Interpolator`class.
        """

        interpolator = Jakob2019Interpolator()
        interpolator.generate(sRGB, CMFS, D65, 4, verbose=True)

        path = os.path.join(self._temporary_directory, 'Jakob2019_Test.coeff')
        interpolator.write(path)
        interpolator.read(path)

        for RGB in [
                np.array([1., 0., 0.]),
                np.array([0., 1., 0.]),
                np.array([0., 0., 1.]),
                np.array([0., 0., 0.]),
                np.array([0.5, 0.5, 0.5]),
                np.array([1., 1., 1.])
        ]:
            XYZ = RGB_to_XYZ(
                RGB,
                sRGB.whitepoint,
                D65_XY,
                sRGB.RGB_to_XYZ_matrix,
            )
            Lab = XYZ_to_Lab(XYZ, D65_XY)

            recovered_sd = interpolator.RGB_to_sd(RGB)
            recovered_XYZ = sd_to_XYZ(recovered_sd, illuminant=D65) / 100
            recovered_Lab = XYZ_to_Lab(recovered_XYZ, D65_XY)

            error = delta_E_CIE1976(Lab, recovered_Lab)
            if error > 2 * ACCEPTABLE_DELTA_E:
                self.fail('Delta E for RGB={0} in colourspace {1} is {2}'
                          .format(RGB, sRGB.name, error))


if __name__ == '__main__':
    unittest.main()
