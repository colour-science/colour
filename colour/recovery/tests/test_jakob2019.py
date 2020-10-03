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
from colour.models import RGB_COLOURSPACE_sRGB, RGB_to_XYZ, XYZ_to_Lab
from colour.recovery.jakob2019 import (
    XYZ_to_sd_Jakob2019, sd_Jakob2019, error_function,
    dimensionalise_coefficients, SPECTRAL_SHAPE_JAKOB2019, LUT3D_Jakob2019)
from colour.utilities import domain_range_scale, full, ones, zeros

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestErrorFunction', 'TestXYZ_to_sd_Jakob2019', 'TestLUT3D_Jakob2019'
]


class TestErrorFunction(unittest.TestCase):
    """
    Defines :func:`colour.recovery.jakob2019.error_function` definition unit
    tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._shape = SPECTRAL_SHAPE_JAKOB2019
        self._cmfs = MSDS_CMFS_STANDARD_OBSERVER[
            'CIE 1931 2 Degree Standard Observer'].copy().align(self._shape)

        self._sd_D65 = SDS_ILLUMINANTS['D65'].copy().align(self._shape)
        self._XYZ_D65 = sd_to_XYZ(self._sd_D65)
        self._XYZ_D65 /= self._XYZ_D65[1]
        self._xy_D65 = CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
            'D65']

        self._Lab_e = np.array([72, -20, 61])

    def test_intermediates(self):
        """
        Tests intermediate results of
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

        for coefficients in coefficient_list:
            error, _derror, R, XYZ, Lab = error_function(
                coefficients,
                self._Lab_e,
                self._cmfs,
                self._sd_D65,
                additional_data=True)

            sd = sd_Jakob2019(
                dimensionalise_coefficients(coefficients, self._shape),
                self._shape)

            sd_XYZ = sd_to_XYZ(sd, self._cmfs, self._sd_D65) / 100
            sd_Lab = XYZ_to_Lab(XYZ, self._xy_D65)
            error_reference = delta_E_CIE1976(self._Lab_e, Lab)

            np.testing.assert_allclose(sd.values, R, atol=1e-14)
            np.testing.assert_allclose(XYZ, sd_XYZ, atol=1e-14)

            self.assertLess(abs(error_reference - error), JND_CIE1976 / 100)
            self.assertLess(delta_E_CIE1976(Lab, sd_Lab), JND_CIE1976 / 100)

    def test_derivatives(self):
        """
        Tests the gradients computed using closed-form expressions of the
        derivatives with finite difference approximations.
        """

        samples = np.linspace(-10, 10, 1000)
        ds = samples[1] - samples[0]

        # Vary one coefficient at a time, keeping the others fixed to 1.
        for coefficient_i in range(3):
            errors = np.empty_like(samples)
            derrors = np.empty_like(samples)

            for i, sample in enumerate(samples):
                coefficients = ones(3)
                coefficients[coefficient_i] = sample

                error, derror = error_function(coefficients, self._Lab_e,
                                               self._cmfs, self._sd_D65)

                errors[i] = error
                derrors[i] = derror[coefficient_i]

            staggered_derrors = (derrors[:-1] + derrors[1:]) / 2
            approximate_derrors = np.diff(errors) / ds

            # The approximated derivatives aren't too accurate, so tolerances
            # have to be rather loose.
            np.testing.assert_allclose(
                staggered_derrors, approximate_derrors, atol=1e-3, rtol=1e-2)


class TestXYZ_to_sd_Jakob2019(unittest.TestCase):
    """
    Defines :func:`colour.recovery.jakob2019.XYZ_to_sd_Jakob2019` definition
    unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._shape = SPECTRAL_SHAPE_JAKOB2019
        self._cmfs = MSDS_CMFS_STANDARD_OBSERVER[
            'CIE 1931 2 Degree Standard Observer'].copy().align(self._shape)
        self._sd_D65 = SDS_ILLUMINANTS['D65'].copy().align(self._shape)

    def test_XYZ_to_sd_Jakob2019(self):
        """
        Tests :func:`colour.recovery.jakob2019.XYZ_to_sd_Jakob2019` definition.
        """

        # Tests the round-trip with values of a colour checker.
        for name, sd in SDS_COLOURCHECKERS['ColorChecker N Ohta'].items():
            XYZ = sd_to_XYZ(sd, self._cmfs, self._sd_D65) / 100

            _recovered_sd, error = XYZ_to_sd_Jakob2019(
                XYZ, self._cmfs, self._sd_D65, additional_data=True)

            if error > JND_CIE1976 / 100:
                self.fail('Delta E for \'{0}\' is {1}!'.format(name, error))

    def test_domain_range_scale_XYZ_to_sd_Jakob2019(self):
        """
        Tests :func:`colour.recovery.jakob2019.XYZ_to_sd_Jakob2019` definition
        domain and range scale support.
        """

        XYZ_i = np.array([0.20654008, 0.12197225, 0.05136952])
        XYZ_o = sd_to_XYZ(
            XYZ_to_sd_Jakob2019(XYZ_i, self._cmfs, self._sd_D65), self._cmfs,
            self._sd_D65)

        d_r = (('reference', 1, 1), (1, 1, 0.01), (100, 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    sd_to_XYZ(
                        XYZ_to_sd_Jakob2019(XYZ_i * factor_a, self._cmfs,
                                            self._sd_D65), self._cmfs,
                        self._sd_D65),
                    XYZ_o * factor_b,
                    decimal=7)


class TestLUT3D_Jakob2019(unittest.TestCase):
    """
    Defines :class:`colour.recovery.jakob2019.LUT3D_Jakob2019`
    definition unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._RGB_colourspace = RGB_COLOURSPACE_sRGB

        self._shape = SPECTRAL_SHAPE_JAKOB2019
        self._cmfs = MSDS_CMFS_STANDARD_OBSERVER[
            'CIE 1931 2 Degree Standard Observer'].copy().align(self._shape)

        self._sd_D65 = SDS_ILLUMINANTS['D65'].copy().align(self._shape)
        self._xy_D65 = CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
            'D65']

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('lightness_scale', 'coefficients', 'size',
                               'interpolator')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(LUT3D_Jakob2019))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__init__', 'generate', 'RGB_to_coefficients',
                            'RGB_to_sd', 'read', 'write')

        for method in required_methods:
            self.assertIn(method, dir(LUT3D_Jakob2019))

    def test_LUT3D_Jakob2019(self):
        """
        Tests the entirety of the
        :class:`colour.recovery.jakob2019.LUT3D_Jakob2019`class.
        """

        LUT = LUT3D_Jakob2019()
        LUT.generate(self._RGB_colourspace, self._cmfs, self._sd_D65, 5)

        path = os.path.join(self._temporary_directory, 'Test_Jakob2019.coeff')

        LUT.write(path)
        LUT.read(path)

        for RGB in [
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1]),
                zeros(3),
                full(3, 0.5),
                ones(3),
        ]:
            XYZ = RGB_to_XYZ(RGB, self._RGB_colourspace.whitepoint,
                             self._xy_D65,
                             self._RGB_colourspace.matrix_RGB_to_XYZ)
            Lab = XYZ_to_Lab(XYZ, self._xy_D65)

            recovered_sd = LUT.RGB_to_sd(RGB)
            recovered_XYZ = sd_to_XYZ(recovered_sd, self._cmfs,
                                      self._sd_D65) / 100
            recovered_Lab = XYZ_to_Lab(recovered_XYZ, self._xy_D65)

            error = delta_E_CIE1976(Lab, recovered_Lab)

            if error > 2 * JND_CIE1976 / 100:
                self.fail('Delta E for RGB={0} in colourspace {1} is {2}!'
                          .format(RGB, self._RGB_colourspace.name, error))


if __name__ == '__main__':
    unittest.main()
