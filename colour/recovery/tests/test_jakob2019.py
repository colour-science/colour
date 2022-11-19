# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.recovery.jakob2019` module."""

import numpy as np
import os
import shutil
import tempfile
import unittest

from colour.characterisation import SDS_COLOURCHECKERS
from colour.colorimetry import handle_spectral_arguments, sd_to_XYZ
from colour.difference import JND_CIE1976, delta_E_CIE1976
from colour.models import (
    RGB_COLOURSPACE_sRGB,
    RGB_to_XYZ,
    XYZ_to_Lab,
    XYZ_to_xy,
)
from colour.recovery.jakob2019 import (
    XYZ_to_sd_Jakob2019,
    sd_Jakob2019,
    error_function,
    dimensionalise_coefficients,
    SPECTRAL_SHAPE_JAKOB2019,
    LUT3D_Jakob2019,
)
from colour.utilities import domain_range_scale, full, ones, zeros

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestErrorFunction",
    "TestXYZ_to_sd_Jakob2019",
    "TestLUT3D_Jakob2019",
]


class TestErrorFunction(unittest.TestCase):
    """
    Define :func:`colour.recovery.jakob2019.error_function` definition unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._shape = SPECTRAL_SHAPE_JAKOB2019
        self._cmfs, self._sd_D65 = handle_spectral_arguments(
            shape_default=self._shape
        )
        self._XYZ_D65 = sd_to_XYZ(self._sd_D65)
        self._xy_D65 = XYZ_to_xy(self._XYZ_D65)

        self._Lab_e = np.array([72, -20, 61])

    def test_intermediates(self):
        """
        Test intermediate results of
        :func:`colour.recovery.jakob2019.error_function` with
        :func:`colour.sd_to_XYZ`, :func:`colour.XYZ_to_Lab` and checks if the
        error is computed correctly by comparing it with
        :func:`colour.difference.delta_E_CIE1976`.
        """

        # Quoted names refer to colours from ColorChecker N Ohta (using D65).
        coefficient_list = [
            np.array([0, 0, 0]),  # 50% gray
            np.array([0, 0, -1e9]),  # Pure black
            np.array([0, 0, +1e9]),  # Pure white
            np.array([1e9, -1e9, 2.1e8]),  # A pathological example
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
                additional_data=True,
            )

            sd = sd_Jakob2019(
                dimensionalise_coefficients(coefficients, self._shape),
                self._shape,
            )

            sd_XYZ = sd_to_XYZ(sd, self._cmfs, self._sd_D65) / 100
            sd_Lab = XYZ_to_Lab(XYZ, self._xy_D65)
            error_reference = delta_E_CIE1976(self._Lab_e, Lab)

            np.testing.assert_allclose(sd.values, R, atol=1e-14)
            np.testing.assert_allclose(XYZ, sd_XYZ, atol=1e-14)

            self.assertLess(abs(error_reference - error), JND_CIE1976 / 100)
            self.assertLess(delta_E_CIE1976(Lab, sd_Lab), JND_CIE1976 / 100)

    def test_derivatives(self):
        """
        Test the gradients computed using closed-form expressions of the
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

                error, derror = error_function(
                    coefficients, self._Lab_e, self._cmfs, self._sd_D65
                )

                errors[i] = error
                derrors[i] = derror[coefficient_i]

            staggered_derrors = (derrors[:-1] + derrors[1:]) / 2
            approximate_derrors = np.diff(errors) / ds

            # The approximated derivatives aren't too accurate, so tolerances
            # have to be rather loose.
            np.testing.assert_allclose(
                staggered_derrors, approximate_derrors, atol=1e-3, rtol=1e-2
            )


class TestXYZ_to_sd_Jakob2019(unittest.TestCase):
    """
    Define :func:`colour.recovery.jakob2019.XYZ_to_sd_Jakob2019` definition
    unit tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._shape = SPECTRAL_SHAPE_JAKOB2019
        self._cmfs, self._sd_D65 = handle_spectral_arguments(
            shape_default=self._shape
        )

    def test_XYZ_to_sd_Jakob2019(self):
        """Test :func:`colour.recovery.jakob2019.XYZ_to_sd_Jakob2019` definition."""

        # Tests the round-trip with values of a colour checker.
        for name, sd in SDS_COLOURCHECKERS["ColorChecker N Ohta"].items():
            XYZ = sd_to_XYZ(sd, self._cmfs, self._sd_D65) / 100

            _recovered_sd, error = XYZ_to_sd_Jakob2019(
                XYZ, self._cmfs, self._sd_D65, additional_data=True
            )

            if error > JND_CIE1976 / 100:  # pragma: no cover
                self.fail(f"Delta E for '{name}' is {error}!")

    def test_domain_range_scale_XYZ_to_sd_Jakob2019(self):
        """
        Test :func:`colour.recovery.jakob2019.XYZ_to_sd_Jakob2019` definition
        domain and range scale support.
        """

        XYZ_i = np.array([0.20654008, 0.12197225, 0.05136952])
        XYZ_o = sd_to_XYZ(
            XYZ_to_sd_Jakob2019(XYZ_i, self._cmfs, self._sd_D65),
            self._cmfs,
            self._sd_D65,
        )

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    sd_to_XYZ(
                        XYZ_to_sd_Jakob2019(
                            XYZ_i * factor_a, self._cmfs, self._sd_D65
                        ),
                        self._cmfs,
                        self._sd_D65,
                    ),
                    XYZ_o * factor_b,
                    decimal=7,
                )


class TestLUT3D_Jakob2019(unittest.TestCase):
    """
    Define :class:`colour.recovery.jakob2019.LUT3D_Jakob2019` definition unit
    tests methods.
    """

    @classmethod
    def generate_LUT(self):
        """
        Generate the *LUT* used for the unit tests.

        Notes
        -----
        -   This method is used to define the slow common tests attributes once
            rather than using the typical :meth:`unittest.TestCase.setUp` that
            is invoked once per-test.
        """

        if not hasattr(self, "_LUT"):
            self._shape = SPECTRAL_SHAPE_JAKOB2019
            self._cmfs, self._sd_D65 = handle_spectral_arguments(
                shape_default=self._shape
            )
            self._XYZ_D65 = sd_to_XYZ(self._sd_D65)
            self._xy_D65 = XYZ_to_xy(self._XYZ_D65)

            self._RGB_colourspace = RGB_COLOURSPACE_sRGB

            self._LUT = LUT3D_Jakob2019()
            self._LUT.generate(
                self._RGB_colourspace, self._cmfs, self._sd_D65, 5
            )

        return self._LUT

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = (
            "size",
            "lightness_scale",
            "coefficients",
            "interpolator",
        )

        for attribute in required_attributes:
            self.assertIn(attribute, dir(LUT3D_Jakob2019))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "generate",
            "RGB_to_coefficients",
            "RGB_to_sd",
            "read",
            "write",
        )

        for method in required_methods:
            self.assertIn(method, dir(LUT3D_Jakob2019))

    def test_size(self):
        """Test :attr:`colour.recovery.jakob2019.LUT3D_Jakob2019.size` property."""

        self.assertEqual(TestLUT3D_Jakob2019.generate_LUT().size, 5)

    def test_lightness_scale(self):
        """
        Test :attr:`colour.recovery.jakob2019.LUT3D_Jakob2019.lightness_scale`
        property.
        """

        np.testing.assert_array_almost_equal(
            TestLUT3D_Jakob2019.generate_LUT().lightness_scale,
            np.array(
                [0.00000000, 0.06561279, 0.50000000, 0.93438721, 1.00000000]
            ),
            decimal=7,
        )

    def test_coefficients(self):
        """
        Test :attr:`colour.recovery.jakob2019.LUT3D_Jakob2019.coefficients`
        property.
        """

        self.assertTupleEqual(
            TestLUT3D_Jakob2019.generate_LUT().coefficients.shape,
            (3, 5, 5, 5, 3),
        )

    def test_LUT3D_Jakob2019(self):
        """
        Test the entirety of the
        :class:`colour.recovery.jakob2019.LUT3D_Jakob2019`class.
        """

        LUT = TestLUT3D_Jakob2019.generate_LUT()

        temporary_directory = tempfile.mkdtemp()
        path = os.path.join(temporary_directory, "Test_Jakob2019.coeff")

        try:
            LUT.write(path)
            LUT_t = LUT3D_Jakob2019().read(path)

            np.testing.assert_array_almost_equal(
                LUT.lightness_scale, LUT_t.lightness_scale, decimal=7
            )
            np.testing.assert_allclose(
                LUT.coefficients,
                LUT_t.coefficients,
                atol=0.000001,
                rtol=0.000001,
            )
        finally:
            shutil.rmtree(temporary_directory)

        for RGB in [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            zeros(3),
            full(3, 0.5),
            ones(3),
        ]:
            XYZ = RGB_to_XYZ(
                RGB,
                self._RGB_colourspace.whitepoint,
                self._xy_D65,
                self._RGB_colourspace.matrix_RGB_to_XYZ,
            )
            Lab = XYZ_to_Lab(XYZ, self._xy_D65)

            recovered_sd = LUT.RGB_to_sd(RGB)
            recovered_XYZ = (
                sd_to_XYZ(recovered_sd, self._cmfs, self._sd_D65) / 100
            )
            recovered_Lab = XYZ_to_Lab(recovered_XYZ, self._xy_D65)

            error = delta_E_CIE1976(Lab, recovered_Lab)

            if error > 2 * JND_CIE1976 / 100:  # pragma: no cover
                self.fail(
                    f"Delta E for RGB={RGB} in colourspace "
                    f"{self._RGB_colourspace.name} is {error}!"
                )

    def test_raise_exception_RGB_to_coefficients(self):
        """
        Test :func:`colour.recovery.jakob2019.LUT3D_Jakob2019.\
RGB_to_coefficients` method raised exception.
        """

        LUT = LUT3D_Jakob2019()

        self.assertRaises(RuntimeError, LUT.RGB_to_coefficients, np.array([]))

    def test_raise_exception_read(self):
        """
        Test :func:`colour.recovery.jakob2019.LUT3D_Jakob2019.read` method
        raised exception.
        """

        LUT = LUT3D_Jakob2019()
        self.assertRaises(ValueError, LUT.read, __file__)


if __name__ == "__main__":
    unittest.main()
