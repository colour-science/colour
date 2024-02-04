"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.itut_h_273` module.
"""

import unittest

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    eotf_H273_ST428_1,
    eotf_inverse_H273_ST428_1,
    oetf_H273_IEC61966_2,
    oetf_H273_Log,
    oetf_H273_LogSqrt,
    oetf_inverse_H273_IEC61966_2,
    oetf_inverse_H273_Log,
    oetf_inverse_H273_LogSqrt,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestOetf_H273_Log",
    "TestOetf_inverse_H273_Log",
    "TestOetf_H273_LogSqrt",
    "TestOetf_inverse_H273_LogSqrt",
    "TestOetf_H273_IEC61966_2",
    "TestOetf_inverse_H273_IEC61966_2",
    "TestEotf_inverse_H273_ST428_1",
    "TestEotf_H273_ST428_1",
]


class TestOetf_H273_Log(unittest.TestCase):
    """
        Define :func:`colour.models.rgb.transfer_functions.itut_h_273.
    oetf_H273_Log` definition unit tests methods.
    """

    def test_oetf_H273_Log(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_Log` definition.
        """

        np.testing.assert_allclose(
            oetf_H273_Log(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            oetf_H273_Log(0.18),
            0.627636252551653,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_H273_Log(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_oetf_H273_Log(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_Log` definition n-dimensional arrays support.
        """

        E = 0.18
        E_p = oetf_H273_Log(E)

        E = np.tile(E, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_allclose(oetf_H273_Log(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = np.reshape(E, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_allclose(oetf_H273_Log(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = np.reshape(E, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_allclose(oetf_H273_Log(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_H273_Log(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_Log` definition domain and range scale support.
        """

        E = 0.18
        E_p = oetf_H273_Log(E)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    oetf_H273_Log(E * factor),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_H273_Log(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_Log` definition nan support.
        """

        oetf_H273_Log(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_H273_Log(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_Log` definition unit tests methods.
    """

    def test_oetf_inverse_H273_Log(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_Log` definition.
        """

        # NOTE: The function is unfortunately clamped and cannot roundtrip
        # properly.
        np.testing.assert_allclose(
            oetf_inverse_H273_Log(0.0), 0.01, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            oetf_inverse_H273_Log(0.627636252551653),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_inverse_H273_Log(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_oetf_inverse_H273_Log(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_Log` definition n-dimensional arrays support.
        """

        E_p = 0.627636252551653
        E = oetf_inverse_H273_Log(E_p)

        E_p = np.tile(E_p, 6)
        E = np.tile(E, 6)
        np.testing.assert_allclose(
            oetf_inverse_H273_Log(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E_p = np.reshape(E_p, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_allclose(
            oetf_inverse_H273_Log(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E_p = np.reshape(E_p, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_allclose(
            oetf_inverse_H273_Log(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_oetf_inverse_H273_Log(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_Log` definition domain and range scale support.
        """

        E_p = 0.627636252551653
        E = oetf_inverse_H273_Log(E_p)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    oetf_inverse_H273_Log(E_p * factor),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_H273_Log(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_Log` definition nan support.
        """

        oetf_inverse_H273_Log(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_H273_LogSqrt(unittest.TestCase):
    """
        Define :func:`colour.models.rgb.transfer_functions.itut_h_273.
    oetf_H273_LogSqrt` definition unit tests methods.
    """

    def test_oetf_H273_LogSqrt(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_LogSqrt` definition.
        """

        np.testing.assert_allclose(
            oetf_H273_LogSqrt(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            oetf_H273_LogSqrt(0.18),
            0.702109002041322,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_H273_LogSqrt(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_oetf_H273_LogSqrt(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_LogSqrt` definition n-dimensional arrays support.
        """

        E = 0.18
        E_p = oetf_H273_LogSqrt(E)

        E = np.tile(E, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_allclose(
            oetf_H273_LogSqrt(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E = np.reshape(E, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_allclose(
            oetf_H273_LogSqrt(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E = np.reshape(E, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_allclose(
            oetf_H273_LogSqrt(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_oetf_H273_LogSqrt(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_LogSqrt` definition domain and range scale support.
        """

        E = 0.18
        E_p = oetf_H273_LogSqrt(E)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    oetf_H273_LogSqrt(E * factor),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_H273_LogSqrt(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_LogSqrt` definition nan support.
        """

        oetf_H273_LogSqrt(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_H273_LogSqrt(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_LogSqrt` definition unit tests methods.
    """

    def test_oetf_inverse_H273_LogSqrt(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_LogSqrt` definition.
        """

        # NOTE: The function is unfortunately clamped and cannot roundtrip
        # properly.
        np.testing.assert_allclose(
            oetf_inverse_H273_LogSqrt(0.0),
            0.003162277660168,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_inverse_H273_LogSqrt(0.702109002041322),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_inverse_H273_LogSqrt(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_oetf_inverse_H273_LogSqrt(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_LogSqrt` definition n-dimensional arrays support.
        """

        E_p = 0.702109002041322
        E = oetf_inverse_H273_LogSqrt(E_p)

        E_p = np.tile(E_p, 6)
        E = np.tile(E, 6)
        np.testing.assert_allclose(
            oetf_inverse_H273_LogSqrt(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E_p = np.reshape(E_p, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_allclose(
            oetf_inverse_H273_LogSqrt(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E_p = np.reshape(E_p, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_allclose(
            oetf_inverse_H273_LogSqrt(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_oetf_inverse_H273_LogSqrt(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_LogSqrt` definition domain and range scale support.
        """

        E_p = 0.702109002041322
        E = oetf_inverse_H273_LogSqrt(E_p)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    oetf_inverse_H273_LogSqrt(E_p * factor),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_H273_LogSqrt(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_LogSqrt` definition nan support.
        """

        oetf_inverse_H273_LogSqrt(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_H273_IEC61966_2(unittest.TestCase):
    """
        Define :func:`colour.models.rgb.transfer_functions.itut_h_273.
    oetf_H273_IEC61966_2` definition unit tests methods.
    """

    def test_oetf_H273_IEC61966_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_IEC61966_2` definition.
        """

        np.testing.assert_allclose(
            oetf_H273_IEC61966_2(-0.18),
            -0.461356129500442,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_H273_IEC61966_2(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            oetf_H273_IEC61966_2(0.18),
            0.461356129500442,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_H273_IEC61966_2(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_oetf_H273_IEC61966_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_IEC61966_2` definition n-dimensional arrays support.
        """

        E = 0.18
        E_p = oetf_H273_IEC61966_2(E)

        E = np.tile(E, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_allclose(
            oetf_H273_IEC61966_2(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E = np.reshape(E, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_allclose(
            oetf_H273_IEC61966_2(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E = np.reshape(E, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_allclose(
            oetf_H273_IEC61966_2(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_oetf_H273_IEC61966_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_IEC61966_2` definition domain and range scale support.
        """

        E = 0.18
        E_p = oetf_H273_IEC61966_2(E)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    oetf_H273_IEC61966_2(E * factor),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_H273_IEC61966_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_IEC61966_2` definition nan support.
        """

        oetf_H273_IEC61966_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_H273_IEC61966_2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_IEC61966_2` definition unit tests methods.
    """

    def test_oetf_inverse_H273_IEC61966_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_IEC61966_2` definition.
        """

        np.testing.assert_allclose(
            oetf_inverse_H273_IEC61966_2(-0.461356129500442),
            -0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_inverse_H273_IEC61966_2(0.0),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_inverse_H273_IEC61966_2(0.461356129500442),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_inverse_H273_IEC61966_2(1.0),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_inverse_H273_IEC61966_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_IEC61966_2` definition n-dimensional arrays support.
        """

        E_p = 0.627636252551653
        E = oetf_inverse_H273_IEC61966_2(E_p)

        E_p = np.tile(E_p, 6)
        E = np.tile(E, 6)
        np.testing.assert_allclose(
            oetf_inverse_H273_IEC61966_2(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E_p = np.reshape(E_p, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_allclose(
            oetf_inverse_H273_IEC61966_2(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E_p = np.reshape(E_p, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_allclose(
            oetf_inverse_H273_IEC61966_2(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_oetf_inverse_H273_IEC61966_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_IEC61966_2` definition domain and range scale support.
        """

        E_p = 0.627636252551653
        E = oetf_inverse_H273_IEC61966_2(E_p)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    oetf_inverse_H273_IEC61966_2(E_p * factor),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_H273_IEC61966_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_IEC61966_2` definition nan support.
        """

        oetf_inverse_H273_IEC61966_2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestEotf_inverse_H273_ST428_1(unittest.TestCase):
    """
        Define :func:`colour.models.rgb.transfer_functions.itut_h_273.
    eotf_inverse_H273_ST428_1` definition unit tests methods.
    """

    def test_eotf_inverse_H273_ST428_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_inverse_H273_ST428_1` definition.
        """

        np.testing.assert_allclose(
            eotf_inverse_H273_ST428_1(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            eotf_inverse_H273_ST428_1(0.18),
            0.500048337717236,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            eotf_inverse_H273_ST428_1(1.0),
            0.967042675317934,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_H273_ST428_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_inverse_H273_ST428_1` definition n-dimensional arrays support.
        """

        E = 0.18
        E_p = eotf_inverse_H273_ST428_1(E)

        E = np.tile(E, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_allclose(
            eotf_inverse_H273_ST428_1(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E = np.reshape(E, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_allclose(
            eotf_inverse_H273_ST428_1(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E = np.reshape(E, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_allclose(
            eotf_inverse_H273_ST428_1(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_eotf_inverse_H273_ST428_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_inverse_H273_ST428_1` definition domain and range scale support.
        """

        E = 0.18
        E_p = eotf_inverse_H273_ST428_1(E)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    eotf_inverse_H273_ST428_1(E * factor),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_H273_ST428_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_inverse_H273_ST428_1` definition nan support.
        """

        eotf_inverse_H273_ST428_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_H273_ST428_1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_H273_ST428_1` definition unit tests methods.
    """

    def test_eotf_H273_ST428_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_H273_ST428_1` definition.
        """

        np.testing.assert_allclose(
            eotf_H273_ST428_1(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            eotf_H273_ST428_1(0.500048337717236),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            eotf_H273_ST428_1(0.967042675317934),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_H273_ST428_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_H273_ST428_1` definition n-dimensional arrays support.
        """

        E_p = 0.500048337717236
        E = eotf_H273_ST428_1(E_p)

        E_p = np.tile(E_p, 6)
        E = np.tile(E, 6)
        np.testing.assert_allclose(
            eotf_H273_ST428_1(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E_p = np.reshape(E_p, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_allclose(
            eotf_H273_ST428_1(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E_p = np.reshape(E_p, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_allclose(
            eotf_H273_ST428_1(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_eotf_H273_ST428_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_H273_ST428_1` definition domain and range scale support.
        """

        E_p = 0.500048337717236
        E = eotf_H273_ST428_1(E_p)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    eotf_H273_ST428_1(E_p * factor),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_H273_ST428_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_H273_ST428_1` definition nan support.
        """

        eotf_H273_ST428_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == "__main__":
    unittest.main()
