"""Define the unit tests for the :mod:`colour.models.prolab` module."""

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import ProLab_to_XYZ, XYZ_to_ProLab
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_ProLab",
    "TestProLab_to_XYZ",
]


class TestXYZ_to_ProLab:
    """
    Define :func:`colour.models.ProLab.TestXYZ_to_ProLab` definition unit
    tests methods.
    """

    def test_XYZ_to_ProLab(self):
        """Test :func:`colour.models.ProLab.XYZ_to_ProLab` definition."""

        np.testing.assert_allclose(
            XYZ_to_ProLab(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([48.7948929, 35.31503175, 13.30044932]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_ProLab(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([64.45929636, -21.67007419, 13.25749056]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_ProLab(np.array([0.96907232, 1.00000000, 0.12179215])),
            np.array([100.0, 5.47367608, 37.26313098]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_ProLab(self):
        """
        Test :func:`colour.models.prolab.XYZ_to_ProLab` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        ProLab = XYZ_to_ProLab(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        ProLab = np.tile(ProLab, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_ProLab(XYZ), ProLab, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        ProLab = np.reshape(ProLab, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_ProLab(XYZ), ProLab, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_ProLab(self):
        """
        Test :func:`colour.models.prolab.XYZ_to_ProLab` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        ProLab = XYZ_to_ProLab(XYZ)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_ProLab(XYZ * factor),
                    ProLab * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_ProLab(self):
        """
        Test :func:`colour.models.ProLab.XYZ_to_ProLab` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_ProLab(cases)


class TestProLab_to_XYZ:
    """
    Define :func:`colour.models.ProLab.ProLab_to_XYZ` definition unit tests
    methods.
    """

    def test_ProLab_to_XYZ(self):
        """Test :func:`colour.models.ProLab.ProLab_to_XYZ` definition."""

        np.testing.assert_allclose(
            ProLab_to_XYZ(np.array([48.7948929, 35.31503175, 13.30044932])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            ProLab_to_XYZ(np.array([64.45929636, -21.67007419, 13.25749056])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            ProLab_to_XYZ(np.array([100.0, 5.47367608, 37.26313098])),
            np.array([0.96907232, 1.00000000, 0.12179215]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_ProLab(self):
        """
        Test :func:`colour.models.prolab.XYZ_to_ProLab` definition
        n-dimensional support.
        """

        ProLab = np.array([48.7948929, 35.31503175, 13.30044932])
        XYZ = ProLab_to_XYZ(ProLab)

        ProLab = np.tile(ProLab, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            ProLab_to_XYZ(ProLab), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        ProLab = np.reshape(ProLab, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            ProLab_to_XYZ(ProLab), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_ProLab(self):
        """
        Test :func:`colour.models.prolab.XYZ_to_ProLab` definition domain and
        range scale support.
        """

        ProLab = np.array([48.7948929, 35.31503175, 13.30044932])
        XYZ = XYZ_to_ProLab(ProLab)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_ProLab(ProLab * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ProLab_to_XYZ(self):
        """
        Test :func:`colour.models.ProLab.ProLab_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        ProLab_to_XYZ(cases)
