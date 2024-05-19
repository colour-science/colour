# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.igpgtg` module."""

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import IgPgTg_to_XYZ, XYZ_to_IgPgTg
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_IgPgTg",
    "TestIgPgTg_to_XYZ",
]


class TestXYZ_to_IgPgTg:
    """
    Define :func:`colour.models.igpgtg.XYZ_to_IgPgTg` definition unit tests
    methods.
    """

    def test_XYZ_to_IgPgTg(self):
        """Test :func:`colour.models.igpgtg.XYZ_to_IgPgTg` definition."""

        np.testing.assert_allclose(
            XYZ_to_IgPgTg(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.42421258, 0.18632491, 0.10689223]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_IgPgTg(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.50912820, -0.14804331, 0.11921472]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_IgPgTg(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.29095152, -0.04057508, -0.18220795]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_IgPgTg(self):
        """
        Test :func:`colour.models.igpgtg.XYZ_to_IgPgTg` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        IgPgTg = XYZ_to_IgPgTg(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        IgPgTg = np.tile(IgPgTg, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_IgPgTg(XYZ), IgPgTg, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        IgPgTg = np.reshape(IgPgTg, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_IgPgTg(XYZ), IgPgTg, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_IgPgTg(self):
        """
        Test :func:`colour.models.igpgtg.XYZ_to_IgPgTg` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        IgPgTg = XYZ_to_IgPgTg(XYZ)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_IgPgTg(XYZ * factor),
                    IgPgTg * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_IgPgTg(self):
        """
        Test :func:`colour.models.igpgtg.XYZ_to_IgPgTg` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_IgPgTg(cases)


class TestIgPgTg_to_XYZ:
    """
    Define :func:`colour.models.igpgtg.IgPgTg_to_XYZ` definition unit tests
    methods.
    """

    def test_IgPgTg_to_XYZ(self):
        """Test :func:`colour.models.igpgtg.IgPgTg_to_XYZ` definition."""

        np.testing.assert_allclose(
            IgPgTg_to_XYZ(np.array([0.42421258, 0.18632491, 0.10689223])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            IgPgTg_to_XYZ(np.array([0.50912820, -0.14804331, 0.11921472])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            IgPgTg_to_XYZ(np.array([0.29095152, -0.04057508, -0.18220795])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_IgPgTg_to_XYZ(self):
        """
        Test :func:`colour.models.igpgtg.IgPgTg_to_XYZ` definition
        n-dimensional support.
        """

        IgPgTg = np.array([0.42421258, 0.18632491, 0.10689223])
        XYZ = IgPgTg_to_XYZ(IgPgTg)

        IgPgTg = np.tile(IgPgTg, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            IgPgTg_to_XYZ(IgPgTg), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        IgPgTg = np.reshape(IgPgTg, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            IgPgTg_to_XYZ(IgPgTg), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_IgPgTg_to_XYZ(self):
        """
        Test :func:`colour.models.igpgtg.IgPgTg_to_XYZ` definition domain and
        range scale support.
        """

        IgPgTg = np.array([0.42421258, 0.18632491, 0.10689223])
        XYZ = IgPgTg_to_XYZ(IgPgTg)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    IgPgTg_to_XYZ(IgPgTg * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_IgPgTg_to_XYZ(self):
        """
        Test :func:`colour.models.igpgtg.IgPgTg_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        IgPgTg_to_XYZ(cases)
