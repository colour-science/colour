"""Define the unit tests for the :mod:`colour.models.ragoo2021` module."""

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import IPT_Ragoo2021_to_XYZ, XYZ_to_IPT_Ragoo2021
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_IPT_Ragoo2021",
    "TestIPT_Ragoo2021_to_XYZ",
]


class TestXYZ_to_IPT_Ragoo2021:
    """
    Define :func:`colour.models.ragoo2021.XYZ_to_IPT_Ragoo2021` definition
    unit tests methods.
    """

    def test_XYZ_to_IPT_Ragoo2021(self):
        """
        Test :func:`colour.models.ragoo2021.XYZ_to_IPT_Ragoo2021` definition.
        """

        np.testing.assert_allclose(
            XYZ_to_IPT_Ragoo2021(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.42248243, 0.29105140, 0.20410663]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_IPT_Ragoo2021(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.54745257, -0.22795249, 0.10109646]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_IPT_Ragoo2021(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.32151337, 0.06071424, -0.27388774]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_IPT_Ragoo2021(self):
        """
        Test :func:`colour.models.ragoo2021.XYZ_to_IPT_Ragoo2021` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        IPT = XYZ_to_IPT_Ragoo2021(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        IPT = np.tile(IPT, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_IPT_Ragoo2021(XYZ), IPT, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        IPT = np.reshape(IPT, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_IPT_Ragoo2021(XYZ), IPT, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_IPT_Ragoo2021(self):
        """
        Test :func:`colour.models.ragoo2021.XYZ_to_IPT_Ragoo2021` definition
        domain and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        IPT = XYZ_to_IPT_Ragoo2021(XYZ)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_IPT_Ragoo2021(XYZ * factor),
                    IPT * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_IPT_Ragoo2021(self):
        """
        Test :func:`colour.models.ragoo2021.XYZ_to_IPT_Ragoo2021` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_IPT_Ragoo2021(cases)


class TestIPT_Ragoo2021_to_XYZ:
    """
    Define :func:`colour.models.ragoo2021.IPT_Ragoo2021_to_XYZ` definition
    unit tests methods.
    """

    def test_IPT_Ragoo2021_to_XYZ(self):
        """
        Test :func:`colour.models.ragoo2021.IPT_Ragoo2021_to_XYZ` definition.
        """

        np.testing.assert_allclose(
            IPT_Ragoo2021_to_XYZ(np.array([0.42248243, 0.29105140, 0.20410663])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            IPT_Ragoo2021_to_XYZ(np.array([0.54745257, -0.22795249, 0.10109646])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            IPT_Ragoo2021_to_XYZ(np.array([0.32151337, 0.06071424, -0.27388774])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_IPT_Ragoo2021_to_XYZ(self):
        """
        Test :func:`colour.models.ragoo2021.IPT_Ragoo2021_to_XYZ` definition
        n-dimensional support.
        """

        IPT = np.array([0.42248243, 0.29105140, 0.20410663])
        XYZ = IPT_Ragoo2021_to_XYZ(IPT)

        IPT = np.tile(IPT, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            IPT_Ragoo2021_to_XYZ(IPT), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        IPT = np.reshape(IPT, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            IPT_Ragoo2021_to_XYZ(IPT), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_IPT_Ragoo2021_to_XYZ(self):
        """
        Test :func:`colour.models.ragoo2021.IPT_Ragoo2021_to_XYZ` definition
        domain and range scale support.
        """

        IPT = np.array([0.42248243, 0.29105140, 0.20410663])
        XYZ = IPT_Ragoo2021_to_XYZ(IPT)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    IPT_Ragoo2021_to_XYZ(IPT * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_IPT_Ragoo2021_to_XYZ(self):
        """
        Test :func:`colour.models.ragoo2021.IPT_Ragoo2021_to_XYZ` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        IPT_Ragoo2021_to_XYZ(cases)
