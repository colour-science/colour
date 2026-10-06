"""Define the unit tests for the :mod:`colour.models.hdr_cie_lab` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import XYZ_to_hdr_CIELab, hdr_CIELab_to_XYZ
from colour.models.hdr_cie_lab import exponent_hdr_CIELab
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestExponent_hdr_CIELab",
    "TestXYZ_to_hdr_CIELab",
    "TestHdr_CIELab_to_XYZ",
]


class TestExponent_hdr_CIELab:
    """
    Define :func:`colour.models.hdr_cie_lab.exponent_hdr_CIELab`
    definition unit tests methods.
    """

    def test_exponent_hdr_CIELab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hdr_cie_lab.exponent_hdr_CIELab`
        definition.
        """

        xp_assert_close(
            exponent_hdr_CIELab(xp_as_array([0.2], xp=xp), xp_as_array([100], xp=xp)),
            [0.473851073746817],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_hdr_CIELab(xp_as_array([0.4], xp=xp), xp_as_array([100], xp=xp)),
            [0.656101486726362],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_hdr_CIELab(
                xp_as_array([0.4], xp=xp),
                xp_as_array([100], xp=xp),
                method="Fairchild 2010",
            ),
            [1.326014370643925],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_hdr_CIELab(xp_as_array([0.2], xp=xp), xp_as_array([1000], xp=xp)),
            [0.710776610620225],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_exponent_hdr_CIELab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hdr_cie_lab.exponent_hdr_CIELab`
        definition n-dimensional arrays support.
        """

        Y_s = 0.2
        Y_abs = 100
        epsilon = as_ndarray(exponent_hdr_CIELab(Y_s, Y_abs))

        Y_s = xp.tile(xp_as_array(Y_s, xp=xp), (6,))
        Y_abs = xp.tile(xp_as_array(Y_abs, xp=xp), (6,))
        epsilon = xp.tile(xp_as_array(epsilon, xp=xp), (6,))
        xp_assert_close(
            exponent_hdr_CIELab(Y_s, Y_abs),
            epsilon,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Y_s = xp_reshape(xp_as_array(Y_s, xp=xp), (2, 3), xp=xp)
        Y_abs = xp_reshape(xp_as_array(Y_abs, xp=xp), (2, 3), xp=xp)
        epsilon = xp_reshape(xp_as_array(epsilon, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            exponent_hdr_CIELab(Y_s, Y_abs),
            epsilon,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Y_s = xp_reshape(xp_as_array(Y_s, xp=xp), (2, 3, 1), xp=xp)
        Y_abs = xp_reshape(xp_as_array(Y_abs, xp=xp), (2, 3, 1), xp=xp)
        epsilon = xp_reshape(xp_as_array(epsilon, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            exponent_hdr_CIELab(Y_s, Y_abs),
            epsilon,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_exponent_hdr_CIELab(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.models.hdr_cie_lab.exponent_hdr_CIELab` definition
        domain and range scale support.
        """

        Y_s = 0.2
        Y_abs = 100
        epsilon = as_ndarray(exponent_hdr_CIELab(Y_s, Y_abs))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    exponent_hdr_CIELab(Y_s * factor, Y_abs),
                    epsilon,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_exponent_hdr_CIELab(self) -> None:
        """
        Test :func:`colour.models.hdr_cie_lab.exponent_hdr_CIELab`
        definition nan support.
        """

        cases = np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        exponent_hdr_CIELab(cases, cases)


class TestXYZ_to_hdr_CIELab:
    """
    Define :func:`colour.models.hdr_cie_lab.XYZ_to_hdr_CIELab` definition unit
    tests methods.
    """

    def test_XYZ_to_hdr_CIELab(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.hdr_cie_lab.XYZ_to_hdr_CIELab` definition."""

        xp_assert_close(
            XYZ_to_hdr_CIELab(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            [51.87002062, 60.47633850, 32.14551912],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_hdr_CIELab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_as_array([0.44757, 0.40745], xp=xp),
            ),
            [51.87002062, 44.49667330, -6.69619196],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_hdr_CIELab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_as_array([0.44757, 0.40745], xp=xp),
                method="Fairchild 2010",
            ),
            [31.99621114, 95.08564341, -14.14047055],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_hdr_CIELab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp), Y_s=0.5
            ),
            [23.10388654, 59.31425004, 23.69960142],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_hdr_CIELab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp), Y_abs=1000
            ),
            [29.77261805, 62.58315675, 27.31232673],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_hdr_CIELab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hdr_cie_lab.XYZ_to_hdr_CIELab` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        Y_s = 0.2
        Y_abs = 100
        Lab_hdr = as_ndarray(XYZ_to_hdr_CIELab(XYZ, illuminant, Y_s, Y_abs))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        Lab_hdr = xp.tile(xp_as_array(Lab_hdr, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_hdr_CIELab(XYZ, illuminant, Y_s, Y_abs),
            Lab_hdr,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        illuminant = xp.tile(xp_as_array(illuminant, xp=xp), (6, 1))
        Y_s = xp.tile(xp_as_array(Y_s, xp=xp), (6,))
        Y_abs = xp.tile(xp_as_array(Y_abs, xp=xp), (6,))
        xp_assert_close(
            XYZ_to_hdr_CIELab(XYZ, illuminant, Y_s, Y_abs),
            Lab_hdr,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        illuminant = xp_reshape(xp_as_array(illuminant, xp=xp), (2, 3, 2), xp=xp)
        Y_s = xp_reshape(xp_as_array(Y_s, xp=xp), (2, 3), xp=xp)
        Y_abs = xp_reshape(xp_as_array(Y_abs, xp=xp), (2, 3), xp=xp)
        Lab_hdr = xp_reshape(xp_as_array(Lab_hdr, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            XYZ_to_hdr_CIELab(XYZ, illuminant, Y_s, Y_abs),
            Lab_hdr,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_hdr_CIELab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hdr_cie_lab.XYZ_to_hdr_CIELab` definition
        domain and range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        Y_s = 0.2
        Y_abs = 100
        Lab_hdr = as_ndarray(XYZ_to_hdr_CIELab(XYZ, illuminant, Y_s, Y_abs))

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_hdr_CIELab(
                        XYZ * xp_as_array(factor_a, xp=xp),
                        illuminant,
                        Y_s * xp_as_array(factor_a, xp=xp),
                        Y_abs,
                    ),
                    Lab_hdr * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_hdr_CIELab(self) -> None:
        """
        Test :func:`colour.models.hdr_cie_lab.XYZ_to_hdr_CIELab` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_hdr_CIELab(cases, cases[..., 0:2], cases[..., 0], cases[..., 0])


class TestHdr_CIELab_to_XYZ:
    """
    Define :func:`colour.models.hdr_cie_lab.hdr_CIELab_to_XYZ` definition unit
    tests methods.
    """

    def test_hdr_CIELab_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.hdr_cie_lab.hdr_CIELab_to_XYZ` definition."""

        xp_assert_close(
            hdr_CIELab_to_XYZ(
                xp_as_array([51.87002062, 60.47633850, 32.14551912], xp=xp)
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            hdr_CIELab_to_XYZ(
                xp_as_array([51.87002062, 44.49667330, -6.69619196], xp=xp),
                xp_as_array([0.44757, 0.40745], xp=xp),
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            hdr_CIELab_to_XYZ(
                xp_as_array([31.99621114, 95.08564341, -14.14047055], xp=xp),
                xp_as_array([0.44757, 0.40745], xp=xp),
                method="Fairchild 2010",
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            hdr_CIELab_to_XYZ(
                xp_as_array([23.10388654, 59.31425004, 23.69960142], xp=xp), Y_s=0.5
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            hdr_CIELab_to_XYZ(
                xp_as_array([29.77261805, 62.58315675, 27.31232673], xp=xp), Y_abs=1000
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_hdr_CIELab_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hdr_cie_lab.hdr_CIELab_to_XYZ` definition
        n-dimensional support.
        """

        Lab_hdr = xp_as_array([51.87002062, 60.47633850, 32.14551912], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        Y_s = 0.2
        Y_abs = 100
        XYZ = as_ndarray(hdr_CIELab_to_XYZ(Lab_hdr, illuminant, Y_s, Y_abs))

        Lab_hdr = xp.tile(xp_as_array(Lab_hdr, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            hdr_CIELab_to_XYZ(Lab_hdr, illuminant, Y_s, Y_abs),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        illuminant = xp.tile(xp_as_array(illuminant, xp=xp), (6, 1))
        Y_s = xp.tile(xp_as_array(Y_s, xp=xp), (6,))
        Y_abs = xp.tile(xp_as_array(Y_abs, xp=xp), (6,))
        xp_assert_close(
            hdr_CIELab_to_XYZ(Lab_hdr, illuminant, Y_s, Y_abs),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Lab_hdr = xp_reshape(xp_as_array(Lab_hdr, xp=xp), (2, 3, 3), xp=xp)
        illuminant = xp_reshape(xp_as_array(illuminant, xp=xp), (2, 3, 2), xp=xp)
        Y_s = xp_reshape(xp_as_array(Y_s, xp=xp), (2, 3), xp=xp)
        Y_abs = xp_reshape(xp_as_array(Y_abs, xp=xp), (2, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            hdr_CIELab_to_XYZ(Lab_hdr, illuminant, Y_s, Y_abs),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_hdr_CIELab_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hdr_cie_lab.hdr_CIELab_to_XYZ` definition
        domain and range scale support.
        """

        Lab_hdr = xp_as_array([26.46461067, -24.61332600, -4.84796811], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        Y_s = 0.2
        Y_abs = 100
        XYZ = as_ndarray(hdr_CIELab_to_XYZ(Lab_hdr, illuminant, Y_s, Y_abs))

        d_r = (("reference", 1, 1, 1), ("1", 0.01, 1, 1), ("100", 1, 100, 100))
        for scale, factor_a, factor_b, factor_c in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    hdr_CIELab_to_XYZ(
                        Lab_hdr * xp_as_array(factor_a, xp=xp),
                        illuminant,
                        Y_s * factor_b,
                        Y_abs,
                    ),
                    XYZ * factor_c,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_hdr_CIELab_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.hdr_cie_lab.hdr_CIELab_to_XYZ` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        hdr_CIELab_to_XYZ(cases, cases[..., 0:2], cases[..., 0], cases[..., 0])
