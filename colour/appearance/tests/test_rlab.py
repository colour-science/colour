"""Define the unit tests for the :mod:`colour.appearance.rlab` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.appearance import (
    D_FACTOR_RLAB,
    VIEWING_CONDITIONS_RLAB,
    XYZ_to_RLAB,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_float_array,
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
    "TestXYZ_to_RLAB",
]


class TestXYZ_to_RLAB:
    """
    Define :func:`colour.appearance.rlab.XYZ_to_RLAB` definition unit tests
    methods.
    """

    def test_XYZ_to_RLAB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.rlab.XYZ_to_RLAB` definition.

        Notes
        -----
        -   Reference values are taken from *Fairchild (2013)* Table 13.2 for
            the *RLAB* colour appearance model.
        """

        sigma = 0.4347

        # Case 1: D65 stimulus, D65 reference white, photopic luminance.
        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_n = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        Y_n = 318.31
        xp_assert_close(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma),
            [49.67, 0.01, 270, 0, np.nan, 0, -0.01],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        # Case 2: chromatic stimulus, D65 reference white, mesopic luminance.
        XYZ = xp_as_array([57.06, 43.06, 31.96], xp=xp)
        Y_n = 31.83
        xp_assert_close(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma),
            [69.33, 49.74, 21.3, 0.72, np.nan, 46.33, 18.09],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        # Case 3: green stimulus, illuminant A reference white, photopic.
        XYZ = xp_as_array([3.53, 6.56, 2.14], xp=xp)
        XYZ_n = xp_as_array([109.85, 100.00, 35.58], xp=xp)
        Y_n = 318.31
        xp_assert_close(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma),
            [30.78, 41.02, 176.9, 1.33, np.nan, -40.96, 2.25],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        # Case 4: D65 stimulus, illuminant A reference white, mesopic.
        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        Y_n = 31.83
        xp_assert_close(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma),
            [49.83, 54.87, 286.5, 1.1, np.nan, 15.57, -52.61],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

    def test_n_dimensional_XYZ_to_RLAB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.rlab.XYZ_to_RLAB` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_n = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        Y_n = 318.31
        sigma = 0.4347
        specification = XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma)

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        specification = np.tile(specification, (6, 1))
        xp_assert_close(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_n = xp.tile(xp_as_array(XYZ_n, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_n = xp_reshape(xp_as_array(XYZ_n, xp=xp), (2, 3, 3), xp=xp)
        specification = xp_reshape(xp_as_array(specification, xp=xp), (2, 3, 7), xp=xp)
        xp_assert_close(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_RLAB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.rlab.XYZ_to_RLAB` definition domain and
        range scale support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_n = xp_as_array([109.85, 100, 35.58], xp=xp)
        Y_n = 31.83
        sigma = VIEWING_CONDITIONS_RLAB["Average"]
        D = D_FACTOR_RLAB["Hard Copy Images"]
        specification = XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D)

        d_r = (
            ("reference", 1, 1),
            ("1", 0.01, np.array([1, 1, 1 / 360, 1, np.nan, 1, 1])),
            ("100", 1, np.array([1, 1, 100 / 360, 1, np.nan, 1, 1])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_RLAB(
                        XYZ * xp_as_array(factor_a, xp=xp),
                        XYZ_n * xp_as_array(factor_a, xp=xp),
                        Y_n,
                        sigma,
                        D,
                    ),
                    as_float_array(specification) * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_RLAB(self) -> None:
        """
        Test :func:`colour.appearance.rlab.XYZ_to_RLAB` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            XYZ = case
            XYZ_n = case
            Y_n = case[0]
            sigma = case[0]
            D = case[0]
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D)
