"""Define the unit tests for the :mod:`colour.appearance.nayatani95` module."""

from __future__ import annotations

import typing
from contextlib import nullcontext

if typing.TYPE_CHECKING:
    import pytest

    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.appearance import XYZ_to_Nayatani95
from colour.appearance.nayatani95 import hue_quadrature
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    array_namespace,
    as_float_array,
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
    "TestXYZ_to_Nayatani95",
    "TestHue_quadrature",
]


class TestXYZ_to_Nayatani95:
    """
    Define :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95` definition
    unit tests methods.
    """

    def test_XYZ_to_Nayatani95(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95`
        definition.

        Notes
        -----
        -   The test values have been generated from data of the following file
            by *Fairchild (2013)*:
            http://rit-mcsl.org/fairchild//files/AppModEx.xls
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_n = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        Y_o = 20
        E_o = 5000
        E_or = 1000
        # Near-achromatic opponent coordinates amplify float32 hue rounding.
        specification = as_ndarray(
            XYZ_to_Nayatani95(
                XYZ,
                XYZ_n,
                Y_o,
                E_o,
                E_or,
                compute_H=True,
            )
        )
        expected = np.array(
            [
                50,
                0.01,
                257.5,
                0.01,
                62.6,
                0.02,
                317.8,
                np.nan,
                50,
            ]
        )
        xp_assert_close(
            specification[[0, 1, 3, 4, 5, 7, 8]],
            expected[[0, 1, 3, 4, 5, 7, 8]],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05),
        )
        hue_tolerance = 0.15 if as_ndarray(XYZ).dtype.itemsize == 4 else 0
        xp_assert_close(
            specification[[2, 6]],
            expected[[2, 6]],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05, hue_tolerance),
        )

        XYZ = xp_as_array([57.06, 43.06, 31.96], xp=xp)
        E_o = 500
        xp_assert_close(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or, compute_H=True),
            [73, 48.3, 21.6, 37.1, 67.3, 42.9, 2.1, np.nan, 75.9],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05),
        )

        XYZ = xp_as_array([3.53, 6.56, 2.14], xp=xp)
        XYZ_n = xp_as_array([109.85, 100.00, 35.58], xp=xp)
        E_o = 5000
        xp_assert_close(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or, compute_H=True),
            [24.5, 49.3, 190.6, 81.3, 37.5, 62.1, 239.4, np.nan, 29.7],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05),
        )

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        E_o = 500
        xp_assert_close(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or, compute_H=True),
            [49.4, 39.9, 236.3, 40.2, 44.2, 35.8, 303.6, np.nan, 49.4],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05),
        )

        xp_assert_close(
            XYZ_to_Nayatani95(
                xp_as_array([19.01, 20.0, 21.78], xp=xp),
                [95.05, 100.0, 108.88],
                20,
                1000,
                1000,
            ),
            XYZ_to_Nayatani95(
                [19.01, 20.0, 21.78], [95.05, 100.0, 108.88], 20, 1000, 1000
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Nayatani95(
        self, xp: ModuleType, request: pytest.FixtureRequest
    ) -> None:
        """
        Test :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_n = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        Y_o = 20
        E_o = 5000
        E_or = 1000
        specification = XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or, compute_H=True)

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        specification = np.tile(specification, (6, 1))
        xp_assert_close(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_n = xp.tile(xp_as_array(XYZ_n, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_n = xp_reshape(xp_as_array(XYZ_n, xp=xp), (2, 3, 3), xp=xp)
        specification = xp_reshape(xp_as_array(specification, xp=xp), (2, 3, 9), xp=xp)
        xp_assert_close(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Batched reference illuminance expands otherwise scalar responses.
        host_arguments: dict[str, typing.Any] = {
            "XYZ": [57.06, 43.06, 31.96],
            "XYZ_n": [95.05, 100.0, 108.88],
            "Y_o": 20.0,
            "E_o": 1000.0,
            "E_or": np.array([1000.0, 2000.0]),
            "n": 1.0,
        }
        expected = XYZ_to_Nayatani95(**host_arguments)
        for parameter, argument in host_arguments.items():
            native = xp_as_array(argument, xp=xp)
            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                actual = XYZ_to_Nayatani95(**{**host_arguments, parameter: native})
            for correlate in ("L_star_P", "L_star_N", "Q", "C", "M"):
                value = getattr(actual, correlate)
                assert array_namespace(value) is array_namespace(native)
                assert getattr(value, "device", None) == getattr(native, "device", None)
                xp_assert_close(
                    value, getattr(expected, correlate), atol=TOLERANCE_ABSOLUTE_TESTS
                )
            if xp.__name__ != "numpy":
                request.getfixturevalue("autodiff")(
                    lambda value, parameter=parameter: XYZ_to_Nayatani95(
                        **{**host_arguments, parameter: value}
                    ).L_star_N,
                    native,
                )

    def test_domain_range_scale_XYZ_to_Nayatani95(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95` definition
        domain and range scale support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_n = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        Y_o = 20.0
        E_o = 5000.0
        E_or = 1000.0
        specification = XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or, compute_H=True)

        d_r = (
            ("reference", 1, 1),
            ("1", 0.01, np.array([1, 1, 1 / 360, 1, 1, 1, 1 / 400, np.nan, 1])),
            (
                "100",
                1,
                np.array([1, 1, 100 / 360, 1, 1, 1, 100 / 400, np.nan, 1]),
            ),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_Nayatani95(
                        XYZ * xp_as_array(factor_a, xp=xp),
                        XYZ_n * xp_as_array(factor_a, xp=xp),
                        Y_o,
                        E_o,
                        E_or,
                        compute_H=True,
                    ),
                    as_float_array(specification) * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Nayatani95(self) -> None:
        """
        Test :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Nayatani95(cases, cases, cases[..., 0], cases[..., 0], cases[..., 0])


class TestHue_quadrature:
    """
    Define :func:`colour.appearance.nayatani95.hue_quadrature` definition unit
    tests methods.
    """

    def test_hue_quadrature(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.nayatani95.hue_quadrature` definition.
        """

        # Interior of the ``H_1`` bin (Y -> G, 90.00..164.25): the bin midpoint
        # maps to exactly ``150``.
        xp_assert_close(
            hue_quadrature(xp_as_array(127.125, xp=xp)),
            xp_as_array(150.0, xp=xp),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # ``h < 20.14`` wraps through ``360`` into the B -> R interval via the
        # ``default`` arm of the selection.
        xp_assert_close(
            hue_quadrature(xp_as_array(10.0, xp=xp)),
            xp_as_array(393.2010191766126, xp=xp),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_compute_H_False_XYZ_to_Nayatani95(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95` ``H`` output
        defaults to ``NaN`` when ``compute_H`` is ``False``.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_n = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        specification = XYZ_to_Nayatani95(XYZ, XYZ_n, 20, 5000, 1000)

        H = specification.H
        assert H is not None
        assert xp.all(xp.isnan(xp_as_array(H, xp=xp)))
