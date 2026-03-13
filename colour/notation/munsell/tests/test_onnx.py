"""
Define the unit tests for the :mod:`colour.notation.munsell.onnx` module.
"""

from __future__ import annotations

import contextlib
from itertools import product

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.notation.munsell import (
    munsell_colour_to_xyY_Onnx,
    munsell_specification_to_xyY_Onnx,
    xyY_to_munsell_colour_Onnx,
    xyY_to_munsell_specification_Onnx,
)
from colour.notation.munsell.tests.test_centore2014 import (
    MUNSELL_SPECIFICATIONS,
)
from colour.utilities import (
    as_float_array,
    domain_range_scale,
    ignore_numpy_errors,
    is_onnxruntime_installed,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestMunsellSpecification_to_xyY_Onnx",
    "TestMunsellColour_to_xyY_Onnx",
    "TestxyY_to_munsell_specification_Onnx",
    "TestxyY_to_munsell_colour_Onnx",
]


@pytest.mark.skipif(
    not is_onnxruntime_installed(), reason="onnxruntime is not installed"
)
class TestMunsellSpecification_to_xyY_Onnx:
    """
    Define
    :func:`colour.notation.munsell.munsell_specification_to_xyY_Onnx`
    definition unit tests methods.
    """

    def test_munsell_specification_to_xyY_Onnx(self) -> None:
        """
        Test
        :func:`colour.notation.munsell.munsell_specification_to_xyY_Onnx`
        definition.
        """

        specification, xyY = (
            as_float_array(list(MUNSELL_SPECIFICATIONS[..., 0])),
            as_float_array(list(MUNSELL_SPECIFICATIONS[..., 1])),
        )
        np.testing.assert_allclose(
            munsell_specification_to_xyY_Onnx(specification),
            xyY,
            atol=5e-2,
        )

    def test_n_dimensional_munsell_specification_to_xyY_Onnx(
        self,
    ) -> None:
        """
        Test
        :func:`colour.notation.munsell.munsell_specification_to_xyY_Onnx`
        definition n-dimensional arrays support.
        """

        specification = np.array([7.18927191, 5.34025196, 16.05861170, 3.00000000])
        xyY = munsell_specification_to_xyY_Onnx(specification)

        specification = np.tile(specification, (6, 1))
        xyY = np.tile(xyY, (6, 1))
        np.testing.assert_allclose(
            munsell_specification_to_xyY_Onnx(specification),
            xyY,
            atol=1e-6,
        )

        specification = np.reshape(specification, (2, 3, 4))
        xyY = np.reshape(xyY, (2, 3, 3))
        np.testing.assert_allclose(
            munsell_specification_to_xyY_Onnx(specification),
            xyY,
            atol=1e-6,
        )

        specification = np.array([np.nan, 8.9, np.nan, np.nan])
        xyY = munsell_specification_to_xyY_Onnx(specification)

        specification = np.tile(specification, (6, 1))
        xyY = np.tile(xyY, (6, 1))
        np.testing.assert_allclose(
            munsell_specification_to_xyY_Onnx(specification),
            xyY,
            atol=1e-6,
        )

        specification = np.reshape(specification, (2, 3, 4))
        xyY = np.reshape(xyY, (2, 3, 3))
        np.testing.assert_allclose(
            munsell_specification_to_xyY_Onnx(specification),
            xyY,
            atol=1e-6,
        )

    def test_domain_range_scale_munsell_specification_to_xyY_Onnx(
        self,
    ) -> None:
        """
        Test
        :func:`colour.notation.munsell.munsell_specification_to_xyY_Onnx`
        definition domain and range scale support.
        """

        specification = np.array([7.18927191, 5.34025196, 16.05861170, 3.00000000])
        xyY = munsell_specification_to_xyY_Onnx(specification)

        d_r = (
            ("reference", 1, 1),
            ("1", np.array([0.1, 0.1, 1 / 50, 0.1]), 1),
            ("100", np.array([10, 10, 2, 10]), np.array([1, 1, 100])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    munsell_specification_to_xyY_Onnx(specification * factor_a),
                    xyY * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_munsell_specification_to_xyY_Onnx(self) -> None:
        """
        Test
        :func:`colour.notation.munsell.munsell_specification_to_xyY_Onnx`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=4))))
        for case in cases:
            with contextlib.suppress(AssertionError, TypeError, ValueError):
                munsell_specification_to_xyY_Onnx(case)


@pytest.mark.skipif(
    not is_onnxruntime_installed(), reason="onnxruntime is not installed"
)
class TestMunsellColour_to_xyY_Onnx:
    """
    Define :func:`colour.notation.munsell.munsell_colour_to_xyY_Onnx`
    definition unit tests methods.
    """

    def test_domain_range_scale_munsell_colour_to_xyY_Onnx(
        self,
    ) -> None:
        """
        Test
        :func:`colour.notation.munsell.munsell_colour_to_xyY_Onnx`
        definition domain and range scale support.
        """

        munsell_colour = "4.2YR 8.1/5.3"
        xyY = munsell_colour_to_xyY_Onnx(munsell_colour)

        d_r = (
            ("reference", 1),
            ("1", 1),
            ("100", np.array([1, 1, 100])),
        )
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    munsell_colour_to_xyY_Onnx(munsell_colour),
                    xyY * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    def test_n_dimensional_munsell_colour_to_xyY_Onnx(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_colour_to_xyY_Onnx`
        definition n-dimensional arrays support.
        """

        munsell_colour = "4.2YR 8.1/5.3"
        xyY = munsell_colour_to_xyY_Onnx(munsell_colour)

        munsell_colour = np.tile(munsell_colour, 6)
        xyY = np.tile(xyY, (6, 1))
        np.testing.assert_allclose(
            munsell_colour_to_xyY_Onnx(munsell_colour),
            xyY,
            atol=1e-6,
        )

        munsell_colour = np.reshape(munsell_colour, (2, 3))
        xyY = np.reshape(xyY, (2, 3, 3))
        np.testing.assert_allclose(
            munsell_colour_to_xyY_Onnx(munsell_colour),
            xyY,
            atol=1e-6,
        )

        munsell_colour = "N8.9"
        xyY = munsell_colour_to_xyY_Onnx(munsell_colour)

        munsell_colour = np.tile(munsell_colour, 6)
        xyY = np.tile(xyY, (6, 1))
        np.testing.assert_allclose(
            munsell_colour_to_xyY_Onnx(munsell_colour),
            xyY,
            atol=1e-6,
        )

        munsell_colour = np.reshape(munsell_colour, (2, 3))
        xyY = np.reshape(xyY, (2, 3, 3))
        np.testing.assert_allclose(
            munsell_colour_to_xyY_Onnx(munsell_colour),
            xyY,
            atol=1e-6,
        )


@pytest.mark.skipif(
    not is_onnxruntime_installed(), reason="onnxruntime is not installed"
)
class TestxyY_to_munsell_specification_Onnx:
    """
    Define
    :func:`colour.notation.munsell.xyY_to_munsell_specification_Onnx`
    definition unit tests methods.
    """

    def test_xyY_to_munsell_specification_Onnx(self) -> None:
        """
        Test
        :func:`colour.notation.munsell.xyY_to_munsell_specification_Onnx`
        definition.
        """

        specification, xyY = (
            as_float_array(list(MUNSELL_SPECIFICATIONS[..., 0])),
            as_float_array(list(MUNSELL_SPECIFICATIONS[..., 1])),
        )

        np.testing.assert_allclose(
            xyY_to_munsell_specification_Onnx(xyY),
            specification,
            atol=1,
        )

    def test_n_dimensional_xyY_to_munsell_specification_Onnx(
        self,
    ) -> None:
        """
        Test
        :func:`colour.notation.munsell.xyY_to_munsell_specification_Onnx`
        definition n-dimensional arrays support.
        """

        xyY = np.array([0.16623068, 0.45684550, 0.22399519])
        specification = xyY_to_munsell_specification_Onnx(xyY)

        xyY = np.tile(xyY, (6, 1))
        specification = np.tile(specification, (6, 1))
        np.testing.assert_allclose(
            xyY_to_munsell_specification_Onnx(xyY),
            specification,
            atol=1e-6,
        )

        xyY = np.reshape(xyY, (2, 3, 3))
        specification = np.reshape(specification, (2, 3, 4))
        np.testing.assert_allclose(
            xyY_to_munsell_specification_Onnx(xyY),
            specification,
            atol=1e-6,
        )

    def test_domain_range_scale_xyY_to_munsell_specification_Onnx(
        self,
    ) -> None:
        """
        Test
        :func:`colour.notation.munsell.xyY_to_munsell_specification_Onnx`
        definition domain and range scale support.
        """

        xyY = [0.16623068, 0.45684550, 0.22399519]
        specification = xyY_to_munsell_specification_Onnx(xyY)

        d_r = (
            ("reference", 1, 1),
            ("1", 1, np.array([0.1, 0.1, 1 / 50, 0.1])),
            ("100", np.array([1, 1, 100]), np.array([10, 10, 2, 10])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    xyY_to_munsell_specification_Onnx(xyY * factor_a),
                    specification * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_xyY_to_munsell_specification_Onnx(self) -> None:
        """
        Test
        :func:`colour.notation.munsell.xyY_to_munsell_specification_Onnx`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            with contextlib.suppress(AssertionError, TypeError, ValueError):
                xyY_to_munsell_specification_Onnx(case)


@pytest.mark.skipif(
    not is_onnxruntime_installed(), reason="onnxruntime is not installed"
)
class TestxyY_to_munsell_colour_Onnx:
    """
    Define :func:`colour.notation.munsell.xyY_to_munsell_colour_Onnx`
    definition unit tests methods.
    """

    def test_domain_range_scale_xyY_to_munsell_colour_Onnx(
        self,
    ) -> None:
        """
        Test
        :func:`colour.notation.munsell.xyY_to_munsell_colour_Onnx`
        definition domain and range scale support.
        """

        xyY = np.array([0.38736945, 0.35751656, 0.59362000])
        munsell_colour = xyY_to_munsell_colour_Onnx(xyY)

        d_r = (
            ("reference", 1),
            ("1", 1),
            ("100", np.array([1, 1, 100])),
        )
        for scale, factor in d_r:
            with domain_range_scale(scale):
                assert xyY_to_munsell_colour_Onnx(xyY * factor) == munsell_colour

    def test_n_dimensional_xyY_to_munsell_colour_Onnx(self) -> None:
        """
        Test :func:`colour.notation.munsell.xyY_to_munsell_colour_Onnx`
        definition n-dimensional arrays support.
        """

        xyY = [0.16623068, 0.45684550, 0.22399519]
        munsell_colour = xyY_to_munsell_colour_Onnx(xyY)

        xyY = np.tile(xyY, (6, 1))
        munsell_colour = np.tile(munsell_colour, 6)
        np.testing.assert_equal(xyY_to_munsell_colour_Onnx(xyY), munsell_colour)

        xyY = np.reshape(xyY, (2, 3, 3))
        munsell_colour = np.reshape(munsell_colour, (2, 3))
        np.testing.assert_equal(xyY_to_munsell_colour_Onnx(xyY), munsell_colour)

        xyY = [0.38736945, 0.35751656, 0.59362000]
        munsell_colour = xyY_to_munsell_colour_Onnx(xyY)

        xyY = np.tile(xyY, (6, 1))
        munsell_colour = np.tile(munsell_colour, 6)
        np.testing.assert_equal(xyY_to_munsell_colour_Onnx(xyY), munsell_colour)

        xyY = np.reshape(xyY, (2, 3, 3))
        munsell_colour = np.reshape(munsell_colour, (2, 3))
        np.testing.assert_equal(xyY_to_munsell_colour_Onnx(xyY), munsell_colour)
