"""
Define the unit tests for the :mod:`colour.notation.munsell` module.
"""

from __future__ import annotations

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.notation.munsell import (
    munsell_colour_to_xyY,
    munsell_colour_to_xyY_Centore2014,
    munsell_specification_to_xyY,
    munsell_specification_to_xyY_Centore2014,
    xyY_to_munsell_colour,
    xyY_to_munsell_colour_Centore2014,
    xyY_to_munsell_specification,
    xyY_to_munsell_specification_Centore2014,
)
from colour.utilities import (
    is_onnxruntime_installed,
    xp_assert_close,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestMunsellSpecification_to_xyY",
    "TestMunsellColour_to_xyY",
    "TestxyY_to_munsell_specification",
    "TestxyY_to_munsell_colour",
]


class TestMunsellSpecification_to_xyY:
    """
    Define :func:`colour.notation.munsell.munsell_specification_to_xyY`
    definition unit tests methods.
    """

    def test_munsell_specification_to_xyY(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_specification_to_xyY`
        definition.
        """

        specification = np.array([7.18927191, 5.34025196, 16.05861170, 3.00000000])

        xp_assert_close(
            munsell_specification_to_xyY(specification),
            munsell_specification_to_xyY_Centore2014(specification),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_munsell_specification_to_xyY_Centore2014(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_specification_to_xyY`
        definition with the *Centore 2014* method.
        """

        specification = np.array([7.18927191, 5.34025196, 16.05861170, 3.00000000])

        xp_assert_close(
            munsell_specification_to_xyY(specification, method="Centore 2014"),
            munsell_specification_to_xyY_Centore2014(specification),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @pytest.mark.skipif(
        not is_onnxruntime_installed(),
        reason="onnxruntime is not installed",
    )
    def test_munsell_specification_to_xyY_Onnx(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_specification_to_xyY`
        definition with the *ONNX* method.
        """

        from colour.notation.munsell import (  # noqa: PLC0415
            munsell_specification_to_xyY_Onnx,
        )

        specification = np.array([7.18927191, 5.34025196, 16.05861170, 3.00000000])

        xp_assert_close(
            munsell_specification_to_xyY(specification, method="ONNX"),
            munsell_specification_to_xyY_Onnx(specification),
            atol=TOLERANCE_ABSOLUTE_TESTS * 10,
        )

    def test_munsell_specification_to_xyY_raise_exception(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_specification_to_xyY`
        definition raised exception.
        """

        with pytest.raises(ValueError):
            munsell_specification_to_xyY(
                np.array([7.18927191, 5.34025196, 16.0586117, 3.0]), method="Invalid"
            )


class TestMunsellColour_to_xyY:
    """
    Define :func:`colour.notation.munsell.munsell_colour_to_xyY`
    definition unit tests methods.
    """

    def test_munsell_colour_to_xyY(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_colour_to_xyY`
        definition.
        """

        xp_assert_close(
            munsell_colour_to_xyY("4.2YR 8.1/5.3"),
            munsell_colour_to_xyY_Centore2014("4.2YR 8.1/5.3"),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_munsell_colour_to_xyY_Centore2014(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_colour_to_xyY`
        definition with the *Centore 2014* method.
        """

        xp_assert_close(
            munsell_colour_to_xyY("4.2YR 8.1/5.3", method="Centore 2014"),
            munsell_colour_to_xyY_Centore2014("4.2YR 8.1/5.3"),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @pytest.mark.skipif(
        not is_onnxruntime_installed(),
        reason="onnxruntime is not installed",
    )
    def test_munsell_colour_to_xyY_Onnx(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_colour_to_xyY`
        definition with the *ONNX* method.
        """

        from colour.notation.munsell import (  # noqa: PLC0415
            munsell_colour_to_xyY_Onnx,
        )

        xp_assert_close(
            munsell_colour_to_xyY("4.2YR 8.1/5.3", method="ONNX"),
            munsell_colour_to_xyY_Onnx("4.2YR 8.1/5.3"),
            atol=TOLERANCE_ABSOLUTE_TESTS * 10,
        )

    def test_munsell_colour_to_xyY_raise_exception(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_colour_to_xyY`
        definition raised exception.
        """

        with pytest.raises(ValueError):
            munsell_colour_to_xyY("4.2YR 8.1/5.3", method="Invalid")


class TestxyY_to_munsell_specification:
    """
    Define :func:`colour.notation.munsell.xyY_to_munsell_specification`
    definition unit tests methods.
    """

    def test_xyY_to_munsell_specification(self) -> None:
        """
        Test :func:`colour.notation.munsell.xyY_to_munsell_specification`
        definition.
        """

        xyY = np.array([0.16623068, 0.45684550, 0.22399519])

        xp_assert_close(
            xyY_to_munsell_specification(xyY),
            xyY_to_munsell_specification_Centore2014(xyY),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_xyY_to_munsell_specification_Centore2014(self) -> None:
        """
        Test :func:`colour.notation.munsell.xyY_to_munsell_specification`
        definition with the *Centore 2014* method.
        """

        xyY = np.array([0.16623068, 0.45684550, 0.22399519])

        xp_assert_close(
            xyY_to_munsell_specification(xyY, method="Centore 2014"),
            xyY_to_munsell_specification_Centore2014(xyY),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @pytest.mark.skipif(
        not is_onnxruntime_installed(),
        reason="onnxruntime is not installed",
    )
    def test_xyY_to_munsell_specification_Onnx(self) -> None:
        """
        Test :func:`colour.notation.munsell.xyY_to_munsell_specification`
        definition with the *ONNX* method.
        """

        from colour.notation.munsell import (  # noqa: PLC0415
            xyY_to_munsell_specification_Onnx,
        )

        xyY = np.array([0.16623068, 0.45684550, 0.22399519])

        xp_assert_close(
            xyY_to_munsell_specification(xyY, method="ONNX"),
            xyY_to_munsell_specification_Onnx(xyY),
            atol=TOLERANCE_ABSOLUTE_TESTS * 10,
        )

    def test_xyY_to_munsell_specification_raise_exception(self) -> None:
        """
        Test :func:`colour.notation.munsell.xyY_to_munsell_specification`
        definition raised exception.
        """

        with pytest.raises(ValueError):
            xyY_to_munsell_specification(
                np.array([0.16623068, 0.4568455, 0.22399519]), method="Invalid"
            )


class TestxyY_to_munsell_colour:
    """
    Define :func:`colour.notation.munsell.xyY_to_munsell_colour`
    definition unit tests methods.
    """

    def test_xyY_to_munsell_colour(self) -> None:
        """
        Test :func:`colour.notation.munsell.xyY_to_munsell_colour`
        definition.
        """

        xyY = np.array([0.38736945, 0.35751656, 0.59362000])

        assert xyY_to_munsell_colour(xyY) == xyY_to_munsell_colour_Centore2014(xyY)

    def test_xyY_to_munsell_colour_Centore2014(self) -> None:
        """
        Test :func:`colour.notation.munsell.xyY_to_munsell_colour`
        definition with the *Centore 2014* method.
        """

        xyY = np.array([0.38736945, 0.35751656, 0.59362000])

        assert xyY_to_munsell_colour(
            xyY, method="Centore 2014"
        ) == xyY_to_munsell_colour_Centore2014(xyY)

    @pytest.mark.skipif(
        not is_onnxruntime_installed(),
        reason="onnxruntime is not installed",
    )
    def test_xyY_to_munsell_colour_Onnx(self) -> None:
        """
        Test :func:`colour.notation.munsell.xyY_to_munsell_colour`
        definition with the *ONNX* method.
        """

        from colour.notation.munsell import (  # noqa: PLC0415
            xyY_to_munsell_colour_Onnx,
        )

        xyY = np.array([0.38736945, 0.35751656, 0.59362000])

        assert xyY_to_munsell_colour(xyY, method="ONNX") == xyY_to_munsell_colour_Onnx(
            xyY
        )

    def test_xyY_to_munsell_colour_raise_exception(self) -> None:
        """
        Test :func:`colour.notation.munsell.xyY_to_munsell_colour`
        definition raised exception.
        """

        with pytest.raises(ValueError):
            xyY_to_munsell_colour(
                np.array([0.38736945, 0.35751656, 0.59362]), method="Invalid"
            )
