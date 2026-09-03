"""Define the unit tests for the :mod:`colour.temperature` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType


import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import CCT_to_uv, CCT_to_xy, uv_to_CCT, xy_to_CCT
from colour.utilities import (
    ColourUsageWarning,
    array_api_enable,
    xp_as_array,
    xp_assert_close,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestUv_to_CCT",
    "TestCCT_to_uv",
    "TestXy_to_CCT",
    "TestCCT_to_xy",
]


class TestUv_to_CCT:
    """Define :func:`colour.temperature.uv_to_CCT` definition unit tests."""

    def test_locus_methods_uv_to_CCT(self, xp: ModuleType) -> None:
        """Test locus-only methods return correlated colour temperature."""

        for method, uv, expected in (
            ("Krystek 1985", [0.20047203, 0.31029290], 6504.389416),
            ("Planck 1900", [0.20042808, 0.31033343], 6504.000071),
        ):
            CCT = uv_to_CCT(xp_as_array(uv, xp=xp), method=method)

            assert CCT.shape == ()
            xp_assert_close(
                CCT,
                expected,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )


class TestCCT_to_uv:
    """Define :func:`colour.temperature.CCT_to_uv` definition unit tests."""

    def test_locus_methods_CCT_to_uv(self, xp: ModuleType) -> None:
        """Test locus-only methods interpret inputs as temperatures."""

        CCT = xp_as_array([4000.0, 7000.0], xp=xp)

        for method, expected in (
            (
                "Krystek 1985",
                [
                    [0.225149641157266, 0.334340395957838],
                    [0.198152565091092, 0.307023596915037],
                ],
            ),
            (
                "Planck 1900",
                [
                    [0.225109670227493, 0.334387366663923],
                    [0.198126929048352, 0.307025980523306],
                ],
            ),
        ):
            uv = CCT_to_uv(CCT, method=method)

            assert uv.shape == (2, 2)
            xp_assert_close(uv, expected, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_CCT_to_uv_cuda_device(self) -> None:
        """Test locus-only methods retain an explicitly selected CUDA device."""

        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable.")

        CCT = torch.tensor([4000.0, 7000.0], device="cuda")

        with array_api_enable(True):
            for method in ("Krystek 1985", "Planck 1900"):
                uv = CCT_to_uv(CCT, method=method)

                assert uv.device == CCT.device

    @pytest.mark.parametrize("method", ["Krystek 1985", "Planck 1900"])
    @pytest.mark.parametrize(
        "CCT",
        [[6504.0, 0.003], [6504.0, float("nan")]],
        ids=["implausibly-low", "non-finite"],
    )
    def test_domain_CCT_to_uv(self, method: str, CCT: list[float]) -> None:
        """Test locus-only methods warn for invalid temperatures."""

        with pytest.warns(
            ColourUsageWarning,
            match="Correlated colour temperature must be finite",
        ):
            CCT_to_uv(CCT, method=method)


class TestXy_to_CCT:
    """
    Define :func:`colour.temperature.xy_to_CCT` definition unit tests methods.
    """

    def test_xy_to_CCT(self, xp: ModuleType) -> None:
        """Test :func:`colour.temperature.xy_to_CCT` definition."""

        xy = xp_as_array([0.31270, 0.32900], xp=xp)

        # Test default method (CIE Illuminant D Series)
        xp_assert_close(
            xy_to_CCT(xy),
            6508.1175148,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        # Test Hernandez 1999 method
        xp_assert_close(
            xy_to_CCT(xy, "Hernandez 1999"),
            6500.7420431,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        # Test McCamy 1992 method
        xp_assert_close(
            xy_to_CCT(xy, "McCamy 1992"),
            6505.08059131,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )


class TestCCT_to_xy:
    """
    Define :func:`colour.temperature.CCT_to_xy` definition unit tests methods.
    """

    def test_CCT_to_xy(self, xp: ModuleType) -> None:
        """Test :func:`colour.temperature.CCT_to_xy` definition."""

        # Test default method (CIE Illuminant D Series)
        xp_assert_close(
            CCT_to_xy(xp_as_array([6500], xp=xp)),
            [[0.31277888, 0.3291835]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test explicit CIE Illuminant D Series method
        xp_assert_close(
            CCT_to_xy(xp_as_array([6500], xp=xp), method="CIE Illuminant D Series"),
            [[0.31277888, 0.3291835]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test Hernandez 1999 method
        xp_assert_close(
            CCT_to_xy(xp_as_array([6500], xp=xp), "Hernandez 1999"),
            [[0.31271354, 0.32900208]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
