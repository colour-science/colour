"""Define the unit tests for the :mod:`colour.recovery` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np
import pytest

from colour.colorimetry import (
    MSDS_CMFS,
    SDS_ILLUMINANTS,
    SpectralShape,
    reshape_msds,
    reshape_sd,
    sd_to_XYZ_integration,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import XYZ_to_RGB
from colour.recovery import MSDS_GAUSSIAN_BASIS, XYZ_to_msds, XYZ_to_sd
from colour.recovery.gaussian import RGB_COLOURSPACE_GAUSSIAN
from colour.recovery.smits1999 import RGB_to_msds_Smits1999
from colour.utilities import (
    ColourUsageWarning,
    domain_range_scale,
    is_numpy_namespace,
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
    "TestXYZ_to_msds",
    "TestXYZ_to_sd",
]


class TestXYZ_to_sd:
    """
    Define :func:`colour.recovery.XYZ_to_sd` definition unit tests
    methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._cmfs = reshape_msds(
            MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
            SpectralShape(360, 780, 10),
        )

        self._sd_D65 = reshape_sd(SDS_ILLUMINANTS["D65"], self._cmfs.shape)

    @pytest.mark.parametrize("method", ["Meng 2015", "Jakob 2019"])
    @pytest.mark.mps_xfail("MPS float32 singular matrix")
    def test_backend_optimisation_warning(self, method: str, xp: ModuleType) -> None:
        """Test backend optimisation warnings through the generic wrapper."""

        if is_numpy_namespace(xp):
            pytest.skip("The warning applies to automatic differentiation backends.")

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)

        with pytest.warns(
            ColourUsageWarning,
            match='method="Gaussian" or method="Mallett 2019"',
        ):
            XYZ_to_sd(
                XYZ,
                method,
                cmfs=self._cmfs,
                illuminant=self._sd_D65,
            )

    @pytest.mark.mps_xfail("MPS float32 singular matrix")
    def test_domain_range_scale_XYZ_to_sd(
        self,
        xp: ModuleType,  # noqa: ARG002
    ) -> None:
        """
        Test :func:`colour.recovery.XYZ_to_sd` definition domain
        and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        m = (
            "Jakob 2019",
            "Mallett 2019",
            "Meng 2015",
            "Otsu 2018",
            "Smits 1999",
        )
        v = [
            sd_to_XYZ_integration(
                XYZ_to_sd(XYZ, method, cmfs=self._cmfs, illuminant=self._sd_D65),
                self._cmfs,
                self._sd_D65,
            )
            for method in m
        ]

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for method, value in zip(m, v, strict=True):
            for scale, factor_a, factor_b in d_r:
                with domain_range_scale(scale):
                    xp_assert_close(
                        sd_to_XYZ_integration(
                            XYZ_to_sd(
                                XYZ * factor_a,
                                method,
                                cmfs=self._cmfs,
                                illuminant=self._sd_D65,
                            ),
                            self._cmfs,
                            self._sd_D65,
                        ),
                        value * factor_b,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )


class TestXYZ_to_msds:
    """
    Define :func:`colour.recovery.XYZ_to_msds` definition unit tests
    methods.
    """

    def test_XYZ_to_msds(self) -> None:
        """
        Test :func:`colour.recovery.XYZ_to_msds` definition.
        """

        XYZ = np.array(
            [
                [0.20654008, 0.12197225, 0.05136952],
                [0.14222010, 0.23042768, 0.10495772],
                [0.07818780, 0.06157201, 0.28099326],
            ]
        )

        # Gaussian method
        msds_gaussian = XYZ_to_msds(XYZ, method="Gaussian", as_array=True)
        assert msds_gaussian.shape == (3, 421)

        # Test with interpolated basis at 10nm for full array comparison
        basis_10nm = MSDS_GAUSSIAN_BASIS.copy().align(SpectralShape(360, 780, 10))
        RGB = XYZ_to_RGB(XYZ, RGB_COLOURSPACE_GAUSSIAN)
        msds_gaussian_10nm = RGB_to_msds_Smits1999(RGB, basis_10nm, as_array=True)

        assert msds_gaussian_10nm.shape == (3, 43)

        xp_assert_close(
            msds_gaussian_10nm,
            [
                [
                    0.04502009,
                    0.04501988,
                    0.04501914,
                    0.04501680,
                    0.04500998,
                    0.04499185,
                    0.04494770,
                    0.04484930,
                    0.04464874,
                    0.04427526,
                    0.04364072,
                    0.04265916,
                    0.04128144,
                    0.03953613,
                    0.03755869,
                    0.03558836,
                    0.03392457,
                    0.03287194,
                    0.03279225,
                    0.03461775,
                    0.04157706,
                    0.06103056,
                    0.10398511,
                    0.17707942,
                    0.26899683,
                    0.34699756,
                    0.37718472,
                    0.37785039,
                    0.37824584,
                    0.37846014,
                    0.37856623,
                    0.37861426,
                    0.37863415,
                    0.37864170,
                    0.37864431,
                    0.37864515,
                    0.37864539,
                    0.37864545,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                ],
                [
                    0.07907249,
                    0.07907301,
                    0.07907546,
                    0.07908565,
                    0.07912360,
                    0.07925012,
                    0.07962779,
                    0.08063705,
                    0.08305097,
                    0.08821555,
                    0.09808908,
                    0.11492450,
                    0.14044675,
                    0.17466072,
                    0.21482897,
                    0.25537051,
                    0.28914692,
                    0.30987195,
                    0.31497706,
                    0.30547331,
                    0.28048697,
                    0.24387836,
                    0.19557999,
                    0.14451545,
                    0.10357764,
                    0.07663160,
                    0.06088546,
                    0.05238953,
                    0.04817010,
                    0.04627048,
                    0.04550213,
                    0.04522370,
                    0.04513336,
                    0.04510712,
                    0.04510030,
                    0.04509871,
                    0.04509838,
                    0.04509832,
                    0.04509831,
                    0.04509831,
                    0.04509831,
                    0.04509831,
                    0.04509831,
                ],
                [
                    0.31783443,
                    0.31783443,
                    0.31783443,
                    0.31783443,
                    0.31783443,
                    0.31783443,
                    0.31783443,
                    0.31783443,
                    0.31783443,
                    0.31368888,
                    0.28984943,
                    0.25028264,
                    0.20369873,
                    0.15863895,
                    0.12105230,
                    0.09341064,
                    0.07527351,
                    0.06457872,
                    0.05888491,
                    0.05613922,
                    0.05493718,
                    0.05429902,
                    0.04596035,
                    0.03204021,
                    0.02165872,
                    0.01705202,
                    0.01574487,
                    0.01550084,
                    0.01547045,
                    0.01546791,
                    0.01546776,
                    0.01546776,
                    0.01546776,
                    0.01546776,
                    0.01546776,
                    0.01546776,
                    0.01546776,
                    0.01546776,
                    0.01546776,
                    0.01546776,
                    0.01546776,
                    0.01546776,
                    0.01546776,
                ],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Smits 1999 method - native 10 wavelengths
        msds_smits = XYZ_to_msds(XYZ, method="Smits 1999", as_array=True)
        assert msds_smits.shape == (3, 10)

        xp_assert_close(
            msds_smits,
            [
                [
                    0.07878305,
                    0.06220187,
                    0.04462067,
                    0.03522208,
                    0.03241491,
                    0.03301050,
                    0.32071155,
                    0.38361649,
                    0.38361649,
                    0.38356492,
                ],
                [
                    0.07808712,
                    0.07712225,
                    0.08553484,
                    0.26638946,
                    0.31507478,
                    0.30136579,
                    0.09098278,
                    0.04509831,
                    0.04509831,
                    0.04568835,
                ],
                [
                    0.31671107,
                    0.31561096,
                    0.28928249,
                    0.14182485,
                    0.05421900,
                    0.05422828,
                    0.02160523,
                    0.02519571,
                    0.02820109,
                    0.02854381,
                ],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_msds(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.recovery.XYZ_to_msds` definition domain
        and range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)

        msds_reference = XYZ_to_msds(XYZ, method="Gaussian", as_array=True)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_msds(XYZ * factor, method="Gaussian", as_array=True),
                    msds_reference,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
