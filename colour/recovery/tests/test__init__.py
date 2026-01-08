"""Define the unit tests for the :mod:`colour.recovery` module."""

from __future__ import annotations

import numpy as np

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
from colour.utilities import domain_range_scale, is_scipy_installed

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

    def test_domain_range_scale_XYZ_to_sd(self) -> None:
        """
        Test :func:`colour.recovery.XYZ_to_sd` definition domain
        and range scale support.
        """

        if not is_scipy_installed():  # pragma: no cover
            return

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
                    np.testing.assert_allclose(
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
        msds_gaussian = XYZ_to_msds(XYZ, method="Gaussian")
        assert msds_gaussian.shape == (3, 421)

        # Test with interpolated basis at 10nm for full array comparison
        basis_10nm = MSDS_GAUSSIAN_BASIS.copy().align(SpectralShape(360, 780, 10))
        RGB = XYZ_to_RGB(XYZ, RGB_COLOURSPACE_GAUSSIAN)
        msds_gaussian_10nm = RGB_to_msds_Smits1999(RGB, basis_10nm)

        assert msds_gaussian_10nm.shape == (3, 43)

        np.testing.assert_allclose(
            msds_gaussian_10nm,
            np.array(
                [
                    [
                        0.04502007,
                        0.04501982,
                        0.04501904,
                        0.04501674,
                        0.04501044,
                        0.04499446,
                        0.04495688,
                        0.04487501,
                        0.04470998,
                        0.04440261,
                        0.04387468,
                        0.04304062,
                        0.04183304,
                        0.04023947,
                        0.03833948,
                        0.03632532,
                        0.03449895,
                        0.03328757,
                        0.03344854,
                        0.03681723,
                        0.04782283,
                        0.07473123,
                        0.12724675,
                        0.20735718,
                        0.29799751,
                        0.36379568,
                        0.37716167,
                        0.37781923,
                        0.37821682,
                        0.37843829,
                        0.37855218,
                        0.37860633,
                        0.37863017,
                        0.37863990,
                        0.37864358,
                        0.37864488,
                        0.37864530,
                        0.37864543,
                        0.37864546,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                    ],
                    [
                        0.07907420,
                        0.07907881,
                        0.07909342,
                        0.07913649,
                        0.07925430,
                        0.07955327,
                        0.08025668,
                        0.08178961,
                        0.08488058,
                        0.09063910,
                        0.10053251,
                        0.11616675,
                        0.13880852,
                        0.16869547,
                        0.20434505,
                        0.24220710,
                        0.27699257,
                        0.30278343,
                        0.31385859,
                        0.30726286,
                        0.28416947,
                        0.24887274,
                        0.20745570,
                        0.16605611,
                        0.12941208,
                        0.10013865,
                        0.07879530,
                        0.06448445,
                        0.05560758,
                        0.05048644,
                        0.04772358,
                        0.04632116,
                        0.04564664,
                        0.04533673,
                        0.04519950,
                        0.04514043,
                        0.04511554,
                        0.04510523,
                        0.04510103,
                        0.04509935,
                        0.04509870,
                        0.04509845,
                        0.04509836,
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
                        0.31783443,
                        0.30460818,
                        0.26619050,
                        0.21415223,
                        0.16176513,
                        0.11867080,
                        0.08864076,
                        0.07059913,
                        0.06116093,
                        0.05593007,
                        0.05174748,
                        0.04745267,
                        0.04276864,
                        0.03787715,
                        0.03311108,
                        0.02877955,
                        0.02508995,
                        0.02213055,
                        0.01988726,
                        0.01827593,
                        0.01717700,
                        0.01646435,
                        0.01602438,
                        0.01576557,
                        0.01562039,
                        0.01554269,
                        0.01550300,
                        0.01548363,
                        0.01547461,
                        0.01547059,
                        0.01546888,
                        0.01546818,
                        0.01546791,
                        0.01546781,
                    ],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Smits 1999 method - native 10 wavelengths
        msds_smits = XYZ_to_msds(XYZ, method="Smits 1999")
        assert msds_smits.shape == (3, 10)

        np.testing.assert_allclose(
            msds_smits,
            np.array(
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
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_msds(self) -> None:
        """
        Test :func:`colour.recovery.XYZ_to_msds` definition domain
        and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])

        msds_reference = XYZ_to_msds(XYZ, method="Gaussian")

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_msds(XYZ * factor, method="Gaussian"),
                    msds_reference,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
