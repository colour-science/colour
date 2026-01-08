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
                        0.04502017,
                        0.04502017,
                        0.04502017,
                        0.04502017,
                        0.04502017,
                        0.04502017,
                        0.04502017,
                        0.04502015,
                        0.04501943,
                        0.04500378,
                        0.04485416,
                        0.04414019,
                        0.04225346,
                        0.03923158,
                        0.03607704,
                        0.03383282,
                        0.03275723,
                        0.03246214,
                        0.03244087,
                        0.03248904,
                        0.03291355,
                        0.03423511,
                        0.03674078,
                        0.23102363,
                        0.37618614,
                        0.37801185,
                        0.37854111,
                        0.37863674,
                        0.37864515,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                    ],
                    [
                        0.07907236,
                        0.07907236,
                        0.07907236,
                        0.07907236,
                        0.07907236,
                        0.07907236,
                        0.07907237,
                        0.07907318,
                        0.07910769,
                        0.07969773,
                        0.08429374,
                        0.10265058,
                        0.14436445,
                        0.20330882,
                        0.25882515,
                        0.29503026,
                        0.31097879,
                        0.31487536,
                        0.31508579,
                        0.31415692,
                        0.30660302,
                        0.28307961,
                        0.23600594,
                        0.16298077,
                        0.09138543,
                        0.05890787,
                        0.0475304,
                        0.04531797,
                        0.04510714,
                        0.04509845,
                        0.04509831,
                        0.04509831,
                        0.04509831,
                        0.04509831,
                        0.04509831,
                        0.04509831,
                        0.04509831,
                        0.04509831,
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
                        0.31231388,
                        0.29111944,
                        0.25470353,
                        0.20896227,
                        0.16223517,
                        0.12192814,
                        0.09209977,
                        0.07302927,
                        0.0624658,
                        0.057393,
                        0.05528178,
                        0.05450763,
                        0.05403372,
                        0.05141869,
                        0.03596995,
                        0.01602146,
                        0.01546784,
                        0.01546777,
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
                        0.01546776,
                        0.01546776,
                        0.01546776,
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
