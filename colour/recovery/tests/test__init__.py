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
                        0.04502017,
                        0.04502015,
                        0.04489210,
                        0.04400064,
                        0.04238958,
                        0.04037806,
                        0.03831136,
                        0.03646870,
                        0.03502057,
                        0.03406722,
                        0.03381382,
                        0.03500919,
                        0.03979465,
                        0.05277835,
                        0.08120475,
                        0.13216747,
                        0.20597126,
                        0.28925717,
                        0.35567926,
                        0.37863546,
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
                        0.37864547,
                        0.37864547,
                    ],
                    [
                        0.07907484,
                        0.07908080,
                        0.07909921,
                        0.07915203,
                        0.07929297,
                        0.07964228,
                        0.08044604,
                        0.08216141,
                        0.08555320,
                        0.09175784,
                        0.10223108,
                        0.11713910,
                        0.13832653,
                        0.16740626,
                        0.20324275,
                        0.24210352,
                        0.27809718,
                        0.30395861,
                        0.31482523,
                        0.30928669,
                        0.28597986,
                        0.25007483,
                        0.20787699,
                        0.16571972,
                        0.12849639,
                        0.09889789,
                        0.07747902,
                        0.06328172,
                        0.05462505,
                        0.04975517,
                        0.04722215,
                        0.04600202,
                        0.04545708,
                        0.04523120,
                        0.04514423,
                        0.04511312,
                        0.04510276,
                        0.04509956,
                        0.04509864,
                        0.04509839,
                        0.04509833,
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
                        0.31553936,
                        0.29676630,
                        0.26127332,
                        0.21659273,
                        0.17175648,
                        0.13317267,
                        0.10403097,
                        0.08434927,
                        0.07133101,
                        0.06285827,
                        0.05784349,
                        0.05198323,
                        0.04566018,
                        0.03912076,
                        0.03288964,
                        0.02748577,
                        0.02321721,
                        0.02013450,
                        0.01809120,
                        0.01684419,
                        0.01614171,
                        0.01577570,
                        0.01559906,
                        0.01552000,
                        0.01548715,
                        0.01547448,
                        0.01546993,
                        0.01546841,
                        0.01546794,
                        0.01546780,
                        0.01546777,
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
