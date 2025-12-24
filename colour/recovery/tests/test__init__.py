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
                        0.04502017,
                        0.04491066,
                        0.04401528,
                        0.04239764,
                        0.04037903,
                        0.03830670,
                        0.03646088,
                        0.03501204,
                        0.03405960,
                        0.03380749,
                        0.03500319,
                        0.03978677,
                        0.05276604,
                        0.08118748,
                        0.13214985,
                        0.20596348,
                        0.28926932,
                        0.35571195,
                        0.37864559,
                        0.37864550,
                        0.37864548,
                        0.37864548,
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
                        0.07907520,
                        0.07908202,
                        0.07910307,
                        0.07916350,
                        0.07932472,
                        0.07972432,
                        0.08064378,
                        0.08260607,
                        0.08648611,
                        0.09328813,
                        0.10285906,
                        0.11714063,
                        0.13830854,
                        0.16737772,
                        0.20321625,
                        0.24209054,
                        0.27810408,
                        0.30456680,
                        0.31590278,
                        0.30938808,
                        0.28600905,
                        0.25008935,
                        0.20787601,
                        0.16570663,
                        0.12847670,
                        0.09887699,
                        0.07746070,
                        0.06326777,
                        0.05461559,
                        0.04974938,
                        0.04721892,
                        0.04600037,
                        0.04545631,
                        0.04523086,
                        0.04514410,
                        0.04511307,
                        0.04510275,
                        0.04509956,
                        0.04509864,
                        0.04509839,
                        0.04509833,
                        0.04509831,
                        0.04509831,
                    ],
                    [
                        0.31783484,
                        0.31783582,
                        0.31783884,
                        0.31784751,
                        0.31787064,
                        0.31792797,
                        0.31805989,
                        0.31834143,
                        0.31889811,
                        0.31728417,
                        0.29748230,
                        0.26127506,
                        0.21657221,
                        0.17172394,
                        0.13314246,
                        0.10401617,
                        0.08435713,
                        0.07202445,
                        0.06408687,
                        0.05795909,
                        0.05201651,
                        0.04567674,
                        0.03911964,
                        0.03287472,
                        0.02746332,
                        0.02319338,
                        0.02011362,
                        0.01807529,
                        0.01683340,
                        0.01613510,
                        0.01577202,
                        0.01559718,
                        0.01551912,
                        0.01548677,
                        0.01547433,
                        0.01546987,
                        0.01546839,
                        0.01546793,
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
