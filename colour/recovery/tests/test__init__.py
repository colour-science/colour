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
                        0.04502016,
                        0.04502016,
                        0.04502011,
                        0.04501991,
                        0.04501907,
                        0.04501594,
                        0.04500555,
                        0.04497485,
                        0.04489411,
                        0.04470519,
                        0.04431242,
                        0.04358810,
                        0.04240741,
                        0.04071801,
                        0.03863221,
                        0.03652768,
                        0.03518988,
                        0.03610952,
                        0.04200427,
                        0.05729605,
                        0.08788588,
                        0.13800219,
                        0.20565314,
                        0.27952175,
                        0.34128296,
                        0.37409306,
                        0.37791491,
                        0.37831891,
                        0.37851419,
                        0.37859806,
                        0.37863011,
                        0.37864101,
                        0.37864431,
                        0.37864520,
                        0.37864542,
                        0.37864546,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                        0.37864547,
                    ],
                    [
                        0.07907261,
                        0.07907343,
                        0.07907663,
                        0.07908805,
                        0.07912551,
                        0.07923814,
                        0.07954835,
                        0.08033055,
                        0.08213408,
                        0.08593120,
                        0.09321741,
                        0.10592741,
                        0.12600935,
                        0.15459412,
                        0.19093799,
                        0.23161419,
                        0.27056360,
                        0.30036989,
                        0.31450783,
                        0.30962016,
                        0.28488808,
                        0.24487049,
                        0.19800977,
                        0.15261392,
                        0.11453763,
                        0.08629538,
                        0.06755391,
                        0.05634676,
                        0.05027814,
                        0.04729195,
                        0.04595303,
                        0.04540484,
                        0.04519954,
                        0.04512910,
                        0.04510694,
                        0.04510054,
                        0.04509884,
                        0.04509843,
                        0.04509833,
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
                        0.31041382,
                        0.28692135,
                        0.25120169,
                        0.20938318,
                        0.16783055,
                        0.13147870,
                        0.10298021,
                        0.08276274,
                        0.06970680,
                        0.06200281,
                        0.05779356,
                        0.05338983,
                        0.04711005,
                        0.03969581,
                        0.03237019,
                        0.02617428,
                        0.02161446,
                        0.01866301,
                        0.01697082,
                        0.01610732,
                        0.01571384,
                        0.01555336,
                        0.01549467,
                        0.01547540,
                        0.01546972,
                        0.01546821,
                        0.01546785,
                        0.01546777,
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
