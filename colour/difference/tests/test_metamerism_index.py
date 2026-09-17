"""Define the unit tests for the :mod:`colour.difference.metamerism_index` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np

from colour import (
    CCS_ILLUMINANTS,
    MSDS_CMFS,
    SDS_ILLUMINANTS,
)
from colour.colorimetry import (
    SpectralDistribution,
    SpectralShape,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.difference.metamerism_index import (
    Lab_to_metamerism_index,
    XYZ_to_metamerism_index,
    sd_to_metamerism_index,
)
from colour.utilities import domain_range_scale, xp_as_array, xp_assert_close

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLab_to_Metamerism_Index",
    "TestXYZ_to_Metamerism_Index",
    "TestSD_to_Metamerism_IndexAutodiff",
]


class TestLab_to_Metamerism_Index:
    """
    Define :func:`colour.difference.metamerism_index.Lab_to_metamerism_index`
    definition unit tests methods.
    """

    def test_domain_range_scale_Lab_to_metamerism_index(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.difference.metamerism_index.Lab_to_metamerism_index`
        definition domain and range scale support.
        """

        Lab_1 = xp_as_array([48.99183622, -0.10561667, 400.65619925], xp=xp)
        offset = xp_as_array([0, 0, 2], xp=xp)

        c = ("Additive", "Multiplicative")
        m = ("CIE 1976", "CIE 1994", "CIE 2000", "CMC", "DIN99")
        it = [
            (
                correction,
                method,
                Lab_to_metamerism_index(
                    Lab_1 + offset,
                    Lab_1,
                    Lab_1,
                    Lab_1,
                    correction=correction,
                    method=method,
                ),
            )
            for correction in c
            for method in m
        ]

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for correction, method, value in it:
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    xp_assert_close(
                        Lab_to_metamerism_index(
                            (Lab_1 + offset) * factor,
                            Lab_1 * factor,
                            Lab_1 * factor,
                            Lab_1 * factor,
                            correction=correction,
                            method=method,
                        ),
                        value,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )


class TestXYZ_to_Metamerism_Index:
    """
    Define :func:`colour.difference.metamerism_index.XYZ_to_metamerism_index`
    definition unit tests methods.
    """

    def test_domain_range_scale_XYZ_to_metamerism_index(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.difference.metamerism_index.XYZ_to_metamerism_index`
        definition domain and range scale support.
        """

        XYZ_1 = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        offset = xp_as_array([0, 0, 0.01], xp=xp)

        c = ("Additive", "Multiplicative")
        m = ("CIE 1976", "CIE 1994", "CIE 2000", "CMC", "DIN99")
        it = [
            (
                correction,
                method,
                XYZ_to_metamerism_index(
                    XYZ_1 + offset,
                    XYZ_1,
                    XYZ_1,
                    XYZ_1,
                    correction=correction,
                    method=method,
                ),
            )
            for correction in c
            for method in m
        ]

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for correction, method, value in it:
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    xp_assert_close(
                        XYZ_to_metamerism_index(
                            (XYZ_1 + offset) * factor,
                            XYZ_1 * factor,
                            XYZ_1 * factor,
                            XYZ_1 * factor,
                            correction=correction,
                            method=method,
                        ),
                        value,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )


class TestSD_to_Metamerism_Index:
    """
    Define :func:`colour.difference.metamerism_index.sd_to_metamerism_index`
    definition unit tests methods.
    """

    def test_domain_range_scale_sd_to_metamerism_index(
        self,
        xp: ModuleType,  # noqa: ARG002
    ) -> None:
        """
        Test :func:`colour.difference.metamerism_index.sd_to_metamerism_index`
        definition domain and range scale support.
        """
        shape = SpectralShape(400, 700, 10)

        N_spl = np.array(
            [
                0.0379,
                0.0403,
                0.0415,
                0.0427,
                0.045,
                0.0483,
                0.0521,
                0.0572,
                0.0624,
                0.0673,
                0.0777,
                0.1026,
                0.1307,
                0.145,
                0.1484,
                0.1455,
                0.1375,
                0.1254,
                0.1099,
                0.0908,
                0.0698,
                0.0526,
                0.0423,
                0.0368,
                0.0331,
                0.0306,
                0.0297,
                0.0311,
                0.034,
                0.038,
                0.0421,
            ]
        )
        N_std = np.array(
            [
                0.099,
                0.1244,
                0.0933,
                0.0596,
                0.0405,
                0.0322,
                0.0299,
                0.0316,
                0.0377,
                0.0507,
                0.0681,
                0.0968,
                0.1522,
                0.2014,
                0.1991,
                0.159,
                0.1162,
                0.0843,
                0.0655,
                0.057,
                0.0553,
                0.0582,
                0.0638,
                0.0716,
                0.0818,
                0.0959,
                0.1131,
                0.1317,
                0.149,
                0.1656,
                0.1832,
            ]
        )

        N_spl = SpectralDistribution(N_spl, shape)
        N_std = SpectralDistribution(N_std, shape)

        o = (
            "CIE 1931 2 Degree Standard Observer",
            "CIE 1964 10 Degree Standard Observer",
        )
        i = ("A", "FL2")
        m = ("CIE 1976", "CIE 1994", "CIE 2000", "CMC", "DIN99")
        it = [
            (
                observer,
                illuminant,
                method,
                sd_to_metamerism_index(
                    N_spl,
                    N_std,
                    MSDS_CMFS[observer],
                    SDS_ILLUMINANTS["D65"],
                    SDS_ILLUMINANTS[illuminant],
                    method=method,
                    illuminant=CCS_ILLUMINANTS[observer][illuminant],
                ),
            )
            for observer in o
            for illuminant in i
            for method in m
        ]

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for observer, illuminant, method, value in it:
            for scale, _ in d_r:
                with domain_range_scale(scale):
                    xp_assert_close(
                        sd_to_metamerism_index(
                            N_spl,
                            N_std,
                            MSDS_CMFS[observer],
                            SDS_ILLUMINANTS["D65"],
                            SDS_ILLUMINANTS[illuminant],
                            method=method,
                            illuminant=CCS_ILLUMINANTS[observer][illuminant],
                        ),
                        value,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )


class TestSD_to_Metamerism_IndexAutodiff:
    """
    Define automatic differentiation regression tests for
    :func:`colour.difference.metamerism_index.sd_to_metamerism_index`.

    The namespace was previously inferred from the NumPy weighting factors, so
    backend spectral values were consumed by NumPy matrix products and detached.
    """

    def test_autodiff_sd_to_metamerism_index(
        self, xp: ModuleType, autodiff: typing.Callable
    ) -> None:
        """Test that a finite gradient reaches both spectral inputs."""

        wavelengths = np.arange(400.0, 701.0, 10.0)

        def function(values_spl: typing.Any, values_std: typing.Any) -> typing.Any:
            """Return the metamerism index for the spectral values."""

            return sd_to_metamerism_index(
                SpectralDistribution(values_spl, wavelengths),
                SpectralDistribution(values_std, wavelengths),
                MSDS_CMFS["CIE 1964 10 Degree Standard Observer"],
                SDS_ILLUMINANTS["D65"],
                SDS_ILLUMINANTS["A"],
                method="CIE 1976",
            )

        _M_t, (gradient_spl, gradient_std), _inputs = autodiff(
            function,
            np.linspace(0.1, 0.9, wavelengths.size),
            np.linspace(0.15, 0.95, wavelengths.size),
        )

        assert xp.isfinite(gradient_spl).all()
        assert xp.isfinite(gradient_std).all()
        assert xp.any(gradient_spl != 0)
        assert xp.any(gradient_std != 0)
