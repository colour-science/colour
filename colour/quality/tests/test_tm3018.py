"""
Define the unit tests for the :mod:`colour.quality.tm3018` module.

Notes
-----
-   Reference data was created using the official Excel spreadsheet, published
    by the IES at this URL:
    http://media.ies.org/docs/errata/TM-30-18_tools_etc.zip.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np
import pytest

from colour.colorimetry import (
    SDS_ILLUMINANTS,
    SDS_LIGHT_SOURCES,
    SPECTRAL_SHAPE_DEFAULT,
    MultiSpectralDistributions,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.quality.tm3018 import (
    averages_area,
    colour_fidelity_index_ANSIIESTM3018,
)
from colour.utilities import as_float_array, as_ndarray, xp_as_array, xp_assert_close

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestColourFidelityIndexANSIIESTM3018",
    "TestAveragesArea",
]


class TestColourFidelityIndexANSIIESTM3018:
    """
    Define :func:`colour.quality.tm3018.colour_fidelity_index_ANSIIESTM3018`
    definition unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-2)
    def test_colour_fidelity_index_ANSIIESTM3018(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.quality.tm3018.colour_fidelity_index_ANSIIESTM3018`
        definition.
        """

        sd_fl2_xp = SDS_ILLUMINANTS["FL2"].copy(xp=xp)

        # Test without additional data (returns R_f only)
        R_f = colour_fidelity_index_ANSIIESTM3018(sd_fl2_xp, additional_data=False)
        xp_assert_close(R_f, 70, atol=TOLERANCE_ABSOLUTE_TESTS * 2e06)

        # Test with additional data (returns full specification)
        specification = colour_fidelity_index_ANSIIESTM3018(
            sd_fl2_xp, additional_data=True
        )

        xp_assert_close(specification.R_f, 70, atol=TOLERANCE_ABSOLUTE_TESTS * 2e06)
        xp_assert_close(specification.R_g, 86, atol=TOLERANCE_ABSOLUTE_TESTS * 5000000)
        xp_assert_close(
            specification.CCT, 4225, atol=TOLERANCE_ABSOLUTE_TESTS * 10000000
        )
        xp_assert_close(
            specification.D_uv, 0.0019, atol=TOLERANCE_ABSOLUTE_TESTS * 10000
        )

        xp_assert_close(
            specification.R_s,
            [
                79,
                59,
                67,
                66,
                36,
                66,
                40,
                35,
                95,
                54,
                48,
                45,
                64,
                87,
                72,
                49,
                56,
                69,
                57,
                44,
                47,
                47,
                80,
                63,
                48,
                59,
                82,
                85,
                62,
                70,
                68,
                62,
                74,
                74,
                86,
                88,
                80,
                76,
                97,
                93,
                91,
                89,
                83,
                99,
                83,
                81,
                87,
                66,
                80,
                81,
                81,
                76,
                69,
                77,
                77,
                66,
                66,
                67,
                79,
                90,
                78,
                87,
                77,
                60,
                61,
                58,
                56,
                62,
                73,
                58,
                64,
                84,
                53,
                96,
                67,
                57,
                76,
                63,
                82,
                85,
                74,
                94,
                91,
                86,
                81,
                64,
                74,
                69,
                66,
                68,
                93,
                51,
                70,
                41,
                62,
                70,
                80,
                67,
                45,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS * 7500000,
        )

        xp_assert_close(
            specification.R_fs,
            [60, 61, 53, 68, 80, 88, 77, 73, 76, 62, 70, 77, 81, 71, 64, 65],
            atol=TOLERANCE_ABSOLUTE_TESTS * 7500000,
        )
        xp_assert_close(
            specification.R_cs,
            [-25, -18, -9, 5, 11, 4, -8, -15, -17, -15, -4, 5, 11, 7, -6, -16],
            atol=TOLERANCE_ABSOLUTE_TESTS * 7500000,
        )
        xp_assert_close(
            specification.R_hs,
            [
                -0.02,
                0.14,
                0.24,
                0.20,
                0.09,
                -0.07,
                -0.12,
                -0.08,
                0.01,
                0.17,
                0.19,
                0.11,
                -0.08,
                -0.15,
                -0.26,
                -0.17,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS * 7500000,
        )

        # A :class:`MultiSpectralDistributions` batch returns the same
        # per-column results as evaluating each distribution individually.
        sds = [
            sd.copy().align(SPECTRAL_SHAPE_DEFAULT)
            for sd in (
                SDS_ILLUMINANTS["FL1"],
                SDS_ILLUMINANTS["FL2"],
                SDS_LIGHT_SOURCES["Neodimium Incandescent"],
                SDS_LIGHT_SOURCES["F32T8/TL841 (Triphosphor)"],
            )
        ]
        msds = MultiSpectralDistributions(
            xp_as_array(np.column_stack([sd.values for sd in sds]), xp=xp),
            sds[0].wavelengths,
            labels=[sd.name for sd in sds],
        )
        xp_assert_close(
            colour_fidelity_index_ANSIIESTM3018(msds),
            xp_as_array(
                [colour_fidelity_index_ANSIIESTM3018(sd.copy(xp=xp)) for sd in sds],
                xp=xp,
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_raise_exception_colour_fidelity_index_ANSIIESTM3018_msds(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.quality.tm3018.colour_fidelity_index_ANSIIESTM3018`
        raises :class:`NotImplementedError` for
        :class:`MultiSpectralDistributions` input combined with
        ``additional_data=True``.
        """

        sd_fl2_xp = SDS_ILLUMINANTS["FL2"].copy(xp=xp)
        msds = MultiSpectralDistributions(
            xp_as_array(sd_fl2_xp.values[:, None], xp=xp),
            sd_fl2_xp.wavelengths,
            labels=["FL2"],
        )

        with pytest.raises(NotImplementedError):
            colour_fidelity_index_ANSIIESTM3018(msds, additional_data=True)  # pyright: ignore[reportCallIssue, reportArgumentType]


class TestAveragesArea:
    """
    Define :func:`colour.quality.tm3018.averages_area` definition unit tests
    methods.
    """

    def test_averages_area(self, xp: ModuleType) -> None:
        """Test :func:`colour.quality.tm3018.averages_area` definition."""

        # Simple 3 * sqrt(2) by sqrt(2) rectangle.
        rectangle = as_float_array([[2, 1], [1, 2], [-2, -1], [-1, -2]])
        np.allclose(averages_area(rectangle), 6)

        # Concave polygon.
        poly = xp_as_array(
            [[1.0, -1], [1, 1], [3, 1], [3, 3], [-1, 3], [-1, -1]], xp=xp
        )
        np.allclose(as_ndarray(averages_area(poly)), 12)
