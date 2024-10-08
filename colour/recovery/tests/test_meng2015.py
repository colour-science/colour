"""Define the unit tests for the :mod:`colour.recovery.meng2015` module."""

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
from colour.recovery import XYZ_to_sd_Meng2015
from colour.utilities import domain_range_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_sd_Meng2015",
]


class TestXYZ_to_sd_Meng2015:
    """
    Define :func:`colour.recovery.meng2015.XYZ_to_sd_Meng2015` definition unit
    tests methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        self._cmfs = reshape_msds(
            MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
            SpectralShape(360, 780, 10),
        )
        self._sd_D65 = reshape_sd(SDS_ILLUMINANTS["D65"], self._cmfs.shape)
        self._sd_E = reshape_sd(SDS_ILLUMINANTS["E"], self._cmfs.shape)

    def test_XYZ_to_sd_Meng2015(self):
        """Test :func:`colour.recovery.meng2015.XYZ_to_sd_Meng2015` definition."""

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        np.testing.assert_allclose(
            sd_to_XYZ_integration(
                XYZ_to_sd_Meng2015(XYZ, self._cmfs, self._sd_D65),
                self._cmfs,
                self._sd_D65,
            )
            / 100,
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            sd_to_XYZ_integration(
                XYZ_to_sd_Meng2015(XYZ, self._cmfs, self._sd_E),
                self._cmfs,
                self._sd_E,
            )
            / 100,
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            sd_to_XYZ_integration(
                XYZ_to_sd_Meng2015(
                    XYZ,
                    self._cmfs,
                    self._sd_D65,
                    optimisation_kwargs={
                        "options": {
                            "ftol": 1e-10,
                        }
                    },
                ),
                self._cmfs,
                self._sd_D65,
            )
            / 100,
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        shape = SpectralShape(400, 700, 5)
        cmfs = reshape_msds(self._cmfs, shape)
        np.testing.assert_allclose(
            sd_to_XYZ_integration(
                XYZ_to_sd_Meng2015(XYZ, cmfs, self._sd_D65), cmfs, self._sd_D65
            )
            / 100,
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_raise_exception_XYZ_to_sd_Meng2015(self):
        """
        Test :func:`colour.recovery.meng2015.XYZ_to_sd_Meng2015`
        definition raised exception.
        """

        pytest.raises(
            RuntimeError,
            XYZ_to_sd_Meng2015,
            np.array([0.0, 0.0, 1.0]),
            optimisation_kwargs={
                "options": {"maxiter": 10},
            },
        )

    def test_domain_range_scale_XYZ_to_sd_Meng2015(self):
        """
        Test :func:`colour.recovery.meng2015.XYZ_to_sd_Meng2015` definition
        domain and range scale support.
        """

        XYZ_i = np.array([0.20654008, 0.12197225, 0.05136952])
        XYZ_o = sd_to_XYZ_integration(
            XYZ_to_sd_Meng2015(XYZ_i, self._cmfs, self._sd_D65),
            self._cmfs,
            self._sd_D65,
        )

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    sd_to_XYZ_integration(
                        XYZ_to_sd_Meng2015(XYZ_i * factor_a, self._cmfs, self._sd_D65),
                        self._cmfs,
                        self._sd_D65,
                    ),
                    XYZ_o * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
