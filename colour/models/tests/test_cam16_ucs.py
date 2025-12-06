"""Define the unit tests for the :mod:`colour.models.cam16_ucs` module."""

from __future__ import annotations

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.cam02_ucs import COEFFICIENTS_UCS_LUO2006
from colour.models.cam16_ucs import (
    UCS_Li2017_to_XYZ,
    XYZ_to_UCS_Li2017,
)
from colour.models.tests.test_cam02_ucs import (
    TestJMh_CIECAM02_to_UCS_Luo2006,
    TestUCS_Luo2006_to_JMh_CIECAM02,
    TestUCS_Luo2006_to_XYZ,
    TestXYZ_to_UCS_Luo2006,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestJMh_CAM16_to_UCS_Li2017",
    "TestUCS_Li2017_to_JMh_CAM16",
    "TestXYZ_to_UCS_Li2017",
    "TestUCS_Li2017_to_XYZ",
]


class TestJMh_CAM16_to_UCS_Li2017(TestJMh_CIECAM02_to_UCS_Luo2006):
    """
    Define :func:`colour.models.cam16_ucs.JMh_CAM16_to_UCS_Li2017`
    definition unit tests methods.

    Notes
    -----
    -   :func:`colour.models.cam16_ucs.JMh_CAM16_to_UCS_Li2017` is a wrapper
        of :func:`colour.models.cam02_ucs.JMh_CIECAM02_to_UCS_Luo2006` and thus
        currently adopts the same unittests.
    """


class TestUCS_Li2017_to_JMh_CAM16(TestUCS_Luo2006_to_JMh_CIECAM02):
    """
    Define :func:`colour.models.cam16_ucs.UCS_Li2017_to_JMh_CAM16`
    definition unit tests methods.

    Notes
    -----
    -   :func:`colour.models.cam16_ucs.UCS_Li2017_to_JMh_CAM16` is a wrapper
        of :func:`colour.models.cam02_ucs.UCS_Luo2006_to_JMh_CIECAM02` and thus
        currently adopts the same unittests.
    """


class TestXYZ_to_UCS_Li2017(TestXYZ_to_UCS_Luo2006):
    """
    Define :func:`colour.models.cam16_ucs.XYZ_to_UCS_Li2017`
    definition unit tests methods.
    """

    def test_XYZ_to_UCS_Li2017(self) -> None:
        """Test :func:`colour.models.cam16_ucs.XYZ_to_UCS_Li2017` definition."""

        np.testing.assert_allclose(
            XYZ_to_UCS_Li2017(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            ),
            np.array([46.06586033, 41.07586491, 14.51025826]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_UCS_Li2017(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
                XYZ_w=np.array([0.95047, 1.0, 1.08883]),
            ),
            np.array([46.06573617, 41.07444159, 14.50807598]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestUCS_Li2017_to_XYZ(TestUCS_Luo2006_to_XYZ):
    """
    Define :func:`colour.models.cam16_ucs.UCS_Li2017_to_XYZ`
    definition unit tests methods.
    """

    def test_UCS_Li2017_to_XYZ(self) -> None:
        """Test :func:`colour.models.cam16_ucs.UCS_Li2017_to_XYZ` definition."""

        np.testing.assert_allclose(
            UCS_Li2017_to_XYZ(
                np.array([46.06586033, 41.07586491, 14.51025826]),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            UCS_Li2017_to_XYZ(
                np.array([46.06586033, 41.07586491, 14.51025826]),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
                XYZ_w=np.array([0.95047, 1.0, 1.08883]),
            ),
            np.array([0.2065444, 0.12197263, 0.05136016]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
