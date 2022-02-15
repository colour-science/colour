"""Defines the unit tests for the :mod:`colour.models.cam16_ucs` module."""

import unittest

from colour.models.tests.test_cam02_ucs import (
    TestJMh_CIECAM02_to_UCS_Luo2006,
    TestUCS_Luo2006_to_JMh_CIECAM02,
    TestXYZ_to_UCS_Luo2006,
    TestUCS_Luo2006_to_XYZ,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
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

    pass


class TestUCS_Li2017_to_XYZ(TestUCS_Luo2006_to_XYZ):
    """
    Define :func:`colour.models.cam16_ucs.UCS_Li2017_to_XYZ`
    definition unit tests methods.
    """

    pass


if __name__ == "__main__":
    unittest.main()
