"""Defines the unit tests for the :mod:`colour.difference.cam16_ucs` module."""

import unittest

from colour.difference.tests.test_cam02_ucs import TestDelta_E_Luo2006

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestDelta_E_Li2017",
]


class TestDelta_E_Li2017(TestDelta_E_Luo2006):
    """
    Define :func:`colour.difference.cam16_ucs.delta_E_Li2017` definition unit
    tests methods.

    Notes
    -----
    -   :func:`colour.difference.cam16_ucs.delta_E_Li2017` is a wrapper
        of :func:`colour.difference.cam02_ucs.delta_E_Luo2006` and thus
        currently adopts the same unittests.
    """


if __name__ == "__main__":
    unittest.main()
