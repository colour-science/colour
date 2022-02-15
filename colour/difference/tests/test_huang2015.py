"""Defines the unit tests for the :mod:`colour.difference.huang2015` module."""

import numpy as np
import unittest

from colour.difference import power_function_Huang2015

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPowerFunctionHuang2015",
]


class TestPowerFunctionHuang2015(unittest.TestCase):
    """
    Define :func:`colour.difference.huang2015.power_function_Huang2015`
    definition unit tests methods.
    """

    def test_power_function_Huang2015(self):
        """
        Test :func:`colour.difference.huang2015.power_function_Huang2015`
        definition.
        """

        d_E = np.array([2.0425, 2.8615, 3.4412])

        np.testing.assert_almost_equal(
            power_function_Huang2015(d_E),
            np.array([2.35748796, 2.98505036, 3.39651062]),
            decimal=7,
        )


if __name__ == "__main__":
    unittest.main()
