"""Defines the unit tests for the :mod:`colour.difference.stress` module."""

import numpy as np
import unittest

from colour.difference import index_stress

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestIndexStress",
]


class TestIndexStress(unittest.TestCase):
    """
    Define :func:`colour.difference.stress.index_stress_Garcia2007` definition
    unit tests methods.
    """

    def test_index_stress(self):
        """
        Test :func:`colour.difference.stress.index_stress_Garcia2007`
        definition.
        """

        d_E = np.array([2.0425, 2.8615, 3.4412])
        d_V = np.array([1.2644, 1.2630, 1.8731])

        self.assertAlmostEqual(
            index_stress(d_E, d_V), 0.121170939369957, places=7
        )


if __name__ == "__main__":
    unittest.main()
