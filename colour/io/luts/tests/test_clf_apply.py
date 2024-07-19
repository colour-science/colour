# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.clf` module."""
import unittest

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

import numpy as np

from colour.io.luts.tests.test_clf_common import snippet_to_process_list
from colour.io.luts import clf


class TestLUT1D(unittest.TestCase):
    EXAMPLE = """
        <LUT1D id="lut-23" name="4 Value Lut" inBitDepth="12i" outBitDepth="12i">
            <Description>1D LUT - Turn 4 grey levels into 4 inverted codes</Description>
            <Array dim="4">
                3
                2
                1
                0
            </Array>
        </LUT1D>
        """

    def test_lut1D(self):
        process_list = snippet_to_process_list(self.EXAMPLE)
        value = np.array([0.0, 1.0, 2.0, 3.0])
        result = clf.apply(process_list, value)
        np.testing.assert_array_almost_equal(result, np.array([3, 2, 1, 0]))



if __name__ == '__main__':
    unittest.main()
