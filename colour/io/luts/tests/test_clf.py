# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.clf` module."""
import os
import unittest

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_CLF",
    "TestReadCLF",
]

from colour.io.luts.clf import read_clf

ROOT_CLF: str = os.path.join(os.path.dirname(__file__), "resources", "clf")


class TestReadCLF(unittest.TestCase):
    def test_read_something(self):
        read_clf(os.path.join(ROOT_CLF, "ACES2065_1_to_ACEScct.xml"))


if __name__ == "__main__":
    unittest.main()
