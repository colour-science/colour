"""Define the unit tests for the :mod:`colour.io.luts.common` module."""


from colour.io.luts.common import path_to_title

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPathToTitle",
]


class TestPathToTitle:
    """
    Define :func:`colour.io.luts.common.path_to_title` definition unit tests
    methods.
    """

    def test_path_to_title(self):
        """Test :func:`colour.io.luts.common.path_to_title` definition."""

        assert (
            path_to_title("colour/io/luts/tests/resources/cinespace/RGB_1_0.5_0.25.csp")
            == "RGB 1 0 5 0 25"
        )
