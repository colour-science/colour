"""Define the unit tests for the :mod:`colour.utilities.documentation` module."""

import os

from colour.utilities.documentation import is_documentation_building

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestIsDocumentationBuilding",
]


class TestIsDocumentationBuilding:
    """
    Define :func:`colour.utilities.documentation.is_documentation_building`
    definition unit tests methods.
    """

    def test_is_documentation_building(self):
        """
        Test :func:`colour.utilities.documentation.is_documentation_building`
        definition.
        """

        try:
            assert not is_documentation_building()

            os.environ["READTHEDOCS"] = "True"
            assert is_documentation_building()

            os.environ["READTHEDOCS"] = "False"
            assert is_documentation_building()

            del os.environ["READTHEDOCS"]
            assert not is_documentation_building()

            os.environ["COLOUR_SCIENCE__DOCUMENTATION_BUILD"] = "True"
            assert is_documentation_building()

            os.environ["COLOUR_SCIENCE__DOCUMENTATION_BUILD"] = "False"
            assert is_documentation_building()

            del os.environ["COLOUR_SCIENCE__DOCUMENTATION_BUILD"]
            assert not is_documentation_building()

        finally:  # pragma: no cover
            if os.environ.get("READTHEDOCS"):
                del os.environ["READTHEDOCS"]

            if os.environ.get("COLOUR_SCIENCE__DOCUMENTATION_BUILD"):
                del os.environ["COLOUR_SCIENCE__DOCUMENTATION_BUILD"]
