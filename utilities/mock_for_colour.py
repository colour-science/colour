"""
Mock for Colour
===============

Defines various mock objects to use with
`Colour <https://github.com/colour-science/colour>`__.
"""

from unittest.mock import MagicMock

__author__ = "Sphinx Team, Colour Developers"
__copyright__ = "Copyright 2007-2019 - Sphinx Team"
__copyright__ += ", "
__copyright__ += "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "mock_scipy_for_colour",
]


def mock_scipy_for_colour():
    """Mock *Scipy* for *Colour*."""

    import sys

    for module in (
        "scipy",
        "scipy.interpolate",
        "scipy.linalg",
        "scipy.ndimage",
        "scipy.optimize",
        "scipy.spatial",
        "scipy.spatial.distance",
    ):
        sys.modules[module] = MagicMock()


if __name__ == "__main__":
    mock_scipy_for_colour()

    import colour

    xyY = (0.4316, 0.3777, 0.1008)
    print(colour.xyY_to_XYZ(xyY))  # noqa: T201
