"""
Plotting - Pytest Configuration
===============================

Configures *pytest* to use the *Matplotlib* *AGG* headless backend. This allows
the plotting unittests to run without creating windows in IDEs such as
*VSCode*.
"""

import matplotlib
import pytest

from colour.hints import Generator

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "mpl_headless_backend",
]


@pytest.fixture(autouse=True, scope="session")
def mpl_headless_backend() -> Generator[None, None, None]:
    """
    Configure *Matplotlib* for headless testing.

    This pytest fixture is automatically applied to any tests in this package
    or any subpackages at the beginning of the pytest session.

    Yields
    ------
    Generator
        *Matplotlib* unit tests.
    """

    current_backend = matplotlib.get_backend()
    matplotlib.use("AGG")
    yield
    matplotlib.use(current_backend)
