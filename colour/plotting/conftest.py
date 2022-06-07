"""
Plotting - Tests
================
Configures pytest to use the AGG headless backend. This allows the plotting
unittests to run without creating windows in IDEs such as VSCode.
"""
from typing import Generator
import pytest
import matplotlib

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"


@pytest.fixture(autouse=True, scope="session")
def mpl_headless_backend() -> Generator[None, None, None]:
    """
    Configure matplotlib for headless testing.

    pytest Fixture automatically applied to any tests in this package or any
    subpackages at the beginning of the pytest session.
    """
    curBackend = matplotlib.get_backend()
    matplotlib.use("AGG")
    yield
    matplotlib.use(curBackend)
