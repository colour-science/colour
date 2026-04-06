"""Define the unit tests for the :mod:`colour.difference.stress` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType


from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.difference import index_stress
from colour.utilities import xp_as_array, xp_assert_close

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestIndexStress",
]


class TestIndexStress:
    """
    Define :func:`colour.difference.stress.index_stress_Garcia2007` definition
    unit tests methods.
    """

    def test_index_stress(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.difference.stress.index_stress_Garcia2007`
        definition.
        """

        d_E = xp_as_array([2.0425, 2.8615, 3.4412], xp=xp)
        d_V = xp_as_array([1.2644, 1.2630, 1.8731], xp=xp)

        xp_assert_close(
            index_stress(d_E, d_V),
            0.121170939369957,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
