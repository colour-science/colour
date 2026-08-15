"""Define the unit tests for the :mod:`colour.utilities.delegate` module."""

from __future__ import annotations

from colour.utilities import (
    Delegate,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestDelegate",
]


class TestDelegate:
    """
    Define :class:`colour.utilities.structures.Delegate` class unit tests
    methods.
    """

    def test_required_methods(self) -> None:
        """Test the presence of required methods."""

        required_methods = ("add_listener", "remove_listener", "notify")

        for method in required_methods:
            assert method in dir(Delegate)

    def test_Delegate(self) -> None:
        """Test the :class:`colour.utilities.structures.Delegate` class."""

        delegate = Delegate()

        data = []

        def _listener(a: int) -> None:
            """Define a unit tests listener."""

            data.append(a)

        delegate.add_listener(_listener)

        delegate.notify("Foo")

        assert data == ["Foo"]

        delegate.remove_listener(_listener)

        delegate.notify("Bar")

        assert data == ["Foo"]
