"""
Delegate - Event Notifications
==============================

Define a delegate class for event notifications:

-   :class:`colour.utilities.Delegate`
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import Any, Callable, List

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = ["Delegate"]


class Delegate:
    """
    Define a delegate allowing listeners to register and be notified of events.

    Methods
    -------
    -   :meth:`~colour.utilities.Delegate.add_listener`
    -   :meth:`~colour.utilities.Delegate.remove_listener`
    -   :meth:`~colour.utilities.Delegate.notify`
    """

    def __init__(self) -> None:
        self._listeners: List = []

    def add_listener(self, listener: Callable) -> None:
        """
        Add the given listener to the delegate.

        Parameters
        ----------
        listener
            Callable listening to the delegate notifications.
        """

        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable) -> None:
        """
        Remove the given listener from the delegate.

        Parameters
        ----------
        listener
            Callable listening to the delegate notifications.
        """

        if listener in self._listeners:
            self._listeners.remove(listener)

    def notify(self, *args: Any, **kwargs: Any) -> None:
        """
        Notify the delegate listeners.
        """

        for listener in self._listeners:
            listener(*args, **kwargs)
