"""
Callback Management
===================

Defines the callback management objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from colour.hints import (
    Any,
    Callable,
    List,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Callback",
    "MixinCallback",
]


@dataclass
class Callback:
    """
    Define a callback.

    Parameters
    ----------
    name
        Callback name.
    function
        Callback callable.
    """

    name: str
    function: Callable


class MixinCallback:
    """
    A mixin providing support for callbacks.

    Attributes
    ----------
    -   :attr:`~colour.utilities.MixinCallback.callbacks`
    -   :attr:`~colour.utilities.MixinCallback.__setattr__`

    Methods
    -------
    -   :meth:`~colour.utilities.MixinCallback.register_callback`
    -   :meth:`~colour.utilities.MixinCallback.unregister_callback`

    Examples
    --------
    >>> class WithCallback(MixinCallback):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.attribute_a = "a"
    ...
    >>> with_callback = WithCallback()
    >>> def _on_attribute_a_changed(self, name: str, value: str) -> str:
    ...     if name == "attribute_a":
    ...         value = value.upper()
    ...     return value
    >>> with_callback.register_callback(
    ...     "on_attribute_a_changed", _on_attribute_a_changed
    ... )
    >>> with_callback.attribute_a = "a"
    >>> with_callback.attribute_a
    'A'
    """

    def __init__(self) -> None:
        super().__init__()

        self._callbacks: List = []

    @property
    def callbacks(self) -> List:
        """
        Getter property for the callbacks.

        Returns
        -------
        :class:`list`
            Callbacks.
        """

        return self._callbacks

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set given value to the attribute with given name.

        Parameters
        ----------
        attribute
            Attribute to set the value of.
        value
            Value to set the attribute with.
        """

        if hasattr(self, "_callbacks"):
            for callback in self._callbacks:
                value = callback.function(self, name, value)

        super().__setattr__(name, value)

    def register_callback(self, name: str, function: Callable) -> None:
        """
        Register the callback with given name.

        Parameters
        ----------
        name
            Callback name.
        function
            Callback callable.

        Examples
        --------
        >>> class WithCallback(MixinCallback):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        >>> with_callback = WithCallback()
        >>> with_callback.register_callback("callback", lambda *args: None)
        >>> with_callback.callbacks  # doctest: +SKIP
        [Callback(name='callback', function=<function <lambda> at 0x10fcf3420>)]
        """

        self._callbacks.append(Callback(name, function))

    def unregister_callback(self, name: str) -> None:
        """
        Unregister the callback with given name.

        Parameters
        ----------
        name
            Callback name.

        Examples
        --------
        >>> class WithCallback(MixinCallback):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        >>> with_callback = WithCallback()
        >>> with_callback.register_callback("callback", lambda s, n, v: v)
        >>> with_callback.callbacks  # doctest: +SKIP
        [Callback(name='callback', function=<function <lambda> at 0x10fcf3420>)]
        >>> with_callback.unregister_callback("callback")
        >>> with_callback.callbacks
        []
        """

        self._callbacks = [
            callback for callback in self._callbacks if callback.name != name
        ]
