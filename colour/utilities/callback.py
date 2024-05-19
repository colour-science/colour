"""
Callback Management
===================

Defines the callback management objects.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from colour.hints import (
    Any,
    Callable,
    List,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
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
    >>> with_callback = WithCallback()
    >>> def _on_attribute_a_changed(self, name: str, value: str) -> str:
    ...     return value.upper()
    >>> with_callback.register_callback(
    ...     "attribute_a", "on_attribute_a_changed", _on_attribute_a_changed
    ... )
    >>> with_callback.attribute_a = "a"
    >>> with_callback.attribute_a
    'A'
    """

    def __init__(self) -> None:
        super().__init__()

        self._callbacks: defaultdict[str, List[Callback]] = defaultdict(list)

    @property
    def callbacks(self) -> defaultdict[str, List[Callback]]:
        """
        Getter property for the callbacks.

        Returns
        -------
        :class:`defaultdict`
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
            for callback in self._callbacks.get(name, []):
                value = callback.function(self, name, value)

        super().__setattr__(name, value)

    def register_callback(self, attribute: str, name: str, function: Callable) -> None:
        """
        Register the callback with given name for given attribute.

        Parameters
        ----------
        attribute
            Attribute to register the callback for.
        name
            Callback name.
        function
            Callback callable.

        Examples
        --------
        >>> class WithCallback(MixinCallback):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.attribute_a = "a"
        ...
        >>> with_callback = WithCallback()
        >>> with_callback.register_callback(
        ...     "attribute_a", "callback", lambda *args: None
        ... )
        >>> with_callback.callbacks  # doctest: +SKIP
        defaultdict(<class 'list'>, {'attribute_a': \
[Callback(name='callback', function=<function <lambda> at 0x...>)]})
        """

        self._callbacks[attribute].append(Callback(name, function))

    def unregister_callback(self, attribute: str, name: str) -> None:
        """
        Unregister the callback with given name for given attribute.

        Parameters
        ----------
        attribute
            Attribute to unregister the callback for.
        name
            Callback name.

        Examples
        --------
        >>> class WithCallback(MixinCallback):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.attribute_a = "a"
        ...
        >>> with_callback = WithCallback()
        >>> with_callback.register_callback(
        ...     "attribute_a", "callback", lambda s, n, v: v
        ... )
        >>> with_callback.callbacks  # doctest: +SKIP
        defaultdict(<class 'list'>, {'attribute_a': \
[Callback(name='callback', function=<function <lambda> at 0x...>)]})
        >>> with_callback.unregister_callback("attribute_a", "callback")
        >>> with_callback.callbacks
        defaultdict(<class 'list'>, {})
        """

        if self._callbacks.get(attribute) is None:  # pragma: no cover
            return

        self._callbacks[attribute] = [
            callback
            for callback in self._callbacks.get(attribute, [])
            if callback.name != name
        ]

        if len(self._callbacks[attribute]) == 0:
            self._callbacks.pop(attribute, None)
