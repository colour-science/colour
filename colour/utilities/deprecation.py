"""
Deprecation Utilities
=====================

Define deprecation management utilities for the Colour library.
"""

from __future__ import annotations

import sys
import typing
from dataclasses import dataclass
from importlib import import_module
from operator import attrgetter

if typing.TYPE_CHECKING:
    from colour.hints import Any, ModuleType

from colour.utilities import MixinDataclassIterable, attest, optional, usage_warning

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ObjectRenamed",
    "ObjectRemoved",
    "ObjectFutureRename",
    "ObjectFutureRemove",
    "ObjectFutureAccessChange",
    "ObjectFutureAccessRemove",
    "ModuleAPI",
    "ArgumentRenamed",
    "ArgumentRemoved",
    "ArgumentFutureRename",
    "ArgumentFutureRemove",
    "get_attribute",
    "build_API_changes",
    "handle_arguments_deprecation",
]


@dataclass(frozen=True)
class ObjectRenamed(MixinDataclassIterable):
    """
    Represent an object that has been renamed in the API.

    Parameters
    ----------
    name
        Object name that has been changed.
    new_name
        New object name.
    """

    name: str
    new_name: str

    def __str__(self) -> str:
        """
        Return a formatted string representation of the class.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return f'"{self.name}" object has been renamed to "{self.new_name}".'


@dataclass(frozen=True)
class ObjectRemoved(MixinDataclassIterable):
    """
    Represent an object that has been removed from the API.

    Parameters
    ----------
    name
        Object name that has been removed.
    """

    name: str

    def __str__(self) -> str:
        """
        Return a formatted string representation of the class.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return f'"{self.name}" object has been removed from the API.'


@dataclass(frozen=True)
class ObjectFutureRename(MixinDataclassIterable):
    """
    Represent an object that will be renamed in a future release.

    Parameters
    ----------
    name
        Object name that will change in a future release.
    new_name
        New object name.
    """

    name: str
    new_name: str

    def __str__(self) -> str:
        """
        Return a formatted string representation of the class.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return (
            f'"{self.name}" object is deprecated and will be renamed to '
            f'"{self.new_name}" in a future release.'
        )


@dataclass(frozen=True)
class ObjectFutureRemove(MixinDataclassIterable):
    """
    Represent an object that will be removed in a future release.

    Parameters
    ----------
    name
        Object name that will be removed in a future release.
    """

    name: str

    def __str__(self) -> str:
        """
        Return a formatted string representation of the class.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return (
            f'"{self.name}" object is deprecated and will be removed in '
            f"a future release."
        )


@dataclass(frozen=True)
class ObjectFutureAccessChange(MixinDataclassIterable):
    """
    Represent an object whose access pattern will change in a future
    release.

    Parameters
    ----------
    access
        Object access that will change in a future release.
    new_access
        New object access pattern.
    """

    access: str
    new_access: str

    def __str__(self) -> str:
        """
        Return a formatted string representation of the class.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return (
            f'"{self.access}" object access is deprecated and will change '
            f'to "{self.new_access}" in a future release.'
        )


@dataclass(frozen=True)
class ObjectFutureAccessRemove(MixinDataclassIterable):
    """
    Represent an object whose access will be removed in a future release.
    be removed in a future release.

    Parameters
    ----------
    name
        Object name whose access will be removed in a future release.
    """

    name: str

    def __str__(self) -> str:
        """
        Return a formatted string representation of the class.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return f'"{self.name}" object access will be removed in a future release.'


@dataclass(frozen=True)
class ArgumentRenamed(MixinDataclassIterable):
    """
    Represent an argument that has been renamed in the API.

    Parameters
    ----------
    name
        Argument name that has been changed.
    new_name
        New argument name.
    """

    name: str
    new_name: str

    def __str__(self) -> str:
        """
        Return a formatted string representation of the class.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return f'"{self.name}" argument has been renamed to "{self.new_name}".'


@dataclass(frozen=True)
class ArgumentRemoved(MixinDataclassIterable):
    """
    Represent an argument that has been removed from the API.

    Parameters
    ----------
    name
        Argument name that has been removed.
    """

    name: str

    def __str__(self) -> str:
        """
        Return a formatted string representation of the class.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return f'"{self.name}" argument has been removed from the API.'


@dataclass(frozen=True)
class ArgumentFutureRename(MixinDataclassIterable):
    """
    Represent an argument that will be renamed in a future release.
    change in a future release.

    Parameters
    ----------
    name
        Argument name that will change in a future release.
    new_name
        New argument name.
    """

    name: str
    new_name: str

    def __str__(self) -> str:
        """
        Return a formatted string representation of the class.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return (
            f'"{self.name}" argument is deprecated and will be renamed to '
            f'"{self.new_name}" in a future release.'
        )


@dataclass(frozen=True)
class ArgumentFutureRemove(MixinDataclassIterable):
    """
    Represent an argument that will be removed in a future release.

    Parameters
    ----------
    name
        Argument name that will be removed in a future release.
    """

    name: str

    def __str__(self) -> str:
        """
        Return a formatted string representation of the class.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return (
            f'"{self.name}" argument is deprecated and will be removed in '
            f"a future release."
        )


class ModuleAPI:
    """
    Define a class enabling customisation of module attribute access with
    built-in deprecation management functionality.

    Parameters
    ----------
    module
        Module for which to customise attribute access behaviour.

    Methods
    -------
    -   :meth:`~colour.utilities.ModuleAPI.__init__`
    -   :meth:`~colour.utilities.ModuleAPI.__getattr__`
    -   :meth:`~colour.utilities.ModuleAPI.__dir__`

    Examples
    --------
    >>> import sys
    >>> sys.modules["colour"] = ModuleAPI(sys.modules["colour"])
    ... # doctest: +SKIP
    """

    def __init__(self, module: ModuleType, changes: dict | None = None) -> None:
        self._module = module
        self._changes = optional(changes, {})

    def __getattr__(self, attribute: str) -> Any:
        """
        Return the specified attribute value while handling deprecation.

        Parameters
        ----------
        attribute
            Attribute name.

        Returns
        -------
        :class:`object`
            Attribute value.

        Raises
        ------
        AttributeError
            If the attribute is not defined.
        """

        change = self._changes.get(attribute)

        if change is not None:
            if not isinstance(change, ObjectRemoved):
                usage_warning(str(change))

                return (
                    getattr(self._module, attribute)
                    if isinstance(change, ObjectFutureRemove)
                    else get_attribute(change.values[1])
                )

            raise AttributeError(str(change))

        return getattr(self._module, attribute)

    def __dir__(self) -> list:
        """
        Return the list of names in the module local scope filtered according
        to the changes.

        Returns
        -------
        :class:`list`
            Filtered list of names in the module local scope.
        """

        return [
            attribute
            for attribute in dir(self._module)
            if attribute not in self._changes
        ]


def get_attribute(attribute: str) -> Any:
    """
    Retrieve the value of the specified attribute from its namespace.

    Parameters
    ----------
    attribute
        Attribute to retrieve, ``attribute`` must have a namespace
        module, e.g., *colour.models.oetf_inverse_BT2020*.

    Returns
    -------
    :class:`object`
        Retrieved attribute value.

    Examples
    --------
    >>> get_attribute("colour.models.oetf_inverse_BT2020")  # doctest: +ELLIPSIS
    <function oetf_inverse_BT2020 at 0x...>
    """

    attest("." in attribute, '"{0}" attribute has no namespace!')

    module_name, attribute = attribute.rsplit(".", 1)

    module = optional(sys.modules.get(module_name), import_module(module_name))

    attest(
        module is not None,
        f'"{module_name}" module does not exists or cannot be imported!',
    )

    return attrgetter(attribute)(module)


def build_API_changes(changes: dict) -> dict:
    """
    Build effective API changes from specified API changes mapping.

    Parameters
    ----------
    changes
        Dictionary of desired API changes.

    Returns
    -------
    :class:`dict`
        API changes

    Examples
    --------
    >>> from pprint import pprint
    >>> changes = {
    ...     "ObjectRenamed": [
    ...         [
    ...             "module.object_1_name",
    ...             "module.object_1_new_name",
    ...         ]
    ...     ],
    ...     "ObjectFutureRename": [
    ...         [
    ...             "module.object_2_name",
    ...             "module.object_2_new_name",
    ...         ]
    ...     ],
    ...     "ObjectFutureAccessChange": [
    ...         [
    ...             "module.object_3_access",
    ...             "module.sub_module.object_3_new_access",
    ...         ]
    ...     ],
    ...     "ObjectRemoved": ["module.object_4_name"],
    ...     "ObjectFutureRemove": ["module.object_5_name"],
    ...     "ObjectFutureAccessRemove": ["module.object_6_access"],
    ... }
    >>> pprint(build_API_changes(changes))  # doctest: +SKIP
    {'object_1_name': ObjectRenamed(name='module.object_1_name', \
new_name='module.object_1_new_name'),
     'object_2_name': ObjectFutureRename(name='module.object_2_name', \
new_name='module.object_2_new_name'),
     'object_3_access': ObjectFutureAccessChange(\
access='module.object_3_access', \
new_access='module.sub_module.object_3_new_access'),
     'object_4_name': ObjectRemoved(name='module.object_4_name'),
     'object_5_name': ObjectFutureRemove(name='module.object_5_name'),
     'object_6_access': ObjectFutureAccessRemove(\
name='module.object_6_access')}
    """

    for rename_type in (
        ObjectRenamed,
        ObjectFutureRename,
        ObjectFutureAccessChange,
        ArgumentRenamed,
        ArgumentFutureRename,
    ):
        for change in changes.pop(rename_type.__name__, []):
            changes[change[0].split(".")[-1]] = rename_type(*change)

    for remove_type in (
        ObjectRemoved,
        ObjectFutureRemove,
        ObjectFutureAccessRemove,
        ArgumentRemoved,
        ArgumentFutureRemove,
    ):
        for change in changes.pop(remove_type.__name__, []):
            changes[change.split(".")[-1]] = remove_type(change)

    return changes


def handle_arguments_deprecation(changes: dict, **kwargs: Any) -> dict:
    """
    Handle argument deprecation according to the specified API changes
    mapping.

    Parameters
    ----------
    changes
        Dictionary of specified API changes defining how arguments should
        be handled during deprecation.

    Other Parameters
    ----------------
    kwargs
        Keyword arguments to process for deprecation handling.

    Returns
    -------
    :class:`dict`
        Processed keyword arguments with deprecation rules applied.

    Examples
    --------
    >>> changes = {
    ...     "ArgumentRenamed": [
    ...         [
    ...             "argument_1_name",
    ...             "argument_1_new_name",
    ...         ]
    ...     ],
    ...     "ArgumentFutureRename": [
    ...         [
    ...             "argument_2_name",
    ...             "argument_2_new_name",
    ...         ]
    ...     ],
    ...     "ArgumentRemoved": ["argument_3_name"],
    ...     "ArgumentFutureRemove": ["argument_4_name"],
    ... }
    >>> handle_arguments_deprecation(
    ...     changes,
    ...     argument_1_name=True,
    ...     argument_2_name=True,
    ...     argument_4_name=True,
    ... )
    ... # doctest: +SKIP
    {'argument_4_name': True, 'argument_1_new_name': True, \
'argument_2_new_name': True}
    """

    changes = build_API_changes(changes)

    for kwarg in kwargs.copy():
        change = changes.get(kwarg)

        if change is None:
            continue

        if not isinstance(change, ArgumentRemoved):
            usage_warning(str(change))

            if isinstance(change, ArgumentFutureRemove):
                continue
            kwargs[change.values[1]] = kwargs.pop(kwarg)
        else:
            kwargs.pop(kwarg)
            usage_warning(str(change))

    return kwargs
