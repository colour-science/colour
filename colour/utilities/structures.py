"""
Data Structures
===============

Define various data structures classes:

-   :class:`colour.utilities.Structure`: An object similar to C/C++ structured
    type.
-   :class:`colour.utilities.Lookup`: A :class:`dict` sub-class acting as a
    lookup to retrieve keys by values.
-   :class:`colour.utilities.CanonicalMapping`: A delimiter and
    case-insensitive :class:`dict`-like object allowing values retrieving from
    keys while ignoring the key case.
-   :class:`colour.utilities.LazyCanonicalMapping`: Another delimiter and
    case-insensitive mapping allowing lazy values retrieving from keys while
    ignoring the key case.

References
----------
-   :cite:`Mansencalc` : Mansencal, T. (n.d.). Lookup.
    https://github.com/KelSolaar/Foundations/blob/develop/foundations/\
structures.py
-   :cite:`Rakotoarison2017` : Rakotoarison, H. (2017). Bunch.
    https://github.com/scikit-learn/scikit-learn/blob/\
fb5a498d0bd00fc2b42fbd19b6ef18e1dfeee47e/sklearn/utils/__init__.py#L65
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import MutableMapping

from colour.hints import (
    Any,
    Generator,
    Iterable,
    Mapping,
)
from colour.utilities.documentation import is_documentation_building

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Structure",
    "Lookup",
    "CanonicalMapping",
    "LazyCanonicalMapping",
]


class Structure(dict):
    """
    Define a :class:`dict`-like object allowing to access key values using dot
    syntax.

    Other Parameters
    ----------------
    args
        Arguments.
    kwargs
        Key / value pairs.

    Methods
    -------
    -   :meth:`~colour.utilities.Structure.__init__`
    -   :meth:`~colour.utilities.Structure.__setattr__`
    -   :meth:`~colour.utilities.Structure.__delattr__`
    -   :meth:`~colour.utilities.Structure.__dir__`
    -   :meth:`~colour.utilities.Structure.__getattr__`
    -   :meth:`~colour.utilities.Structure.__setstate__`

    References
    ----------
    :cite:`Rakotoarison2017`

    Examples
    --------
    >>> person = Structure(first_name="John", last_name="Doe", gender="male")
    >>> person.first_name
    'John'
    >>> sorted(person.keys())
    ['first_name', 'gender', 'last_name']
    >>> person["gender"]
    'male'
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __setattr__(self, name: str, value: Any):
        """
        Assign given value to the attribute with given name.

        Parameters
        ----------
        name
            Name of the attribute to assign the ``value`` to.
        value
            Value to assign to the attribute.
        """

        self[name] = value

    def __delattr__(self, name: str):
        """
        Delete the attribute with given name.

        Parameters
        ----------
        name
            Name of the attribute to delete.
        """

        del self[name]

    def __dir__(self) -> Iterable:
        """
        Return a list of valid attributes for the :class:`dict`-like object.

        Returns
        -------
        :class:`list`
            List of valid attributes for the :class:`dict`-like object.
        """

        return self.keys()

    def __getattr__(self, name: str) -> Any:
        """
        Return the value from the attribute with given name.

        Parameters
        ----------
        name
            Name of the attribute to get the value from.

        Returns
        -------
        :class:`object`

        Raises
        ------
        AttributeError
            If the attribute is not defined.
        """

        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setstate__(self, state):
        """Set the object state when unpickling."""
        # See https://github.com/scikit-learn/scikit-learn/issues/6196 for more
        # information.


class Lookup(dict):
    """
    Extend :class:`dict` type to provide a lookup by value(s).

    Methods
    -------
    -   :meth:`~colour.utilities.Lookup.keys_from_value`
    -   :meth:`~colour.utilities.Lookup.first_key_from_value`

    References
    ----------
    :cite:`Mansencalc`

    Examples
    --------
    >>> person = Lookup(first_name="John", last_name="Doe", gender="male")
    >>> person.first_key_from_value("John")
    'first_name'
    >>> persons = Lookup(John="Doe", Jane="Doe", Luke="Skywalker")
    >>> sorted(persons.keys_from_value("Doe"))
    ['Jane', 'John']
    """

    def keys_from_value(self, value: Any) -> list:
        """
        Get the keys associated with given value.

        Parameters
        ----------
        value
            Value to find the associated keys.

        Returns
        -------
        :class:`list`
            Keys associated with given value.
        """

        keys = []
        for key, data in self.items():
            matching = data == value
            try:
                matching = all(matching)

            except TypeError:
                matching = all((matching,))

            if matching:
                keys.append(key)

        return keys

    def first_key_from_value(self, value: Any) -> Any:
        """
        Get the first key associated with given value.

        Parameters
        ----------
        value
            Value to find the associated first key.

        Returns
        -------
        :class:`object`
            First key associated with given value.
        """

        return self.keys_from_value(value)[0]


class CanonicalMapping(MutableMapping):
    """
    Implement a delimiter and case-insensitive :class:`dict`-like object with
    support for slugs, i.e. *SEO* friendly and human-readable version of the
    keys but also canonical keys, i.e. slugified keys without delimiters.

    The item keys are expected to be :class:`str`-like objects thus supporting
    the :meth:`str.lower` method. Setting items is done by using the given
    keys. Retrieving or deleting an item and testing whether an item exist is
    done by transforming the item's key in a sequence as follows:

    -   *Original Key*
    -   *Lowercase Key*
    -   *Slugified Key*
    -   *Canonical Key*

    For example, given the ``McCamy 1992`` key:

    -   *Original Key* : ``McCamy 1992``
    -   *Lowercase Key* : ``mccamy 1992``
    -   *Slugified Key* : ``mccamy-1992``
    -   *Canonical Key* : ``mccamy1992``

    Parameters
    ----------
    data
        Data to store into the delimiter and case-insensitive
        :class:`dict`-like object at initialisation.

    Other Parameters
    ----------------
    kwargs
        Key / value pairs to store into the mapping at initialisation.

    Attributes
    ----------
    -   :attr:`~colour.utilities.CanonicalMapping.data`

    Methods
    -------
    -   :meth:`~colour.utilities.CanonicalMapping.__init__`
    -   :meth:`~colour.utilities.CanonicalMapping.__repr__`
    -   :meth:`~colour.utilities.CanonicalMapping.__setitem__`
    -   :meth:`~colour.utilities.CanonicalMapping.__getitem__`
    -   :meth:`~colour.utilities.CanonicalMapping.__delitem__`
    -   :meth:`~colour.utilities.CanonicalMapping.__contains__`
    -   :meth:`~colour.utilities.CanonicalMapping.__iter__`
    -   :meth:`~colour.utilities.CanonicalMapping.__len__`
    -   :meth:`~colour.utilities.CanonicalMapping.__eq__`
    -   :meth:`~colour.utilities.CanonicalMapping.__ne__`
    -   :meth:`~colour.utilities.CanonicalMapping.copy`
    -   :meth:`~colour.utilities.CanonicalMapping.lower_keys`
    -   :meth:`~colour.utilities.CanonicalMapping.lower_items`
    -   :meth:`~colour.utilities.CanonicalMapping.slugified_keys`
    -   :meth:`~colour.utilities.CanonicalMapping.slugified_items`
    -   :meth:`~colour.utilities.CanonicalMapping.canonical_keys`
    -   :meth:`~colour.utilities.CanonicalMapping.canonical_items`

    Examples
    --------
    >>> methods = CanonicalMapping({"McCamy 1992": 1, "Hernandez 1999": 2})
    >>> methods["mccamy 1992"]
    1
    >>> methods["MCCAMY 1992"]
    1
    >>> methods["mccamy-1992"]
    1
    >>> methods["mccamy1992"]
    1
    """

    def __init__(self, data: Generator | Mapping | None = None, **kwargs: Any) -> None:
        self._data: dict = {}

        self.update({} if data is None else data, **kwargs)

    @property
    def data(self) -> dict:
        """
        Getter property for the delimiter and case-insensitive
        :class:`dict`-like object data.

        Returns
        -------
        :class:`dict`
            Data.
        """

        return self._data

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the delimiter and
        case-insensitive :class:`dict`-like object.

        Returns
        -------
        :class:`str`
            Evaluable string representation.
        """

        if is_documentation_building():  # pragma: no cover
            representation = repr(dict(zip(self.keys(), ["..."] * len(self)))).replace(
                "'...'", "..."
            )
            return f"{self.__class__.__name__}({representation})"
        else:
            return f"{self.__class__.__name__}({dict(self.items())})"

    def __setitem__(self, item: str | Any, value: Any):
        """
        Set given item with given value in the delimiter and case-insensitive
        :class:`dict`-like object.

        Parameters
        ----------
        item
            Item to set in the delimiter and case-insensitive
            :class:`dict`-like object.
        value
            Value to store in the delimiter and case-insensitive
            :class:`dict`-like object.
        """

        self._data[item] = value

    def __getitem__(self, item: str | Any) -> Any:
        """
        Return the value of given item from the delimiter and case-insensitive
        :class:`dict`-like object.

        Parameters
        ----------
        item
            Item to retrieve the value of from the delimiter and
            case-insensitive :class:`dict`-like object.

        Returns
        -------
        :class:`object`
            Item value.

        Notes
        -----
        -   The item value can be retrieved by using either its lower-case,
            slugified or canonical variant.
        """

        try:
            return self._data[item]
        except KeyError:
            pass

        try:
            return self[dict(zip(self.lower_keys(), self.keys()))[str(item).lower()]]
        except KeyError:
            pass

        try:
            return self[dict(zip(self.slugified_keys(), self.keys()))[item]]
        except KeyError:
            pass

        return self[dict(zip(self.canonical_keys(), self.keys()))[item]]

    def __delitem__(self, item: str | Any):
        """
        Delete given item from the delimiter and case-insensitive
        :class:`dict`-like object.

        Parameters
        ----------
        item
            Item to delete from the delimiter and case-insensitive
            :class:`dict`-like object.

        Notes
        -----
        -   The item can be deleted by using either its lower-case, slugified
            or canonical variant.
        """

        try:
            del self._data[item]
            return
        except KeyError:
            pass

        try:
            del self._data[dict(zip(self.lower_keys(), self.keys()))[str(item).lower()]]
            return
        except KeyError:
            pass

        try:
            del self[dict(zip(self.slugified_keys(), self.keys()))[item]]
            return
        except KeyError:
            pass

        del self[dict(zip(self.canonical_keys(), self.keys()))[item]]

    def __contains__(self, item: str | Any) -> bool:
        """
        Return whether the delimiter and case-insensitive :class:`dict`-like
        object contains given item.

        Parameters
        ----------
        item
            Item to find whether it is in the delimiter and case-insensitive
            :class:`dict`-like object.

        Returns
        -------
        :class:`bool`
            Whether given item is in the delimiter and case-insensitive
            :class:`dict`-like object.

        Notes
        -----
        -   The item presence can be checked by using either its lower-case,
            slugified or canonical variant.
        """

        return bool(
            any(
                [
                    item in self._data,
                    str(item).lower() in self.lower_keys(),
                    item in self.slugified_keys(),
                    item in self.canonical_keys(),
                ]
            )
        )

    def __iter__(self) -> Generator:
        """
        Iterate over the items of the delimiter and case-insensitive
        :class:`dict`-like object.

        Yields
        ------
        Generator
            Item generator.

        Notes
        -----
        -   The iterated items are the original items.
        """

        yield from self._data.keys()

    def __len__(self) -> int:
        """
        Return the items count.

        Returns
        -------
        :class:`int`
            Items count.
        """

        return len(self._data)

    def __eq__(self, other: Any) -> bool:
        """
        Return whether the delimiter and case-insensitive :class:`dict`-like
        object is equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is equal to the delimiter and
            case-insensitive :class:`dict`-like object

        Returns
        -------
        :class:`bool`
            Whether given object is equal to the delimiter and case-insensitive
            :class:`dict`-like object.
        """

        if isinstance(other, Mapping):
            other_mapping = CanonicalMapping(other)
        else:
            raise TypeError(
                f"Impossible to test equality with "
                f'"{other.__class__.__name__}" class type!'
            )

        return self._data == other_mapping.data

    def __ne__(self, other: Any) -> bool:
        """
        Return whether the delimiter and case-insensitive :class:`dict`-like
        object is not equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is not equal to the delimiter and
            case-insensitive :class:`dict`-like object

        Returns
        -------
        :class:`bool`
            Whether given object is not equal to the delimiter and
            case-insensitive :class:`dict`-like object.
        """

        return not (self == other)

    @staticmethod
    def _collision_warning(keys: list):
        """
        Issue a runtime warning when given keys are colliding.

        Parameters
        ----------
        keys
        """

        from colour.utilities import usage_warning

        collisions = [key for (key, value) in Counter(keys).items() if value > 1]

        if collisions:
            usage_warning(f"{list(set(keys))} key(s) collide(s)!")

    def copy(self) -> CanonicalMapping:
        """
        Return a copy of the delimiter and case-insensitive :class:`dict`-like
        object.

        Returns
        -------
        :class:`CanonicalMapping`
            Case-insensitive :class:`dict`-like object copy.

        Warnings
        --------
        -   The :class:`CanonicalMapping` class copy returned is a
            *copy* of the object not a *deepcopy*!
        """

        return CanonicalMapping(dict(**self._data))

    def lower_keys(self) -> Generator:
        """
        Iterate over the lower-case keys of the delimiter and case-insensitive
        :class:`dict`-like object.

        Yields
        ------
        Generator
            Item generator.
        """

        lower_keys = [str(key).lower() for key in self._data]

        self._collision_warning(lower_keys)

        yield from iter(lower_keys)

    def lower_items(self) -> Generator:
        """
        Iterate over the lower-case items of the delimiter and case-insensitive
        :class:`dict`-like object.

        Yields
        ------
        Generator
            Item generator.
        """

        yield from ((str(key).lower(), value) for (key, value) in self._data.items())

    def slugified_keys(self) -> Generator:
        """
        Iterate over the slugified keys of the delimiter and case-insensitive
        :class:`dict`-like object.

        Yields
        ------
        Generator
            Item generator.
        """

        from colour.utilities import slugify

        slugified_keys = [slugify(key) for key in self.lower_keys()]

        self._collision_warning(slugified_keys)

        yield from iter(slugified_keys)

    def slugified_items(self) -> Generator:
        """
        Iterate over the slugified items of the delimiter and case-insensitive
        :class:`dict`-like object.

        Yields
        ------
        Generator
            Item generator.
        """

        yield from zip(self.slugified_keys(), self.values())

    def canonical_keys(self) -> Generator:
        """
        Iterate over the canonical keys of the delimiter and case-insensitive
        :class:`dict`-like object.

        Yields
        ------
        Generator
            Item generator.
        """

        canonical_keys = [re.sub("-|_", "", key) for key in self.slugified_keys()]

        self._collision_warning(canonical_keys)

        yield from iter(canonical_keys)

    def canonical_items(self) -> Generator:
        """
        Iterate over the canonical items of the delimiter and case-insensitive
        :class:`dict`-like object.

        Yields
        ------
        Generator
            Item generator.
        """

        yield from zip(self.canonical_keys(), self.values())


class LazyCanonicalMapping(CanonicalMapping):
    """
    Implement a lazy delimiter and case-insensitive :class:`dict`-like object
    inheriting from :class:`colour.utilities.CanonicalMapping` class.

    The lazy retrieval is performed as follows: If the value is a callable,
    then it is evaluated and its return value is stored in place of the current
    value.

    Parameters
    ----------
    data
        Data to store into the lazy delimiter and case-insensitive
        :class:`dict`-like object at initialisation.

    Other Parameters
    ----------------
    kwargs
        Key / value pairs to store into the mapping at initialisation.

    Methods
    -------
    -   :meth:`~colour.utilities.LazyCanonicalMapping.__getitem__`

    Examples
    --------
    >>> def callable_a():
    ...     print(2)
    ...     return 2
    >>> methods = LazyCanonicalMapping({"McCamy": 1, "Hernandez": callable_a})
    >>> methods["mccamy"]
    1
    >>> methods["hernandez"]
    2
    2
    """

    def __getitem__(self, item: str | Any) -> Any:
        """
        Return the value of given item from the lazy delimiter and
        case-insensitive :class:`dict`-like object.

        Parameters
        ----------
        item
            Item to retrieve the value of from the lazy delimiter and
            case-insensitive :class:`dict`-like object.

        Returns
        -------
        :class:`object`
            Item value.
        """

        import colour

        value = super().__getitem__(item)

        if callable(value) and hasattr(colour, "__disable_lazy_load__"):
            value = value()
            super().__setitem__(item, value)

        return value
