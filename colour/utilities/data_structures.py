# -*- coding: utf-8 -*-
"""
Data Structures
===============

Defines various data structures classes:

-   :class:`colour.utilities.Structure`: An object similar to C/C++ structured
    type.
-   :class:`colour.utilities.Lookup`: A *dict* sub-class acting as a lookup to
    retrieve keys by values.
-   :class:`colour.utilities.CaseInsensitiveMapping`: A case insensitive
    mapping allowing values retrieving from keys while ignoring the key case.

References
----------
-   :cite:`Mansencalc` : Mansencal, T. (n.d.). Lookup. Retrieved from
    https://github.com/KelSolaar/Foundations/blob/develop/foundations/\
data_structures.py
-   :cite:`Mansencald` : Mansencal, T. (n.d.). Structure. Retrieved from
    https://github.com/KelSolaar/Foundations/blob/develop/foundations/\
data_structures.py
-   :cite:`Reitza` : Reitz, K. (n.d.). CaseInsensitiveDict. Retrieved from
    https://github.com/kennethreitz/requests/blob/v1.2.3/requests/\
structures.py#L37
"""

from __future__ import division, unicode_literals

from collections import Mapping, MutableMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Structure', 'Lookup', 'CaseInsensitiveMapping']


class Structure(dict):
    """
    Defines an object similar to C/C++ structured type.

    Other Parameters
    ----------------
    \\*args : list, optional
        Arguments.
    \\**kwargs : dict, optional
        Key / Value pairs.


    References
    ----------
    :cite:`Mansencald`

    Examples
    --------
    >>> person = Structure(first_name='Doe', last_name='John', gender='male')
    >>> # Doctests skip for Python 2.x compatibility.
    >>> person.first_name  # doctest: +SKIP
    'Doe'
    >>> sorted(person.keys())
    ['first_name', 'gender', 'last_name']
    >>> # Doctests skip for Python 2.x compatibility.
    >>> person['gender']  # doctest: +SKIP
    'male'
    """

    def __init__(self, *args, **kwargs):
        super(Structure, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Lookup(dict):
    """
    Extends *dict* type to provide a lookup by value(s).

    Methods
    -------
    keys_from_value
    first_key_from_value

    References
    ----------
    :cite:`Mansencalc`

    Examples
    --------
    >>> person = Lookup(first_name='Doe', last_name='John', gender='male')
    >>> person.first_key_from_value('Doe')
    'first_name'
    >>> persons = Lookup(John='Doe', Jane='Doe', Luke='Skywalker')
    >>> sorted(persons.keys_from_value('Doe'))
    ['Jane', 'John']
    """

    def keys_from_value(self, value):
        """
        Gets the keys with given value.

        Parameters
        ----------
        value : object
            Value.
        Returns
        -------
        object
            Keys.
        """

        keys = []
        for key, data in self.items():
            matching = data == value
            try:
                matching = all(matching)

            except TypeError:
                matching = all((matching, ))

            if matching:
                keys.append(key)

        return keys

    def first_key_from_value(self, value):
        """
        Gets the first key with given value.

        Parameters
        ----------
        value : object
            Value.
        Returns
        -------
        object
            Key.
        """

        try:
            return self.keys_from_value(value)[0]
        except IndexError:
            pass


class CaseInsensitiveMapping(MutableMapping):
    """
    Implements a case-insensitive mutable mapping / *dict* object.

    Allows values retrieving from keys while ignoring the key case.
    The keys are expected to be unicode or string-like objects supporting the
    :meth:`str.lower` method.

    Parameters
    ----------
    data : dict
        *dict* of data to store into the mapping at initialisation.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Key / Value pairs to store into the mapping at initialisation.

    Methods
    -------
    __setitem__
    __getitem__
    __delitem__
    __contains__
    __iter__
    __len__
    __eq__
    __ne__
    __repr__
    copy
    lower_items

    Warning
    -------
    The keys are expected to be unicode or string-like objects.

    References
    ----------
    :cite:`Reitza`

    Examples
    --------
    >>> methods = CaseInsensitiveMapping({'McCamy': 1, 'Hernandez': 2})
    >>> methods['mccamy']
    1
    """

    def __init__(self, data=None, **kwargs):
        self._data = dict()

        self.update({} if data is None else data, **kwargs)

    @property
    def data(self):
        """
        Getter and setter property for the data.

        Parameters
        ----------
        value : dict
            Value to set the data with.

        Returns
        -------
        dict
            Data.
        """

        return self._data

    def __setitem__(self, item, value):
        """
        Sets given item with given value.

        The item is stored as lower in the mapping while the original name and
        its value are stored together as the value in a *tuple*:

        {"item.lower()": ("item", value)}

        Parameters
        ----------
        item : object
            Attribute.
        value : object
            Value.
        """

        self._data[item.lower()] = (item, value)

    def __getitem__(self, item):
        """
        Returns the value of given item.

        The item value is retrieved using its lower name in the mapping.

        Parameters
        ----------
        item : unicode
            Item name.

        Returns
        -------
        object
            Item value.
        """

        return self._data[item.lower()][1]

    def __delitem__(self, item):
        """
        Deletes the item with given name.

        The item is deleted from the mapping using its lower name.

        Parameters
        ----------
        item : unicode
            Item name.
        """

        del self._data[item.lower()]

    def __contains__(self, item):
        """
        Returns if the mapping contains given item.

        Parameters
        ----------
        item : unicode
            Item name.

        Returns
        -------
        bool
            Is item in mapping.
        """

        return item.lower() in self._data

    def __iter__(self):
        """
        Iterates over the items names in the mapping.

        The item names returned are the original input ones.

        Returns
        -------
        generator
            Item names.
        """

        return (item for item, value in self._data.values())

    def __len__(self):
        """
        Returns the items count.

        Returns
        -------
        int
            Items count.
        """

        return len(self._data)

    def __eq__(self, item):
        """
        Returns the equality with given object.

        Parameters
        ----------
        item
            Object item.

        Returns
        -------
        bool
            Equality.
        """

        if isinstance(item, Mapping):
            item = CaseInsensitiveMapping(item)
        else:
            return NotImplemented

        return dict(self.lower_items()) == dict(item.lower_items())

    def __ne__(self, item):
        """
        Returns the inequality with given object.

        Parameters
        ----------
        item
            Object item.

        Returns
        -------
        bool
            Inequality.
        """

        return not (self == item)

    def __repr__(self):
        """
        Returns the mapping representation with the original item names.

        Returns
        -------
        unicode
            Mapping representation.
        """

        return '{0}({1})'.format(self.__class__.__name__, dict(self.items()))

    def copy(self):
        """
        Returns a copy of the mapping.

        Returns
        -------
        CaseInsensitiveMapping
            Mapping copy.

        Notes
        -----
        -   The :class:`colour.utilities.CaseInsensitiveMapping` class copy
            returned is a simple *copy* not a *deepcopy*.
        """

        return CaseInsensitiveMapping(self._data.values())

    def lower_items(self):
        """
        Iterates over the lower items names.

        Returns
        -------
        generator
            Lower item names.
        """

        return ((item, value[1]) for (item, value) in self._data.items())
