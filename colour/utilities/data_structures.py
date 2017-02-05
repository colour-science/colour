#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Structures
===============

Defines various data structures classes:

-   :class:`ArbitraryPrecisionMapping`: A mutable mapping / *dict* like object
    where numeric keys are stored with an arbitrary precision.
-   :class:`Structure`: An object similar to C/C++ structured type.
-   :class:`Lookup`: A *dict* sub-class acting as a lookup to retrieve keys by
    values.
-   :class:`CaseInsensitiveMapping`: A case insensitive mapping allowing values
    retrieving from keys while ignoring the key case.
"""

from __future__ import division, unicode_literals

from collections import Mapping, MutableMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ArbitraryPrecisionMapping',
           'Structure',
           'Lookup',
           'CaseInsensitiveMapping']


class ArbitraryPrecisionMapping(MutableMapping):
    """
    Implements a mutable mapping / *dict* like object where numeric keys are
    stored with an arbitrary precision.

    Parameters
    ----------
    data : dict, optional
        *dict* of data to store into the mapping at initialisation.
    key_decimals : int, optional
        Decimals count the keys will be rounded at

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        Key / Value pairs to store into the mapping at initialisation.

    Attributes
    ----------
    key_decimals

    Methods
    -------
    __setitem__
    __getitem__
    __delitem__
    __contains__
    __iter__
    __len__

    Examples
    --------
    >>> data1 = {0.1999999998: 'Nemo', 0.2000000000: 'John'}
    >>> apm1 = ArbitraryPrecisionMapping(data1, key_decimals=10)
    >>> # Doctests skip for Python 2.x compatibility.
    >>> tuple(apm1.keys())  # doctest: +SKIP
    (0.1999999998, 0.2)
    >>> apm2 = ArbitraryPrecisionMapping(data1, key_decimals=7)
    >>> # Doctests skip for Python 2.x compatibility.
    >>> tuple(apm2.keys())  # doctest: +SKIP
    (0.2,)
    """

    def __init__(self, data=None, key_decimals=0, **kwargs):
        self._data = dict()
        self._key_decimals = None
        self.key_decimals = key_decimals

        self.update({} if data is None else data, **kwargs)

    @property
    def data(self):
        """
        Property for **self.data** attribute.

        Returns
        -------
        dict
            :class:`ArbitraryPrecisionMapping` data structure.

        Warning
        -------
        :attr:`ArbitraryPrecisionMapping.data` is read only.
        """

        return self._data

    @data.setter
    def data(self, value):
        """
        Setter for **self.data** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('data'))

    @property
    def key_decimals(self):
        """
        Property for **self._key_decimals** private attribute.

        Returns
        -------
        unicode
            self._key_decimals.
        """

        return self._key_decimals

    @key_decimals.setter
    def key_decimals(self, value):
        """
        Setter for **self._key_decimals** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, int), (
                '"{0}" attribute: "{1}" is not a "int" instance!').format(
                'key_decimals', value)
        self._key_decimals = value

    def _round(self, item):
        """
        Rounds given item if numeric.

        Parameters
        ----------
        item : object
            Attribute.

        Parameters
        ----------
        item : object
            Attribute.

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__setitem__` method.
        """

        try:
            return round(item, self._key_decimals)
        except TypeError:
            return item

    def __setitem__(self, item, value):
        """
        Sets given item (rounded if numeric) with given value.

        Parameters
        ----------
        item : object
            Attribute.
        value : object
            Value.

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__setitem__` method.
        """

        self._data[self._round(item)] = value

    def __getitem__(self, item):
        """
        Returns the value of given item (rounded if numeric).

        Parameters
        ----------
        item : unicode
            Item (rounded if numeric) name.

        Returns
        -------
        object
            Item value.

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__getitem__` method.
        """

        return self._data[self._round(item)]

    def __delitem__(self, item):
        """
        Deletes the item (rounded if numeric) with given value.

        Parameters
        ----------
        item : unicode
            Item (rounded if numeric) name.

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__delitem__` method.
        """

        del self._data[self._round(item)]

    def __contains__(self, item):
        """
        Returns if the mapping contains given item (rounded if numeric).

        Parameters
        ----------
        item : unicode
            Item (rounded if numeric) name.

        Returns
        -------
        bool
            Is item in mapping.

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__contains__` method.
        """

        return self._round(item) in self._data

    def __iter__(self):
        """
        Iterates over the items (rounded if numeric) names in the mapping.

        Returns
        -------
        generator
            Item names.

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__iter__` method.
        """

        return iter(self._data)

    def __len__(self):
        """
        Returns the items count.

        Returns
        -------
        int
            Items count.

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__iter__` method.
        """

        return len(self._data)


class Structure(dict):
    """
    Defines an object similar to C/C++ structured type.

    Other Parameters
    ----------------
    \*args : list, optional
        Arguments.
    \**kwargs : dict, optional
        Key / Value pairs.

    Methods
    -------
    __getattr__
    __setattr__
    __delattr__
    update

    References
    ----------
    .. [1]  Mansencal, T. (n.d.). Structure. Retrieved from
            https://github.com/KelSolaar/Foundations/\
blob/develop/foundations/data_structures.py

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
        dict.__init__(self, **kwargs)
        self.__dict__.update(**kwargs)

    def __getattr__(self, attribute):
        """
        Returns given attribute value.

        Parameters
        ----------
        attribute : unicode
            Attribute name.

        Notes
        -----
        -   Reimplements the :meth:`dict.__getattr__` method.

        Returns
        -------
        object
            Attribute value.

        Raises
        ------
        AttributeError
            If the attribute is not defined.
        """

        try:
            return dict.__getitem__(self, attribute)
        except KeyError:
            raise AttributeError('"{0}" object has no attribute "{1}"'.format(
                self.__class__.__name__, attribute))

    def __setattr__(self, attribute, value):
        """
        Sets both key and sibling attribute with given value.

        Parameters
        ----------
        attribute : object
            Attribute.
        value : object
            Value.

        Notes
        -----
        -   Reimplements the :meth:`dict.__setattr__` method.
        """

        dict.__setitem__(self, attribute, value)
        object.__setattr__(self, attribute, value)

    __setitem__ = __setattr__

    def __delattr__(self, attribute):
        """
        Deletes both key and sibling attribute.

        Parameters
        ----------
        attribute : object
            Attribute.

        Notes
        -----
        -   Reimplements the :meth:`dict.__delattr__` method.
        """

        dict.__delitem__(self, attribute)
        object.__delattr__(self, attribute)

    __delitem__ = __delattr__

    def update(self, *args, **kwargs):
        """
        Updates both keys and sibling attributes.

        Other Parameters
        ----------------
        \*args : list, optional
            Arguments.
        \**kwargs : dict, optional
            Keywords arguments.

        Notes
        -----
        -   Reimplements the :meth:`dict.update` method.
        """

        dict.update(self, *args, **kwargs)
        self.__dict__.update(*args, **kwargs)


class Lookup(dict):
    """
    Extends *dict* type to provide a lookup by value(s).

    Methods
    -------
    first_key_from_value
    keys_from_value

    References
    ----------
    .. [2]  Mansencal, T. (n.d.). Lookup. Retrieved from
            https://github.com/KelSolaar/Foundations/\
blob/develop/foundations/data_structures.py

    Examples
    --------
    >>> person = Lookup(first_name='Doe', last_name='John', gender='male')
    >>> person.first_key_from_value('Doe')
    'first_name'
    >>> persons = Lookup(John='Doe', Jane='Doe', Luke='Skywalker')
    >>> sorted(persons.keys_from_value('Doe'))
    ['Jane', 'John']
    """

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

        for key, data in self.items():
            if data == value:
                return key

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

        return [key for key, data in self.items() if data == value]


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
    \**kwargs : dict, optional
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
    .. [3]  Reitz, K. (n.d.). CaseInsensitiveDict. Retrieved from
            https://github.com/kennethreitz/requests/\
blob/v1.2.3/requests/structures.py#L37

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
        Property for **self.data** attribute.

        Returns
        -------
        dict
            :class:`ArbitraryPrecisionMapping` data structure.

        Warning
        -------
        :attr:`ArbitraryPrecisionMapping.data` is read only.
        """

        return self._data

    @data.setter
    def data(self, value):
        """
        Setter for **self.data** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('data'))

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

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__setitem__` method.
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

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__getitem__` method.
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

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__delitem__` method.
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

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__contains__` method.
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

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__iter__` method.
        """

        return (item for item, value in self._data.values())

    def __len__(self):
        """
        Returns the items count.

        Returns
        -------
        int
            Items count.

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__iter__` method.
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

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__eq__` method.
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

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__ne__` method.
        """

        return not (self == item)

    def __repr__(self):
        """
        Returns the mapping representation with the original item names.

        Returns
        -------
        unicode
            Mapping representation.

        Notes
        -----
        -   Reimplements the :meth:`MutableMapping.__repr__` method.
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
        -   The :class:`CaseInsensitiveMapping` class copy returned is a simple
            *copy* not a *deepcopy*.
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
