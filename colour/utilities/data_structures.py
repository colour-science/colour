#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**data_structures.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package data structures objects.

**Others:**

"""

from __future__ import unicode_literals


__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["Structure",
           "Lookup"]


class Structure(dict):
    """
    Defines an object similar to C/C++ structured type.

    Usage:

        >>> person = Structure(firstName="Doe", lastName="John", gender="male")
        >>> person.firstName
        'Doe'
        >>> person.keys()
        ['gender', 'firstName', 'lastName']
        >>> person["gender"]
        'male'
        >>> del(person["gender"])
        >>> person["gender"]
        Traceback (most recent call last):
          File "<console>", line 1, in <module>
        KeyError: 'gender'
        >>> person.gender
        Traceback (most recent call last):
          File "<console>", line 1, in <module>
        AttributeError: 'Structure' object has no attribute 'gender'

    References:

    -  https://github.com/KelSolaar/Foundations/blob/develop/foundations/data_structures.py
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the class.

        :param \*args: Arguments.
        :type \*args: \*
        :param \*\*kwargs: Key / Value pairs.
        :type \*\*kwargs: dict
        """

        dict.__init__(self, **kwargs)
        self.__dict__.update(**kwargs)

    def __getattr__(self, attribute):
        """
        Returns given attribute value.

        :return: Attribute value.
        :rtype: object
        """

        try:
            return dict.__getitem__(self, attribute)
        except KeyError:
            raise AttributeError("'{0}' object has no attribute '{1}'".format(
                self.__class__.__name__, attribute))

    def __setattr__(self, attribute, value):
        """
        Sets both key and sibling attribute with given value.

        :param attribute: Attribute.
        :type attribute: object
        :param value: Value.
        :type value: object
        """

        dict.__setitem__(self, attribute, value)
        object.__setattr__(self, attribute, value)

    __setitem__ = __setattr__

    def __delattr__(self, attribute):
        """
        Deletes both key and sibling attribute.

        :param attribute: Attribute.
        :type attribute: object
        """

        dict.__delitem__(self, attribute)
        object.__delattr__(self, attribute)

    __delitem__ = __delattr__

    def update(self, *args, **kwargs):
        """
        Reimplements the :meth:`Dict.update` method.

        :param \*args: Arguments.
        :type \*args: \*
        :param \*\*kwargs: Keywords arguments.
        :type \*\*kwargs: \*\*
        """

        dict.update(self, *args, **kwargs)
        self.__dict__.update(*args, **kwargs)


class Lookup(dict):
    """
    Extends dict type to provide a lookup by value(s).

    Usage:

        >>> person = Lookup(firstName="Doe", lastName="John", gender="male")
        >>> person.get_first_key_from_value("Doe")
        'firstName'
        >>> persons = Lookup(John="Doe", Jane="Doe", Luke="Skywalker")
        >>> persons.get_keys_from_value("Doe")
        ['Jane', 'John']

    References:

    -  https://github.com/KelSolaar/Foundations/blob/develop/foundations/data_structures.py
    """

    def get_first_key_from_value(self, value):
        """
        Gets the first key from given value.

        :param value: Value.
        :type value: object
        :return: Key.
        :rtype: object
        """

        for key, data in self.items():
            if data == value:
                return key

    def get_keys_from_value(self, value):
        """
        Gets the keys from given value.

        :param value: Value.
        :type value: object
        :return: Keys.
        :rtype: object
        """

        return [key for key, data in self.items() if data == value]