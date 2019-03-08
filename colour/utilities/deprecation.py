# -*- coding: utf-8 -*-
"""
Deprecation Utilities
=====================

Defines various deprecations management related objects.
"""

from __future__ import division, unicode_literals

import sys
from importlib import import_module
from collections import namedtuple
from operator import attrgetter

from colour.utilities import usage_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'Renamed', 'Removed', 'FutureRename', 'FutureRemove', 'FutureAccessChange',
    'FutureAccessRemove', 'ModuleAPI', 'get_attribute'
]


class Renamed(namedtuple('Renamed', ('name', 'new_name'))):
    """
    A class used for an object that has been renamed.

    Parameters
    ----------
    name : unicode
        Object name that changed.
    new_name : unicode
        Object new name.
    """

    def __str__(self):
        """
        Returns a formatted string representation of the class.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return ('"{0}" object has been renamed to "{1}".'.format(
            self.name, self.new_name))


class Removed(namedtuple('Removed', ('name', ))):
    """
    A class used for an object that has been removed.

    Parameters
    ----------
    name : unicode
        Object name that has been removed.
    """

    def __str__(self):
        """
        Returns a formatted string representation of the class.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return '"{0}" object has been removed from the API.'.format(self.name)


class FutureRename(namedtuple('FutureRename', ('name', 'new_name'))):
    """
    A class used for future object name deprecation, i.e. object name will
    change in a future release.

    Parameters
    ----------
    name : unicode
        Object name that will change in a future release.
    new_name : unicode
        Object future release name.
    """

    def __str__(self):
        """
        Returns a formatted string representation of the deprecation type.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return ('"{0}" object is deprecated and will be renamed to "{1}" '
                'in a future release.'.format(self.name, self.new_name))


class FutureRemove(namedtuple('FutureRemove', ('name', 'access'))):
    """
    A class used for future object removal.

    Parameters
    ----------
    name : unicode
        Object name that will be removed in a future release.
    access : unicode
        Object current access.
    """

    def __str__(self):
        """
        Returns a formatted string representation of the deprecation type.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return ('"{0}" object is deprecated and will be removed '
                'in a future release.'.format(self.name))


class FutureAccessChange(
        namedtuple('FutureAccessChange', ('access', 'new_access'))):
    """
    A class used for future object access deprecation, i.e. object access will
    change in a future release.

    Parameters
    ----------
    access : unicode
        Object access that will change in a future release.
    new_access : unicode
        Object future release access.
    """

    def __str__(self):
        """
        Returns a formatted string representation of the deprecation type.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return ('"{0}" object access is deprecated and will change to '
                '"{1}" in a future release.'.format(self.access,
                                                    self.new_access))


class FutureAccessRemove(namedtuple('FutureAccessRemove', ('name', 'access'))):
    """
    A class used for future object access removal, i.e. object access will
    be removed in a future release.

    Parameters
    ----------
    name : unicode
        Object name whose access will removed in a future release.
    access : unicode
        Object access that will be removed in a future release.
    """

    def __str__(self):
        """
        Returns a formatted string representation of the deprecation type.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return (
            '"{0}" object access will be removed in a future release.'.format(
                self.name))


class ModuleAPI(object):
    """
    Define a class that allows customisation of module attributes access with
    deprecation management.

    Parameters
    ----------
    module : module
        Module to customise attributes access.

    Methods
    -------
    __init__
    __getattr__
    __dir__

    Examples
    --------
    >>> import sys
    >>> sys.modules['colour'] = ModuleAPI(sys.modules['colour'])
    ... # doctest: +SKIP
    """

    def __init__(self, module, changes=None):
        self._module = module
        self._changes = changes or {}

    def __getattr__(self, attribute):
        """
        Returns given attribute value while handling deprecation.

        Parameters
        ----------
        attribute : unicode
            Attribute name.

        Returns
        -------
        object
            Attribute value.

        Raises
        ------
        AttributeError
            If the attribute is not defined.
        """

        change = self._changes.get(attribute)
        if change is not None:
            if not isinstance(change, Removed):
                usage_warning(str(change))
                return get_attribute(change[1])
            else:
                raise AttributeError(str(change))

        return getattr(self._module, attribute)

    def __dir__(self):
        """
        Returns list of names in the module local scope filtered according to
        the changes.

        Returns
        -------
        list
            Filtered list of names in the module local scope.
        """

        attributes = [
            attribute for attribute in dir(self._module)
            if attribute not in self._changes
        ]

        return attributes


def get_attribute(attribute):
    """
    Returns given attribute.

    Parameters
    ----------
    attribute : unicode
        Attribute to retrieve, ``attribute`` must have a namespace module, e.g.
        *colour.models.eotf_BT2020*.

    Returns
    -------
    object
        Retrieved attribute.

    Examples
    --------
    >>> get_attribute('colour.models.eotf_BT2020')  # doctest: +ELLIPSIS
    <function eotf_BT2020 at 0x...>
    """

    assert '.' in attribute, '"{0}" attribute has no namespace!'

    module, attribute = attribute.split('.', 1)

    module = sys.modules.get(module)
    if module is None:
        module = import_module(module)

    assert module is not None, (
        '"{0}" module does not exists or is not imported!')

    return attrgetter(attribute)(module)
