# -*- coding: utf-8 -*-
"""
Deprecation Utilities
=====================

Defines various deprecation management related objects.
"""

from __future__ import division, unicode_literals

import sys
from importlib import import_module
from collections import namedtuple
from operator import attrgetter

from colour.utilities import usage_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'ObjectRenamed', 'ObjectRemoved', 'ObjectFutureRename',
    'ObjectFutureRemove', 'ObjectFutureAccessChange',
    'ObjectFutureAccessRemove', 'ModuleAPI', 'ArgumentRenamed',
    'ArgumentRemoved', 'ArgumentFutureRename', 'ArgumentFutureRemove',
    'get_attribute', 'build_API_changes', 'handle_arguments_deprecation'
]


class ObjectRenamed(namedtuple('ObjectRenamed', ('name', 'new_name'))):
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


class ObjectRemoved(namedtuple('ObjectRemoved', ('name', ))):
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


class ObjectFutureRename(
        namedtuple('ObjectFutureRename', ('name', 'new_name'))):
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


class ObjectFutureRemove(namedtuple('ObjectFutureRemove', ('name', ))):
    """
    A class used for future object removal.

    Parameters
    ----------
    name : unicode
        Object name that will be removed in a future release.
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


class ObjectFutureAccessChange(
        namedtuple('ObjectFutureAccessChange', ('access', 'new_access'))):
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


class ObjectFutureAccessRemove(
        namedtuple('ObjectFutureAccessRemove', ('name', ))):
    """
    A class used for future object access removal, i.e. object access will
    be removed in a future release.

    Parameters
    ----------
    name : unicode
        Object name whose access will removed in a future release.
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


class ArgumentRenamed(namedtuple('ArgumentRenamed', ('name', 'new_name'))):
    """
    A class used for an argument that has been renamed.

    Parameters
    ----------
    name : unicode
        Argument name that changed.
    new_name : unicode
        Argument new name.
    """

    def __str__(self):
        """
        Returns a formatted string representation of the class.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return ('"{0}" argument has been renamed to "{1}".'.format(
            self.name, self.new_name))


class ArgumentRemoved(namedtuple('ArgumentRemoved', ('name', ))):
    """
    A class used for an argument that has been removed.

    Parameters
    ----------
    name : unicode
        Argument name that has been removed.
    """

    def __str__(self):
        """
        Returns a formatted string representation of the class.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return '"{0}" argument has been removed from the API.'.format(
            self.name)


class ArgumentFutureRename(
        namedtuple('ArgumentFutureRename', ('name', 'new_name'))):
    """
    A class used for future argument name deprecation, i.e. argument name will
    change in a future release.

    Parameters
    ----------
    name : unicode
        Argument name that will change in a future release.
    new_name : unicode
        Argument future release name.
    """

    def __str__(self):
        """
        Returns a formatted string representation of the deprecation type.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return ('"{0}" argument is deprecated and will be renamed to "{1}" '
                'in a future release.'.format(self.name, self.new_name))


class ArgumentFutureRemove(namedtuple('ArgumentFutureRemove', ('name', ))):
    """
    A class used for future argument removal.

    Parameters
    ----------
    name : unicode
        Argument name that will be removed in a future release.
    """

    def __str__(self):
        """
        Returns a formatted string representation of the deprecation type.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return ('"{0}" argument is deprecated and will be removed '
                'in a future release.'.format(self.name))


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
            if not isinstance(change, ObjectRemoved):

                usage_warning(str(change))

                return (getattr(self._module, attribute) if isinstance(
                    change, ObjectFutureRemove) else get_attribute(change[1]))
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

    module_name, attribute = attribute.rsplit('.', 1)

    module = sys.modules.get(module_name)
    if module is None:
        module = import_module(module_name)

    assert module is not None, (
        '"{0}" module does not exists or cannot be imported!'.format(
            module_name))

    return attrgetter(attribute)(module)


def build_API_changes(changes):
    """
    Builds the effective API changes for a desired API changes mapping.

    Parameters
    ----------
    changes : dict
        Dictionary of desired API changes.

    Returns
    -------
    dict
        API changes

    Examples
    --------
    >>> from pprint import pprint
    >>> changes = {
    ...     'ObjectRenamed': [[
    ...         'module.object_1_name',
    ...         'module.object_1_new_name',
    ...     ]],
    ...     'ObjectFutureRename': [[
    ...         'module.object_2_name',
    ...         'module.object_2_new_name',
    ...     ]],
    ...     'ObjectFutureAccessChange': [[
    ...         'module.object_3_access',
    ...         'module.sub_module.object_3_new_access',
    ...     ]],
    ...     'ObjectRemoved': ['module.object_4_name'],
    ...     'ObjectFutureRemove': ['module.object_5_name'],
    ...     'ObjectFutureAccessRemove': ['module.object_6_access'],
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

    for change_type in (ObjectRenamed, ObjectFutureRename,
                        ObjectFutureAccessChange, ArgumentRenamed,
                        ArgumentFutureRename):
        for change in changes.pop(change_type.__name__, []):
            changes[change[0].split('.')[-1]] = change_type(*change)  # noqa

    for change_type in (ObjectRemoved, ObjectFutureRemove,
                        ObjectFutureAccessRemove, ArgumentRemoved,
                        ArgumentFutureRemove):
        for change in changes.pop(change_type.__name__, []):
            changes[change.split('.')[-1]] = change_type(change)  # noqa

    return changes


def handle_arguments_deprecation(changes, **kwargs):
    """
    Handles arguments deprecation according to desired API changes mapping.

    Parameters
    ----------
    changes : dict
        Dictionary of desired API changes.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments to handle.

    Returns
    -------
    dict
        Handled keywords arguments.

    Examples
    --------
    >>> changes = {
    ...     'ArgumentRenamed': [[
    ...         'argument_1_name',
    ...         'argument_1_new_name',
    ...     ]],
    ...     'ArgumentFutureRename': [[
    ...         'argument_2_name',
    ...         'argument_2_new_name',
    ...     ]],
    ...     'ArgumentRemoved': ['argument_3_name'],
    ...     'ArgumentFutureRemove': ['argument_4_name'],
    ... }
    >>> handle_arguments_deprecation(changes, argument_1_name=True,
    ...                             argument_2_name=True, argument_4_name=True)
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
            else:
                kwargs[change[1]] = kwargs.pop(kwarg)
        else:
            raise ValueError(str(change))

    return kwargs
