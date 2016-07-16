#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common Metadata
===============

Defines the objects implementing the base metadata system support:

-   :class:`Metadata`
-   :class:`EntityMetadata`
-   :class:`CallableMetadata`
-   :class:`FunctionMetadata`
-   :func:`set_callable_metadata`
"""

from __future__ import division, unicode_literals

import functools
from weakref import WeakValueDictionary
from colour.utilities import is_iterable

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Metadata',
           'EntityMetadata',
           'CallableMetadata',
           'FunctionMetadata',
           'set_metadata']


class Metadata(object):
    """
    Defines the base metadata class.

    *Colour* tries to adopt a functional programming style and declares classes
    only when relevant. A negative side effect of this approach is that the
    definitions are not providing enough contextual information that could be
    used for plotting purposes and other usages.

    It is for example quite convenient to define (in a programmatically
    usable way) that `colour.lightness_CIE1976` has output entity
    :math:`CIE L^\star`.

    Parameters
    ----------
    name : unicode
        Metadata object name.
    strict_name : unicode, optional
        Metadata strict object name, the scientific name for use in diagrams,
        figures, etc...

    Attributes
    ----------
    family
    instances
    index
    name
    strict_name

    Methods
    -------
    __new__
    __init__
    __str__
    __repr__

    Examples
    --------
    >>> Metadata('Lambda', '$\Lambda$')
    Metadata('Lambda', '$\Lambda$')
    >>> # Doctests skip for Python 2.x compatibility.
    >>> Metadata('Lambda', '$\Lambda$').family  # doctest: +SKIP
    'Metadata'
    """

    _FAMILY = 'Metadata'
    """
    Metadata class family.

    _FAMILY : unicode
    """
    _INSTANCES = WeakValueDictionary()
    """
    Metadata instances.

    _INSTANCES : WeakValueDictionary
    """

    _INSTANCES_COUNTER = 0
    """
    Metadata instances counter.

    _INSTANCES_COUNTER : integer
    """

    def __new__(cls, *args, **kwargs):
        """
        Constructor of the class.

        Parameters
        ----------
        \*args : list, optional
            Arguments.
        \**kwargs : dict, optional
            Keywords arguments.

        Returns
        -------
        Metadata
            Class instance.
        """

        instance = super(Metadata, cls).__new__(cls)

        index = cls._INSTANCES_COUNTER
        instance._index = index
        Metadata._INSTANCES[instance.index] = instance
        Metadata._INSTANCES_COUNTER = index + 1

        return instance

    def __init__(self, name, strict_name=None):
        self._name = None
        self.name = name
        self._strict_name = None
        self.strict_name = strict_name

    @property
    def family(self):
        """
        Property for **self._FAMILY** private attribute.

        Returns
        -------
        unicode
            self._FAMILY.
        """

        return self._FAMILY

    @family.setter
    def family(self, value):
        """
        Setter for **self._FAMILY** private attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('family'))

    @property
    def instances(self):
        """
        Property for **self._INSTANCES** private attribute.

        Returns
        -------
        WeakValueDictionary
            self._INSTANCES.
        """

        return self._INSTANCES

    @instances.setter
    def instances(self, value):
        """
        Setter for **self._INSTANCES** private attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError(
            '"{0}" attribute is read only!'.format('instances'))

    @property
    def index(self):
        """
        Property for **self._index** private attribute.

        Returns
        -------
        unicode
            self._index.
        """

        return self._index

    @index.setter
    def index(self, value):
        """
        Setter for **self._index** private attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError(
            '"{0}" attribute is read only!'.format('index'))

    @property
    def name(self):
        """
        Property for **self._name** private attribute.

        Returns
        -------
        unicode
            self._name.
        """

        return self._name

    @name.setter
    def name(self, value):
        """
        Setter for **self._name** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('name', value))
        self._name = value

    @property
    def strict_name(self):
        """
        Property for **self._strict_name** private attribute.

        Returns
        -------
        unicode
            self._strict_name.
        """

        if self._strict_name is not None:
            return self._strict_name
        else:
            return self._name

    @strict_name.setter
    def strict_name(self, value):
        """
        Setter for **self._strict_name** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('strict_name', value))
        self._strict_name = value

    def __str__(self):
        """
        Returns a pretty formatted string representation of the metadata.

        Returns
        -------
        unicode
            Pretty formatted string representation.

        See Also
        --------
        Metadata.__repr__

        Notes
        -----
        -   Reimplements the :meth:`object.__str__` method.

        Examples
        --------
        >>> print(Metadata('Lambda', '$\Lambda$'))
        Metadata
            Name        : Lambda
            Strict name : $\Lambda$
        """

        text = self.family
        text += '\n    Name        : {0}\n    Strict name : {1}'.format(
            self.name, self.strict_name)

        return text

    def __repr__(self):
        """
        Returns a formatted string representation of the metadata.

        Returns
        -------
        unicode
            Formatted string representation.

        See Also
        --------
        Metadata.__str__

        Notes
        -----
        -   Reimplements the :meth:`object.__repr__` method.

        Examples
        --------
        >>> Metadata('Lambda', '$\Lambda$')
        Metadata('Lambda', '$\Lambda$')
        """

        text = '{0}(\'{1}\', \'{2}\')'.format(
            self.__class__.__name__, self.name, self.strict_name)

        return text


class EntityMetadata(Metadata):
    """
    Defines the metadata class used for entities.
    """

    _FAMILY = 'Entity'
    """
    Metadata class family.

    _FAMILY : unicode
    """


class CallableMetadata(Metadata):
    """
    Defines the metadata class for callable objects.

    Parameters
    ----------
    name : unicode
        Metadata object name.
    strict_name : unicode, optional
        Metadata strict object name, the scientific name for use in diagrams,
        figures, etc...
    callable_ : callable, optional
        Callable to store within the metadata.

    Attributes
    ----------
    callable

    Methods
    -------
    __init__

    Examples
    --------
    >>> CallableMetadata(  # doctest: +ELLIPSIS
    ...     'Lambda', '$\Lambda$', lambda x: x).callable
    <function <lambda> at 0x...>
    """

    _FAMILY = 'Callable'
    """
    Metadata class family.

    _FAMILY : unicode
    """

    def __init__(self, name, strict_name=None, callable_=None):
        super(CallableMetadata, self).__init__(name, strict_name)

        self._callable = None
        self.callable = callable_

    @property
    def callable(self):
        """
        Property for **self._callable** private attribute.

        Returns
        -------
        EntityMetadata
            self._callable.
        """

        return self._callable

    @callable.setter
    def callable(self, value):
        """
        Setter for **self._callable** private attribute.

        Parameters
        ----------
        value : EntityMetadata
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not a "callable"!'.format(
                    'callable', value))

        self._callable = value


class FunctionMetadata(CallableMetadata):
    """
    Defines the metadata class for function converting an input entity into
    an output entity using a given method.

    Parameters
    ----------
    input_entity : EntityMetadata
        Input entity.
    output_entity : EntityMetadata
        Output entity.
    input_domain : array_like
        Input domain.
    output_range : array_like
        Output range.
    method : unicode, optional
        Method used by the function.
    strict_method : unicode, optional
        Strict method name, the scientific name for use in diagrams,
        figures, etc...
    callable_ : callable, optional
        Callable to store within the metadata.

    Attributes
    ----------
    input_entity
    output_entity
    input_domain
    output_range
    method
    strict_method

    Methods
    -------
    __init__
    __str__
    __repr__

    Examples
    --------
    >>> FunctionMetadata(
    ...     EntityMetadata('Luminance', '$Luminance (Y)$'),
    ...     EntityMetadata('Lightness', '$Lightness (L^*)$'),
    ...     (0, 100),
    ...     (0, 100),
    ...     'CIE 1976',
    ...     '$CIE 1976$')
    FunctionMetadata(EntityMetadata('Luminance', '$Luminance (Y)$'), \
EntityMetadata('Lightness', '$Lightness (L^*)$'), (0, 100), (0, 100), \
'CIE 1976', '$CIE 1976$')
    """

    _FAMILY = 'Function'
    """
    Metadata class family.

    _FAMILY : unicode
    """

    def __init__(self,
                 input_entity,
                 output_entity,
                 input_domain,
                 output_range,
                 method=None,
                 strict_method=None,
                 callable_=None):

        self._input_entity = None
        self.input_entity = input_entity
        self._output_entity = None
        self.output_entity = output_entity
        self._input_domain = None
        self.input_domain = input_domain
        self._output_range = None
        self.output_range = output_range
        self._method = None
        self.method = method
        self._strict_method = None
        self.strict_method = strict_method

        name = '{0} {1} to {2} {3}'.format(
            input_entity.name, '[{0}, {1}]'.format(*input_domain),
            output_entity.name, '[{0}, {1}]'.format(*output_range))
        if method:
            name = '{0} - {1}'.format(name, method)

        strict_name = '{0} {1} to {2} {3}'.format(
            input_entity.strict_name, '[{0}, {1}]'.format(*input_domain),
            output_entity.strict_name, '[{0}, {1}]'.format(*output_range))
        if strict_method:
            strict_name = '{0} - {1}'.format(strict_name, strict_method)

        super(FunctionMetadata, self).__init__(name, strict_name, callable_)

    @property
    def input_entity(self):
        """
        Property for **self._input_entity** private attribute.

        Returns
        -------
        EntityMetadata
            self._input_entity.
        """

        return self._input_entity

    @input_entity.setter
    def input_entity(self, value):
        """
        Setter for **self._input_entity** private attribute.

        Parameters
        ----------
        value : EntityMetadata
            Attribute value.
        """

        if value is not None:
            assert issubclass(type(value), EntityMetadata), (
                ('"{0}" attribute: "{1}" is not a '
                 '"EntityMetadata" instance!').format(
                    'input_entity', value))

        self._input_entity = value

    @property
    def output_entity(self):
        """
        Property for **self._output_entity** private attribute.

        Returns
        -------
        EntityMetadata
            self._output_entity.
        """

        return self._output_entity

    @output_entity.setter
    def output_entity(self, value):
        """
        Setter for **self._output_entity** private attribute.

        Parameters
        ----------
        value : EntityMetadata
            Attribute value.
        """

        if value is not None:
            assert issubclass(type(value), EntityMetadata), (
                ('"{0}" attribute: "{1}" is not a '
                 '"EntityMetadata" instance!').format(
                    'output_entity', value))

        self._output_entity = value

    @property
    def input_domain(self):
        """
        Property for **self._input_domain** private attribute.

        Returns
        -------
        EntityMetadata
            self._input_domain.
        """

        return self._input_domain

    @input_domain.setter
    def input_domain(self, value):
        """
        Setter for **self._input_domain** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            assert is_iterable(value), (
                '"{0}" attribute: "{1}" is not an "iterable"!'.format(
                    'input_domain', value))

            assert len(value) == 2, (
                '"{0}" attribute: "{1}" must have exactly '
                'two elements!'.format('input_domain', value))

        self._input_domain = value

    @property
    def output_range(self):
        """
        Property for **self._output_range** private attribute.

        Returns
        -------
        EntityMetadata
            self._output_range.
        """

        return self._output_range

    @output_range.setter
    def output_range(self, value):
        """
        Setter for **self._output_range** private attribute.

        Parameters
        ----------
        value : EntityMetadata
            Attribute value.
        """

        if value is not None:
            assert is_iterable(value), (
                '"{0}" attribute: "{1}" is not an "iterable"!'.format(
                    'output_range', value))

            assert len(value) == 2, (
                '"{0}" attribute: "{1}" must have exactly '
                'two elements!'.format('output_range', value))

        self._output_range = value

    @property
    def method(self):
        """
        Property for **self._method** private attribute.

        Returns
        -------
        unicode
            self._method.
        """

        return self._method

    @method.setter
    def method(self, value):
        """
        Setter for **self._method** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('method', value))
        self._method = value

    @property
    def strict_method(self):
        """
        Property for **self._strict_method** private attribute.

        Returns
        -------
        unicode
            self._strict_method.
        """

        if self._strict_method is not None:
            return self._strict_method
        else:
            return self._method

    @strict_method.setter
    def strict_method(self, value):
        """
        Setter for **self._strict_method** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('strict_method', value))
        self._strict_method = value

    def __str__(self):
        """
        Returns a pretty formatted string representation of the metadata.

        Returns
        -------
        unicode
            Pretty formatted string representation.

        See Also
        --------
        FunctionMetadata.__repr__

        Notes
        -----
        -   Reimplements the :meth:`Metadata.__str__` method.

        Examples
        --------
        >>> print(FunctionMetadata(
        ... EntityMetadata('Luminance', '$Luminance (Y)$'),
        ... EntityMetadata('Lightness', '$Lightness (L^*)'),
        ... (0, 100),
        ... (0, 100),
        ... 'CIE 1976',
        ... '$CIE 1976$'))
        Function
            Name          : Luminance [0, 100] to Lightness [0, 100] - CIE 1976
            Strict name   : $Luminance (Y)$ [0, 100] to \
$Lightness (L^*) [0, 100] - $CIE 1976$
            Entity
                Name        : Luminance
                Strict name : $Luminance (Y)$
            Entity
                Name        : Lightness
                Strict name : $Lightness (L^*)
            Method        : CIE 1976
            Strict method : $CIE 1976$
        """

        text = self.family
        text += '\n    Name          : {0}'.format(self.name)
        text += '\n    Strict name   : {0}'.format(self.strict_name)
        text += '\n    '
        text += str(self.input_entity).replace('\n', '\n    ')
        text += '\n    '
        text += str(self.output_entity).replace('\n', '\n    ')
        text += '\n    Method        : {0}'.format(self.method)
        text += '\n    Strict method : {0}'.format(self.strict_method)

        return text

    def __repr__(self):
        """
        Returns a formatted string representation of the metadata.

        Returns
        -------
        unicode
            Formatted string representation.

        See Also
        --------
        FunctionMetadata.__repr__

        Notes
        -----
        -   Reimplements the :meth:`Metadata.__repr__` method.

        Examples
        --------
        >>> FunctionMetadata(
        ... EntityMetadata('Luminance', '$Y$'),
        ... EntityMetadata('Lightness', '$L^\star$'),
        ... (0, 100),
        ... (0, 100),
        ... 'CIE 1976',
        ... '$CIE 1976$')
        FunctionMetadata(EntityMetadata('Luminance', '$Y$'), \
EntityMetadata('Lightness', '$L^\star$'), \
(0, 100), (0, 100), 'CIE 1976', '$CIE 1976$')
        """

        text = '{0}({1}, {2}, {3}, {4}, \'{5}\', \'{6}\')'.format(
            self.__class__.__name__,
            repr(self.input_entity),
            repr(self.output_entity),
            repr(self.input_domain),
            repr(self.output_range),
            self.method,
            self.strict_method)

        return text


def set_metadata(metadata, *args, **kwargs):
    """
    Decorator setting given metadata to decorated object.

    Parameters
    ----------
    \*args : list, optional
        Arguments.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    object

    Examples
    --------
    >>> @set_metadata(Metadata, 'Lambda', '$\Lambda$')
    ... def f():
    ...     pass
    >>> f.__metadata__
    Metadata('Lambda', '$\Lambda$')
    >>> m = Metadata('Gamma', '$\Gamma$')
    >>> @set_metadata(m)
    ... def f():
    ...     pass
    >>> f.__metadata__
    Metadata('Gamma', '$\Gamma$')
    """

    if not isinstance(metadata, Metadata):
        metadata = metadata(*args, **kwargs)

    def wrapper(function):
        """
        Wrapper for given function.
        """

        function.__metadata__ = metadata
        function.__metadata__.callable = function

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            """
            Wrapped function.
            """

            return function(*args, **kwargs)

        return wrapped

    return wrapper
