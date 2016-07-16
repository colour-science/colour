#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common Metadata
===============

Defines the objects implementing the base metadata system support:

-   :class:`Metadata`
-   :class:`TypeMetadata`
-   :class:`CallableMetadata`
-   :class:`CallableMetadata`
-   :func:`set_callable_metadata`
"""

from __future__ import division, unicode_literals

import functools
from weakref import WeakValueDictionary
from colour.utilities import CaseInsensitiveMapping, is_iterable

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Metadata',
           'TypeMetadata',
           'CallableMetadata',
           'TYPES',
           'metadata']


class Metadata(object):
    """
    Defines the base metadata class.

    *Colour* tries to adopt a functional programming style and declares classes
    only when relevant. A negative side effect of this approach is that the
    definitions are not providing enough contextual information that could be
    used for plotting purposes and other usages.

    It is for example quite convenient to define (in a programmatically
    usable way) that `colour.lightness_CIE1976` has output type
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


class TypeMetadata(Metadata):
    """
    Defines the metadata class used for types.

    Parameters
    ----------
    name : unicode
        Metadata object name.
    strict_name : unicode, optional
        Metadata strict object name, the scientific name for use in diagrams,
        figures, etc...
    constraint : array_like, optional
        Input domain or output range constraint.

    Attributes
    ----------
    constraint

    Methods
    -------
    __init__
    __str__
    __repr__

    Examples
    --------
    >>> TypeMetadata('Luminance', '$Luminance (Y)$', (0, 100))
    TypeMetadata('Luminance', '$Luminance (Y)$', (0, 100))
    """

    _FAMILY = 'Type'
    """
    Metadata class family.

    _FAMILY : unicode
    """

    def __init__(self,
                 name,
                 strict_name=None,
                 constraint=None):
        self._constraint = None
        self.constraint = constraint

        super(TypeMetadata, self).__init__(name, strict_name)

    @property
    def constraint(self):
        """
        Property for **self._constraint** private attribute.

        Returns
        -------
        TypeMetadata
            self._constraint.
        """

        return self._constraint

    @constraint.setter
    def constraint(self, value):
        """
        Setter for **self._constraint** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            assert is_iterable(value), (
                '"{0}" attribute: "{1}" is not an "iterable"!'.format(
                    'constraint', value))

            assert len(value) == 2, (
                '"{0}" attribute: "{1}" must have exactly '
                'two elements!'.format('constraint', value))

        self._constraint = value

    def __str__(self):
        """
        Returns a pretty formatted string representation of the metadata.

        Returns
        -------
        unicode
            Pretty formatted string representation.

        See Also
        --------
        TypeMetadata.__repr__

        Notes
        -----
        -   Reimplements the :meth:`Metadata.__str__` method.

        Examples
        --------
        >>> print(TypeMetadata('Luminance', '$Luminance (Y)$', (0, 100)))
        Type
            Name          : Luminance
            Strict name   : $Luminance (Y)$
            Constraint    : (0, 100)
        """

        text = self.family
        text += '\n    Name          : {0}'.format(self.name)
        text += '\n    Strict name   : {0}'.format(self.strict_name)
        if self.constraint is not None:
            text += '\n    Constraint    : {0}'.format(self.constraint)

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
        TypeMetadata.__repr__

        Notes
        -----
        -   Reimplements the :meth:`Metadata.__repr__` method.

        Examples
        --------
        >>> TypeMetadata('Luminance', '$Luminance (Y)$', (0, 100))
        TypeMetadata('Luminance', '$Luminance (Y)$', (0, 100))
        >>> TypeMetadata('Luminance', '$Luminance (Y)$')
        TypeMetadata('Luminance', '$Luminance (Y)$', None)
        """

        text = '{0}({1}, {2}, {3})'.format(
            self.__class__.__name__,
            repr(self.name),
            repr(self.strict_name),
            repr(self.constraint))

        return text


class CallableMetadata(Metadata):
    """
    Defines the metadata class for callable converting an input type into
    an output type using a given method.

    Parameters
    ----------
    input_type : TypeMetadata
        Input type.
    output_type : TypeMetadata
        Output type.
    method : unicode, optional
        Method used by the function.
    strict_method : unicode, optional
        Strict method name, the scientific name for use in diagrams,
        figures, etc...
    callable_ : callable, optional
        Callable to store within the metadata.

    Attributes
    ----------
    input_type
    output_type
    method
    strict_method
    callable

    Methods
    -------
    __init__
    __str__
    __repr__

    Examples
    --------
    >>> CallableMetadata(
    ...     TypeMetadata('Luminance', '$Luminance (Y)$', (0, 100)),
    ...     TypeMetadata('Lightness', '$Lightness (L^*)', (0, 100)),
    ...     'CIE 1976',
    ...     '$CIE 1976$')
    CallableMetadata(TypeMetadata('Luminance', '$Luminance (Y)$', (0, 100)), \
TypeMetadata('Lightness', '$Lightness (L^*)', (0, 100)), \
'CIE 1976', '$CIE 1976$')
    """

    _FAMILY = 'Function'
    """
    Metadata class family.

    _FAMILY : unicode
    """

    def __init__(self,
                 input_type,
                 output_type,
                 method=None,
                 strict_method=None,
                 callable_=None):

        self._input_type = None
        self.input_type = input_type
        self._output_type = None
        self.output_type = output_type
        self._method = None
        self.method = method
        self._strict_method = None
        self.strict_method = strict_method
        self._callable = None
        self.callable = callable_

        constraint_i = str(list(input_type.constraint))
        constraint_o = str(list(output_type.constraint))
        name = '{0} {1} to {2} {3}'.format(
            input_type.name, constraint_i,
            output_type.name, constraint_o)
        if method:
            name = '{0} - {1}'.format(name, method)

        strict_name = '{0} {1} to {2} {3}'.format(
            input_type.strict_name, constraint_i,
            output_type.strict_name, constraint_o)
        if strict_method:
            strict_name = '{0} - {1}'.format(strict_name, strict_method)

        super(CallableMetadata, self).__init__(name, strict_name)

    @property
    def input_type(self):
        """
        Property for **self._input_type** private attribute.

        Returns
        -------
        TypeMetadata
            self._input_type.
        """

        return self._input_type

    @input_type.setter
    def input_type(self, value):
        """
        Setter for **self._input_type** private attribute.

        Parameters
        ----------
        value : TypeMetadata
            Attribute value.
        """

        if value is not None:
            assert issubclass(type(value), TypeMetadata), (
                ('"{0}" attribute: "{1}" is not a '
                 '"TypeMetadata" instance!').format(
                    'input_type', value))

        self._input_type = value

    @property
    def output_type(self):
        """
        Property for **self._output_type** private attribute.

        Returns
        -------
        TypeMetadata
            self._output_type.
        """

        return self._output_type

    @output_type.setter
    def output_type(self, value):
        """
        Setter for **self._output_type** private attribute.

        Parameters
        ----------
        value : TypeMetadata
            Attribute value.
        """

        if value is not None:
            assert issubclass(type(value), TypeMetadata), (
                ('"{0}" attribute: "{1}" is not a '
                 '"TypeMetadata" instance!').format(
                    'output_type', value))

        self._output_type = value

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

    @property
    def callable(self):
        """
        Property for **self._callable** private attribute.

        Returns
        -------
        TypeMetadata
            self._callable.
        """

        return self._callable

    @callable.setter
    def callable(self, value):
        """
        Setter for **self._callable** private attribute.

        Parameters
        ----------
        value : TypeMetadata
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not a "callable"!'.format(
                    'callable', value))

        self._callable = value

    def __str__(self):
        """
        Returns a pretty formatted string representation of the metadata.

        Returns
        -------
        unicode
            Pretty formatted string representation.

        See Also
        --------
        CallableMetadata.__repr__

        Notes
        -----
        -   Reimplements the :meth:`Metadata.__str__` method.

        Examples
        --------
        >>> print(CallableMetadata(
        ... TypeMetadata('Luminance', '$Luminance (Y)$', (0, 100)),
        ... TypeMetadata('Lightness', '$Lightness (L^*)', (0, 100)),
        ... 'CIE 1976',
        ... '$CIE 1976$'))
        Function
            Name          : \
Luminance [0, 100] to Lightness [0, 100] - CIE 1976
            Strict name   : \
$Luminance (Y)$ [0, 100] to $Lightness (L^*) [0, 100] - $CIE 1976$
            Type
                Name          : Luminance
                Strict name   : $Luminance (Y)$
                Constraint    : (0, 100)
            Type
                Name          : Lightness
                Strict name   : $Lightness (L^*)
                Constraint    : (0, 100)
            Method        : CIE 1976
            Strict method : $CIE 1976$
        """

        text = self.family
        text += '\n    Name          : {0}'.format(self.name)
        text += '\n    Strict name   : {0}'.format(self.strict_name)
        text += '\n    '
        text += str(self.input_type).replace('\n', '\n    ')
        text += '\n    '
        text += str(self.output_type).replace('\n', '\n    ')
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
        CallableMetadata.__repr__

        Notes
        -----
        -   Reimplements the :meth:`Metadata.__repr__` method.

        Examples
        --------
        >>> CallableMetadata(
        ...     TypeMetadata('Luminance', '$Luminance (Y)$', (0, 100)),
        ...     TypeMetadata('Lightness', '$Lightness (L^*)', (0, 100)),
        ...     'CIE 1976',
        ...     '$CIE 1976$')
        CallableMetadata(\
TypeMetadata('Luminance', '$Luminance (Y)$', (0, 100)), \
TypeMetadata('Lightness', '$Lightness (L^*)', (0, 100)), \
'CIE 1976', '$CIE 1976$')
        """

        text = '{0}({1}, {2}, \'{3}\', \'{4}\')'.format(
            self.__class__.__name__,
            repr(self.input_type),
            repr(self.output_type),
            self.method,
            self.strict_method)

        return text


TYPES = CaseInsensitiveMapping(
    {'CIE Lab': TypeMetadata('CIE Lab', '$CIE L^*a^*b^*$'),
     'CIE LCHab': TypeMetadata('CIE LCHab', '$CIE LCH^*a^*b^*$'),
     'CIE XYZ': TypeMetadata('CIE XYZ', '$CIE XYZ$'),
     'Lightness Lstar': TypeMetadata('Lightness Lstar',
                                     '$Lightness (L^*)$'),
     'Lightness L': TypeMetadata('Lightness L', '$Lightness (L)$'),
     'Lightness W': TypeMetadata('Lightness W', '$Lightness (W)$'),
     'Luminance Y': TypeMetadata('Luminance Y', '$Luminance (Y)$'),
     'Luminance R_Y': TypeMetadata('Luminance R_Y',
                                   '$Luminance (R_Y)$'),
     'Munsell Value': TypeMetadata('Munsell Value',
                                   '$Munsell Value (V)$'),
     'Video Signal': TypeMetadata('Video Signal',
                                  "$Video Signal (V')$"),
     'Tristimulus Value': TypeMetadata('Tristimulus Value',
                                       '$Tristimulus Value (L)$')})


def metadata(*args, **kwargs):
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
    >>> @metadata(Metadata, 'Lambda', '$\Lambda$')
    ... def f():
    ...     pass
    >>> f.__metadata__
    Metadata('Lambda', '$\Lambda$')
    >>> m = Metadata('Gamma', '$\Gamma$')
    >>> @metadata(m)
    ... def f():
    ...     pass
    >>> f.__metadata__
    Metadata('Gamma', '$\Gamma$')
    """

    # if not isinstance(metadata, Metadata):
    #     metadata = metadata(*args, **kwargs)

    def wrapper(function):
        """
        Wrapper for given function.
        """

        # function.__metadata__ = metadata
        # function.__metadata__.callable = function

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            """
            Wrapped function.
            """

            return function(*args, **kwargs)

        return wrapped

    return wrapper
