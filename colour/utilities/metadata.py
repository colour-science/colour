#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metadata
========

Defines the objects implementing the base metadata system support:

-   :class:`Metadata`
-   :class:`UnitMetadata`
-   :class:`CallableMetadata`
-   :class:`FunctionMetadata`
"""

from __future__ import division, unicode_literals

from weakref import WeakValueDictionary

import colour  # noqa

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Metadata',
           'UnitMetadata',
           'CallableMetadata',
           'FunctionMetadata']


class Metadata(object):
    """
    Defines the base metadata class.

    *Colour* tries to adopt a functional programming style and declares classes
    only when relevant. A negative side effect of this approach is that the
    definitions are not providing enough contextual information that could be
    used for plotting purposes and other usages.

    It is for example quite convenient to define (in a programmatically
    usable way) that `colour.lightness_CIE1976` has output unit of measurement
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
    identity
    instances
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

    _INSTANCE_ID = 0
    """
    Metadata instance id number.

    _INSTANCE_ID : integer
    """

    _INSTANCES = WeakValueDictionary()
    """
    Metadata instances.

    _INSTANCES : WeakValueDictionary
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

        instance_id = cls._INSTANCE_ID
        instance._identity = instance_id
        Metadata._INSTANCES[instance.identity] = instance
        Metadata._INSTANCE_ID = instance_id + 1

        return instance

    def __init__(self, name, strict_name=None):
        self._identity = None

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
    def identity(self):
        """
        Property for **self._identity** private attribute.

        Returns
        -------
        unicode
            self._identity.
        """

        return self._identity

    @identity.setter
    def identity(self, value):
        """
        Setter for **self._identity** private attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError(
            '"{0}" attribute is read only!'.format('identity'))

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
            self._name, self._strict_name)

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
            self.__class__.__name__, self._name, self._strict_name)

        return text


class UnitMetadata(Metadata):
    """
    Defines the metadata class used for unit of measurement.
    """

    _FAMILY = 'Unit'
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
        UnitMetadata
            self._callable.
        """

        return self._callable

    @callable.setter
    def callable(self, value):
        """
        Setter for **self._callable** private attribute.

        Parameters
        ----------
        value : UnitMetadata
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not a "callable"!'.format(
                    'callable', value))

        self._callable = value


class FunctionMetadata(CallableMetadata):
    """
    Defines the metadata class for function converting an input unit of
    measurement into an output unit of measurement using a given method.

    Parameters
    ----------
    input_unit : UnitMetadata
        Input unit of measurement metadata.
    output_unit : UnitMetadata
        Output unit of measurement.
    method : unicode
        Method used by the function.
    strict_method : unicode, optional
        Strict method name, the scientific name for use in diagrams,
        figures, etc...
    callable_ : callable, optional
        Callable to store within the metadata.

    Attributes
    ----------
    input_unit
    output_unit
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
    ... UnitMetadata('Luminance', '$Y$'),
    ... UnitMetadata('Lightness', '$L^\star$'),
    ... 'CIE 1976',
    ... '$CIE 1976$')
    FunctionMetadata(UnitMetadata('Luminance', '$Y$'), \
UnitMetadata('Lightness', '$L^\star$'), 'CIE 1976', '$CIE 1976$')
    """

    _FAMILY = 'Function'
    """
    Metadata class family.

    _FAMILY : unicode
    """

    def __init__(self,
                 input_unit,
                 output_unit,
                 method,
                 strict_method=None,
                 callable_=None):

        self._input_unit = None
        self.input_unit = input_unit
        self._output_unit = None
        self.output_unit = output_unit
        self._method = None
        self.method = method
        self._strict_method = None
        self.strict_method = strict_method

        name = '{0} to {1} - {2}'.format(
            input_unit.name, output_unit.name, method)
        strict_name = '{0} to {1} - {2}'.format(
            input_unit.strict_name, output_unit.strict_name, strict_method)

        super(FunctionMetadata, self).__init__(name, strict_name, callable_)

    @property
    def input_unit(self):
        """
        Property for **self._input_unit** private attribute.

        Returns
        -------
        UnitMetadata
            self._input_unit.
        """

        return self._input_unit

    @input_unit.setter
    def input_unit(self, value):
        """
        Setter for **self._input_unit** private attribute.

        Parameters
        ----------
        value : UnitMetadata
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, UnitMetadata), (
                ('"{0}" attribute: "{1}" is not a '
                 '"UnitMetadata" instance!').format('input_unit', value))

        self._input_unit = value

    @property
    def output_unit(self):
        """
        Property for **self._output_unit** private attribute.

        Returns
        -------
        UnitMetadata
            self._output_unit.
        """

        return self._output_unit

    @output_unit.setter
    def output_unit(self, value):
        """
        Setter for **self._output_unit** private attribute.

        Parameters
        ----------
        value : UnitMetadata
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, UnitMetadata), (
                ('"{0}" attribute: "{1}" is not a '
                 '"UnitMetadata" instance!').format('output_unit', value))

        self._output_unit = value

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
        -   Reimplements the :meth:`object.__str__` method.

        Examples
        --------
        >>> print(FunctionMetadata(
        ... UnitMetadata('Luminance', '$Y$'),
        ... UnitMetadata('Lightness', '$L^\star$'),
        ... 'CIE 1976',
        ... '$CIE 1976$'))
        Function
            Name          : Luminance to Lightness - CIE 1976
            Strict name   : $Y$ to $L^\star$ - $CIE 1976$
            Unit
                Name        : Luminance
                Strict name : $Y$
            Unit
                Name        : Lightness
                Strict name : $L^\star$
            Method        : CIE 1976
            Strict method : $CIE 1976$
        """

        text = self.family
        text += '\n    Name          : {0}'.format(self.name)
        text += '\n    Strict name   : {0}'.format(self.strict_name)
        text += '\n    '
        text += str(self._input_unit).replace('\n', '\n    ')
        text += '\n    '
        text += str(self._output_unit).replace('\n', '\n    ')
        text += '\n    Method        : {0}'.format(self._method)
        text += '\n    Strict method : {0}'.format(self._strict_method)

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
        -   Reimplements the :meth:`object.__repr__` method.

        Examples
        --------
        >>> FunctionMetadata(
        ... UnitMetadata('Luminance', '$Y$'),
        ... UnitMetadata('Lightness', '$L^\star$'),
        ... 'CIE 1976',
        ... '$CIE 1976$')
        FunctionMetadata(UnitMetadata('Luminance', '$Y$'), \
UnitMetadata('Lightness', '$L^\star$'), 'CIE 1976', '$CIE 1976$')
        """

        text = '{0}({1}, {2}, \'{3}\', \'{4}\')'.format(
            self.__class__.__name__,
            repr(self._input_unit),
            repr(self._output_unit),
            self.method,
            self.strict_method)

        return text
