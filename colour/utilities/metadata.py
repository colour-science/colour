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
    instances
    name
    strict_name

    Methods
    -------
    __str__
    __repr__
    __new__
    __init__

    Examples
    --------
    >>> Metadata('Lambda', '$\Lambda$')
    Metadata('Lambda', '$\Lambda$')
    >>> # Doctests skip for Python 2.x compatibility.
    >>> Metadata('Lambda', '$\Lambda$').family  # doctest: +SKIP
    'Metadata'
    """

    __family = 'Metadata'
    """
    Metadata class family.

    __family : unicode
    """

    __instance_id = 0
    """
    Metadata instance id number.

    __instance_id : integer
    """

    __instances = dict()
    """
    Metadata instances.

    __instances : dict
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

        instance_id = getattr(Metadata, '_Metadata__instance_id')
        setattr(instance, '_Metadata__identity', instance_id)
        getattr(Metadata, '_Metadata__instances')[instance.identity] = instance
        setattr(Metadata, '_Metadata__instance_id', instance_id + 1)

        return instance

    def __init__(self, name, strict_name=None):
        self.__identity = None

        self.__name = None
        self.name = name
        self.__strict_name = None
        self.strict_name = strict_name

    @property
    def family(self):
        """
        Property for **self.__family** private attribute.

        Returns
        -------
        unicode
            self.__family.
        """

        return getattr(self,
                       "_{0}__{1}".format(self.__class__.__name__, "family"))

    @family.setter
    def family(self, value):
        """
        Setter for **self.__family** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('family'))

    @property
    def identity(self):
        """
        Property for **self.__identity** private attribute.

        Returns
        -------
        unicode
            self.__identity.
        """

        return self.__identity

    @identity.setter
    def identity(self, value):
        """
        Setter for **self.__identity** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        raise AttributeError(
            '"{0}" attribute is read only!'.format('identity'))

    @property
    def instances(self):
        """
        Property for **self.__instances** private attribute.

        Returns
        -------
        WeakValueDictionary
            self.__instances.
        """

        return self.__instances

    @instances.setter
    def instances(self, value):
        """
        Setter for **self.__instances** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        raise AttributeError(
            '"{0}" attribute is read only!'.format('instances'))

    @property
    def name(self):
        """
        Property for **self.__name** private attribute.

        Returns
        -------
        unicode
            self.__name.
        """

        return self.__name

    @name.setter
    def name(self, value):
        """
        Setter for **self.__name** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('name', value))
        self.__name = value

    @property
    def strict_name(self):
        """
        Property for **self.__strict_name** private attribute.

        Returns
        -------
        unicode
            self.__strict_name.
        """

        if self.__strict_name is not None:
            return self.__strict_name
        else:
            return self.__name

    @strict_name.setter
    def strict_name(self, value):
        """
        Setter for **self.__strict_name** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('strict_name', value))
        self.__strict_name = value

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
            self.__name, self.__strict_name)

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
        Metadata.__repr__

        Notes
        -----
        -   Reimplements the :meth:`object.__str__` method.

        Examples
        --------
        >>> Metadata('Lambda', '$\Lambda$')
        Metadata('Lambda', '$\Lambda$')
        """

        text = '{0}(\'{1}\', \'{2}\')'.format(
            self.__class__.__name__, self.__name, self.__strict_name)

        return text


class UnitMetadata(Metadata):
    """
    Defines the metadata class used for unit of measurement.
    """

    __family = 'Unit'
    """
    Metadata class family.

    __family : unicode
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

    Examples
    --------
    >>> CallableMetadata(  # doctest: +ELLIPSIS
    ...     'Lambda', '$\Lambda$', lambda x: x).callable
    <function <lambda> at 0x...>
    """

    __family = 'Callable'
    """
    Metadata class family.

    __family : unicode
    """

    def __init__(self, name, strict_name=None, callable_=None):
        super(CallableMetadata, self).__init__(name, strict_name)

        self.__callable = None
        self.callable = callable_

    @property
    def callable(self):
        """
        Property for **self.__callable** private attribute.

        Returns
        -------
        UnitMetadata
            self.__callable.
        """

        return self.__callable

    @callable.setter
    def callable(self, value):
        """
        Setter for **self.__callable** private attribute.

        Parameters
        ----------
        value : UnitMetadata
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not a "callable"!'.format(
                    'callable', value))

        self.__callable = value


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

    __family = 'Function'
    """
    Metadata class family.

    __family : unicode
    """

    def __init__(self,
                 input_unit,
                 output_unit,
                 method,
                 strict_method=None,
                 callable_=None):

        self.__input_unit = None
        self.input_unit = input_unit
        self.__output_unit = None
        self.output_unit = output_unit
        self.__method = None
        self.method = method
        self.__strict_method = None
        self.strict_method = strict_method

        name = '{0} to {1} - {2}'.format(
            input_unit.name, output_unit.name, method)
        strict_name = '{0} to {1} - {2}'.format(
            input_unit.strict_name, output_unit.strict_name, strict_method)

        super(FunctionMetadata, self).__init__(name, strict_name, callable_)

    @property
    def input_unit(self):
        """
        Property for **self.__input_unit** private attribute.

        Returns
        -------
        UnitMetadata
            self.__input_unit.
        """

        return self.__input_unit

    @input_unit.setter
    def input_unit(self, value):
        """
        Setter for **self.__input_unit** private attribute.

        Parameters
        ----------
        value : UnitMetadata
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, UnitMetadata), (
                ('"{0}" attribute: "{1}" is not a '
                 '"UnitMetadata" instance!').format('input_unit', value))

        self.__input_unit = value

    @property
    def output_unit(self):
        """
        Property for **self.__output_unit** private attribute.

        Returns
        -------
        UnitMetadata
            self.__output_unit.
        """

        return self.__output_unit

    @output_unit.setter
    def output_unit(self, value):
        """
        Setter for **self.__output_unit** private attribute.

        Parameters
        ----------
        value : UnitMetadata
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, UnitMetadata), (
                ('"{0}" attribute: "{1}" is not a '
                 '"UnitMetadata" instance!').format('output_unit', value))

        self.__output_unit = value

    @property
    def method(self):
        """
        Property for **self.__method** private attribute.

        Returns
        -------
        unicode
            self.__method.
        """

        return self.__method

    @method.setter
    def method(self, value):
        """
        Setter for **self.__method** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('method', value))
        self.__method = value

    @property
    def strict_method(self):
        """
        Property for **self.__strict_method** private attribute.

        Returns
        -------
        unicode
            self.__strict_method.
        """

        if self.__strict_method is not None:
            return self.__strict_method
        else:
            return self.__method

    @strict_method.setter
    def strict_method(self, value):
        """
        Setter for **self.__strict_method** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('strict_method', value))
        self.__strict_method = value

    def __repr__(self):
        """
        Returns a formatted string representation of the metadata.

        Returns
        -------
        unicode
            Formatted string representation.

        See Also
        --------
        Metadata.__repr__

        Notes
        -----
        -   Reimplements the :meth:`object.__str__` method.

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
            repr(self.__input_unit),
            repr(self.__output_unit),
            self.method,
            self.strict_method)

        return text
