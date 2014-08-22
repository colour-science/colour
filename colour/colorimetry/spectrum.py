#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spectrum
========

Defines the classes handling spectral data computation:

-   :class:`SpectralShape`
-   :class:`SpectralPowerDistribution`
-   :class:`TriSpectralPowerDistribution`
"""

from __future__ import division, unicode_literals

import copy
import itertools
import math
import numpy as np

from colour.algebra import (
    is_iterable,
    is_numeric,
    is_uniform,
    get_steps,
    to_ndarray)
from colour.algebra import (
    Extrapolator1d,
    LinearInterpolator1d,
    SplineInterpolator,
    SpragueInterpolator)
from colour.utilities import is_string

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['SpectralShape',
           'DEFAULT_SPECTRAL_SHAPE',
           'SpectralPowerDistribution',
           'TriSpectralPowerDistribution',
           'constant_spd',
           'zeros_spd',
           'ones_spd']


class SpectralShape(object):
    """
    Defines the base object for spectral power distribution shape.

    Parameters
    ----------
    start : numeric, optional
        Wavelengths :math:`\lambda_{i}`: range start in nm.
    end : numeric, optional
        Wavelengths :math:`\lambda_{i}`: range end in nm.
    steps : numeric, optional
        Wavelengths :math:`\lambda_{i}`: range steps.

    Attributes
    ----------
    start
    end
    steps

    Methods
    -------
    __repr__
    __contains__
    __len__
    __eq__
    __ne__
    range

    Examples
    --------
    >>> SpectralShape(360, 830, 1)
    SpectralShape(360, 830, 1)
    """

    def __init__(self, start=None, end=None, steps=None):
        # Attribute storing the spectral shape range for caching purpose.
        self.__range = None

        self.__start = None
        self.__end = None
        self.__steps = None
        self.start = start
        self.end = end
        self.steps = steps

    @property
    def start(self):
        """
        Property for **self.__start** private attribute.

        Returns
        -------
        numeric
            self.__start.
        """

        return self.__start

    @start.setter
    def start(self, value):
        """
        Setter for **self.__start** private attribute.

        Parameters
        ----------
        value : str or unicode
            Attribute value.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" type is not "numeric"!'.format(
                    'start', value))
            if self.__end is not None:
                assert value < self.__end, (
                    '"{0}" attribute value must be strictly less than '
                    '"{1}"!'.format('start', self.__end))

        # Invalidating the *range* cache.
        if value != self.__start:
            self.__range = None

        self.__start = value

    @property
    def end(self):
        """
        Property for **self.__end** private attribute.

        Returns
        -------
        numeric
            self.__end.
        """

        return self.__end

    @end.setter
    def end(self, value):
        """
        Setter for **self.__end** private attribute.

        Parameters
        ----------
        value : str or unicode
            Attribute value.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" type is not "numeric"!'.format(
                    'end', value))
            if self.__start is not None:
                assert value > self.__start, (
                    '"{0}" attribute value must be strictly greater than '
                    '"{1}"!'.format('end', self.__start))

        # Invalidating the *range* cache.
        if value != self.__end:
            self.__range = None

        self.__end = value

    @property
    def steps(self):
        """
        Property for **self.__steps** private attribute.

        Returns
        -------
        numeric
            self.__steps.
        """

        return self.__steps

    @steps.setter
    def steps(self, value):
        """
        Setter for **self.__steps** private attribute.

        Parameters
        ----------
        value : str or unicode
            Attribute value.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" type is not "numeric"!'.format(
                    'steps', value))

        # Invalidating the *range* cache.
        if value != self.__steps:
            self.__range = None

        self.__steps = value

    def __str__(self):
        """
        Returns a nice formatted string representation.

        Returns
        -------
        unicode
            Nice formatted string representation.
        """

        return '({0}, {1}, {2})'.format(self.__start,
                                        self.__end,
                                        self.__steps)

    def __repr__(self):
        """
        Returns a formatted string representation.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return 'SpectralShape({0}, {1}, {2})'.format(self.__start,
                                                     self.__end,
                                                     self.__steps)

    def __iter__(self):
        """
        Returns a generator for the spectral power distribution data.

        Returns
        -------
        generator
            Spectral power distribution data generator.

        Notes
        -----
        -   Reimplements the :meth:`object.__iter__` method.

        Examples
        --------
        >>> shape = SpectralShape(0, 10, 1)
        >>> for wavelength in shape:
        >>>     print(wavelength)
        0
        1
        2
        3
        4
        5
        6
        7
        8
        9
        10
        """

        return iter(self.range())

    def __contains__(self, wavelength):
        """
        Returns if the spectral shape contains the given wavelength
        :math:`\lambda`.

        Parameters
        ----------
        wavelength : numeric
            Wavelength :math:`\lambda`.

        Returns
        -------
        bool
            Is wavelength :math:`\lambda` in the spectral shape.

        Notes
        -----
        -   Reimplements the :meth:`object.__contains__` method.

        Examples
        --------
        >>> 0.5 in SpectralShape(0, 10, 0.1)
        True
        >>> 0.51 in SpectralShape(0, 10, 0.1)
        False
        """

        return wavelength in self.range()

    def __len__(self):
        """
        Returns the spectral shape wavelengths :math:`\lambda_n`
        count.

        Returns
        -------
        int
            Spectral shape wavelengths :math:`\lambda_n` count.

        Notes
        -----
        -   Reimplements the :meth:`object.__len__` method.

        Examples
        --------
        >>> len(colour.SpectralShape(0, 10, 0.1))
        101
        """

        return len(self.range())

    def __eq__(self, shape):
        """
        Returns the spectral shape equality with given other spectral shape.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape to compare for equality.

        Returns
        -------
        bool
            Spectral shape equality.

        Notes
        -----
        -   Reimplements the :meth:`object.__eq__` method.

        Examples
        --------
        >>> SpectralShape(0, 10, 0.1) == SpectralShape(0, 10, 0.1)
        True
        >>> SpectralShape(0, 10, 0.1) == SpectralShape(0, 10, 1)
        False
        """

        return isinstance(shape, self.__class__) and np.array_equal(
            self.range(), shape.range())

    def __ne__(self, shape):
        """
        Returns the spectral shape inequality with given other spectral shape.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape to compare for inequality.

        Returns
        -------
        bool
            Spectral shape inequality.

        Notes
        -----
        -   Reimplements the :meth:`object.__ne__` method.

        Examples
        --------
        >>> SpectralShape(0, 10, 0.1) != SpectralShape(0, 10, 0.1)
        False
        >>> SpectralShape(0, 10, 0.1) != SpectralShape(0, 10, 1)
        True
        """

        return not (self == shape)

    def range(self):
        """
        Returns an iterable range for the spectral power distribution shape.

        Returns
        -------
        ndarray
            Iterable range for the spectral power distribution shape

        Examples
        --------
        >>> SpectralShape(0, 10, 0.1).range()
        array([  0. ,   0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7,   0.8,
                 0.9,   1. ,   1.1,   1.2,   1.3,   1.4,   1.5,   1.6,   1.7,
                 1.8,   1.9,   2. ,   2.1,   2.2,   2.3,   2.4,   2.5,   2.6,
                 2.7,   2.8,   2.9,   3. ,   3.1,   3.2,   3.3,   3.4,   3.5,
                 3.6,   3.7,   3.8,   3.9,   4. ,   4.1,   4.2,   4.3,   4.4,
                 4.5,   4.6,   4.7,   4.8,   4.9,   5. ,   5.1,   5.2,   5.3,
                 5.4,   5.5,   5.6,   5.7,   5.8,   5.9,   6. ,   6.1,   6.2,
                 6.3,   6.4,   6.5,   6.6,   6.7,   6.8,   6.9,   7. ,   7.1,
                 7.2,   7.3,   7.4,   7.5,   7.6,   7.7,   7.8,   7.9,   8. ,
                 8.1,   8.2,   8.3,   8.4,   8.5,   8.6,   8.7,   8.8,   8.9,
                 9. ,   9.1,   9.2,   9.3,   9.4,   9.5,   9.6,   9.7,   9.8,
                 9.9,  10. ])
        """

        if None in (self.__start, self.__end, self.__steps):
            raise ValueError(('One of the spectral shape "start", "end" or '
                              '"steps" attributes is not defined!'))

        if self.__range is None:
            self.__range = np.arange(self.__start,
                                     self.__end + self.__steps,
                                     self.__steps)

        return self.__range


DEFAULT_SPECTRAL_SHAPE = SpectralShape(360, 830, 1)
"""
Default spectral shape using the shape of
*CIE 1931 2 Degree Standard Observer*.

DEFAULT_SPECTRAL_SHAPE : SpectralShape
"""


class SpectralPowerDistribution(object):
    """
    Defines the base object for spectral data computations.

    Parameters
    ----------
    name : str or unicode
        Spectral power distribution name.
    data : dict
        Spectral power distribution data in a *dict* as follows:
        {wavelength :math:`\lambda_{i}`: value,
        wavelength :math:`\lambda_{i+1}`,
        ...,
        wavelength :math:`\lambda_{i+n}`}

    Attributes
    ----------
    name
    data
    wavelengths
    values
    items
    shape

    Methods
    -------
    __getitem__
    __init__
    __setitem__
    __iter__
    __contains__
    __len__
    __eq__
    __ne__
    __add__
    __sub__
    __mul__
    __div__
    __truediv__
    get
    is_uniform
    extrapolate
    interpolate
    align
    zeros
    normalise
    clone

    Examples
    --------
    >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
    >>> spd = colour.SpectralPowerDistribution('Spd', data)
    >>> spd.wavelengths
    array([510, 520, 530, 540])
    >>> spd.values
    array([ 49.67,  69.59,  81.73,  88.19])
    >>> spd.shape
    SpectralShape(510, 540, 10)
    """

    def __init__(self, name, data):
        self.__name = None
        self.name = name
        self.__data = None
        self.data = data

    @property
    def name(self):
        """
        Property for **self.__name** private attribute.

        Returns
        -------
        str or unicode
            self.__name.
        """

        return self.__name

    @name.setter
    def name(self, value):
        """
        Setter for **self.__name** private attribute.

        Parameters
        ----------
        value : str or unicode
            Attribute value.
        """

        if value is not None:
            assert type(value) in (str, unicode), \
                '"{0}" attribute: "{1}" type is not "str" or "unicode"!'.format(
                    'name', value)
        self.__name = value

    @property
    def data(self):
        """
        Property for **self.__data** private attribute.

        Returns
        -------
        dict
            self.__data.
        """

        return self.__data

    @data.setter
    def data(self, value):
        """
        Setter for **self.__data** private attribute.

        Parameters
        ----------
        value : dict
            Attribute value.
        """

        if value is not None:
            assert type(value) is dict, \
                '"{0}" attribute: "{1}" type is not "dict"!'.format(
                    'data', value)
        self.__data = value

    @property
    def wavelengths(self):
        """
        Property for **self.wavelengths** attribute.

        Returns
        -------
        ndarray
            Spectral power distribution wavelengths :math:`\lambda_n`.

        Warning
        -------
        :attr:`SpectralPowerDistribution.wavelengths` is read only.
        """

        return np.array(sorted(self.__data.keys()))

    @wavelengths.setter
    def wavelengths(self, value):
        """
        Setter for **self.wavelengths** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError(
            '"{0}" attribute is read only!'.format('wavelengths'))

    @property
    def values(self):
        """
        Property for **self.values** attribute.

        Returns
        -------
        ndarray
            Spectral power distribution wavelengths :math:`\lambda_n` values.

        Warning
        -------
        :attr:`SpectralPowerDistribution.values` is read only.
        """

        return np.array([self.get(wavelength)
                         for wavelength in self.wavelengths])

    @values.setter
    def values(self, value):
        """
        Setter for **self.values** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('values'))

    @property
    def items(self):
        """
        Property for **self.items** attribute. This is a convenient attribute
        used to iterate over the spectral power distribution.

        Returns
        -------
        generator
            Spectral power distribution data generator.
        """

        return self.__iter__()

    @items.setter
    def items(self, value):
        """
        Setter for **self.items** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('items'))

    @property
    def shape(self):
        """
        Property for **self.shape** attribute.

        Returns the shape of the spectral power distribution in the form of a
        :class:`SpectralShape` class instance.

        Returns
        -------
        SpectralShape
            Spectral power distribution shape.

        See Also
        --------
        SpectralPowerDistribution.is_uniform

        Notes
        -----
        -   A non uniform spectral power distribution may will have multiple
            different steps, in that case
            :attr:`SpectralPowerDistribution.shape` returns the *minimum* steps
            size.

        Warning
        -------
        :attr:`SpectralPowerDistribution.shape` is read only.

        Examples
        --------
        Uniform spectral power distribution:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> colour.SpectralPowerDistribution('Spd', data).shape
        SpectralShape(510, 550, 10)

        Non uniform spectral power distribution:

        >>> data = {512.3: 49.6700, 524.5: 69.5900, 532.4: 81.7300, 545.7: 88.1900}
        >>> colour.SpectralPowerDistribution('Spd', data).shape
        SpectralShape(512.3, 545.7, 7.8999999999999773)
        """

        return SpectralShape(min(self.data.keys()),
                             max(self.data.keys()),
                             min(get_steps(self.wavelengths)))

    @shape.setter
    def shape(self, value):
        """
        Setter for **self.shape** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('shape'))

    def __hash__(self):
        """
        Returns the spectral power distribution hash value.

        Returns
        -------
        int
            Object hash.

        Notes
        -----
        -   Reimplements the :meth:`object.__hash__` method.

        Warning
        -------
        :class:`SpectralPowerDistribution` class is mutable and should not be
        hashable. However, so that it can be used as a key in some data caches,
        we provide a *__hash__* implementation, **assuming that the underlying
        data will not change for those specific cases**.

        References
        ----------
        .. [1]  http://stackoverflow.com/a/16162138/931625
                (Last accessed 8 August 2014)
        """

        return hash(frozenset(self.__data))

    def __getitem__(self, wavelength):
        """
        Returns the value for given wavelength :math:`\lambda`.

        Parameters
        ----------
        wavelength: numeric
            Wavelength :math:`\lambda` to retrieve the value.

        Returns
        -------
        numeric
            Wavelength :math:`\lambda` value.

        See Also
        --------
        SpectralPowerDistribution.get

        Notes
        -----
        -   Reimplements the :meth:`object.__getitem__` method.

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd[510]
        49.67
        """

        return self.__data.__getitem__(wavelength)

    def __setitem__(self, wavelength, value):
        """
        Sets the wavelength :math:`\lambda` with given value.

        Parameters
        ----------
        wavelength : numeric
            Wavelength :math:`\lambda` to set.
        value : numeric
            Value for wavelength :math:`\lambda`.

        Notes
        -----
        -   Reimplements the :meth:`object.__setitem__` method.

        Examples
        --------
        >>> spd = colour.SpectralPowerDistribution('Spd', {})
        >>> spd[510] = 49.6700
        >>> spd.values
        array([ 49.67])
        """

        self.__data.__setitem__(wavelength, value)

    def __iter__(self):
        """
        Returns a generator for the spectral power distribution data.

        Returns
        -------
        generator
            Spectral power distribution data generator.

        Notes
        -----
        -   Reimplements the :meth:`object.__iter__` method.

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> for wavelength, value in spd:
        >>>     print(wavelength, value)
        (510, 49.67)
        (520, 69.59)
        (530, 81.73)
        (540, 88.19)
        """

        return iter(sorted(self.__data.items()))

    def __contains__(self, wavelength):
        """
        Returns if the spectral power distribution contains the given
        wavelength :math:`\lambda`.

        Parameters
        ----------
        wavelength : numeric
            Wavelength :math:`\lambda`.

        Returns
        -------
        bool
            Is wavelength :math:`\lambda` in the spectral power distribution.

        Notes
        -----
        -   Reimplements the :meth:`object.__contains__` method.

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> 510 in spd
        True
        """

        return wavelength in self.__data

    def __len__(self):
        """
        Returns the spectral power distribution wavelengths :math:`\lambda_n`
        count.

        Returns
        -------
        int
            Spectral power distribution wavelengths :math:`\lambda_n` count.

        Notes
        -----
        -   Reimplements the :meth:`object.__len__` method.

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> len(spd)
        4
        """

        return len(self.__data)

    def __eq__(self, spd):
        """
        Returns the spectral power distribution equality with given other
        spectral power distribution.

        Parameters
        ----------
        spd : SpectralPowerDistribution
            Spectral power distribution to compare for equality.

        Returns
        -------
        bool
            Spectral power distribution equality.

        Notes
        -----
        -   Reimplements the :meth:`object.__eq__` method.

        Examples
        --------
        >>> data1 = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> data2 = {510: 48.6700, 520: 69.5900, 530: 81.7300, 540: 88.1900}
        >>> spd1 = colour.SpectralPowerDistribution('Spd', data1)
        >>> spd2 = colour.SpectralPowerDistribution('Spd', data2)
        >>> spd3 = colour.SpectralPowerDistribution('Spd', data2)
        >>> spd1 == spd2
        False
        >>> spd2 == spd3
        True
        """

        return isinstance(spd, self.__class__) and spd.data == self.data

    def __ne__(self, spd):
        """
        Returns the spectral power distribution inequality with given other
        spectral power distribution.

        Parameters
        ----------
        spd : SpectralPowerDistribution
            Spectral power distribution to compare for inequality.

        Returns
        -------
        bool
            Spectral power distribution inequality.

        Notes
        -----
        -   Reimplements the :meth:`object.__ne__` method.

        Examples
        --------
        >>> data1 = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> data2 = {510: 48.6700, 520: 69.5900, 530: 81.7300, 540: 88.1900}
        >>> spd1 = colour.SpectralPowerDistribution('Spd', data1)
        >>> spd2 = colour.SpectralPowerDistribution('Spd', data2)
        >>> spd3 = colour.SpectralPowerDistribution('Spd', data2)
        >>> spd1 != spd2
        True
        >>> spd2 != spd3
        False
        """

        return not (self == spd)

    def __format_operand(self, x):
        """
        Formats given :math:`x` variable operand to *numeric* or *ndarray*.

        This method is a convenient method to prepare the given :math:`x`
        variable for the arithmetic operations below.

        Parameters
        ----------
        x : numeric or ndarray or SpectralPowerDistribution
            Variable to format.

        Returns
        -------
        numeric or ndarray
            Formatted operand.
        """

        if issubclass(type(x), SpectralPowerDistribution):
            x = x.values
        elif is_iterable(x):
            x = to_ndarray(x)

        return x

    def __add__(self, x):
        """
        Implements support for spectral power distribution addition.

        Parameters
        ----------
        x : numeric or array_like or SpectralPowerDistribution
            Variable to add.

        Returns
        -------
        SpectralPowerDistribution
            Variable added spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.__sub__, SpectralPowerDistribution.__mul__,
        SpectralPowerDistribution.__div__

        Notes
        -----
        -   Reimplements the :meth:`object.__add__` method.

        Warning
        -------
        The addition operation happens in place.

        Examples
        --------
        Adding a single *numeric* variable:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd + 10
        >>> spd.values
        array([ 59.67,  79.59,  91.73,  98.19])

        Adding an *array_like* variable:

        >>> spd + [1, 2, 3, 4]
        >>> spd.values
        array([  60.67,   81.59,   94.73,  102.19])

        Adding a :class:`SpectralPowerDistribution` class variable:

        >>> spd_alternate = colour.SpectralPowerDistribution('Spd', data)
        >>> spd + spd_alternate
        >>> spd.values
        array([ 110.34,  151.18,  176.46,  190.38])
        """

        self.__data = dict(zip(self.wavelengths,
                               self.values + self.__format_operand(x)))

        return self

    def __sub__(self, x):
        """
        Implements support for spectral power distribution subtraction.

        Parameters
        ----------
        x : numeric or array_like or SpectralPowerDistribution
            Variable to subtract.

        Returns
        -------
        SpectralPowerDistribution
            Variable subtracted spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.__add__, SpectralPowerDistribution.__mul__,
        SpectralPowerDistribution.__div__

        Notes
        -----
        -   Reimplements the :meth:`object.__sub__` method.

        Warning
        -------
        The subtraction operation happens in place.

        Examples
        --------
        Subtracting a single *numeric* variable:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd - 10
        >>> spd.values
        array([ 39.67,  59.59,  71.73,  78.19])

        Subtracting an *array_like* variable:

        >>> spd - [1, 2, 3, 4]
        >>> spd.values
        array([ 38.67,  57.59,  68.73,  74.19])

        Subtracting a :class:`SpectralPowerDistribution` class variable:

        >>> spd_alternate = colour.SpectralPowerDistribution('Spd', data)
        >>> spd - spd_alternate
        >>> spd.values
        array([-11., -12., -13., -14.])
        """

        return self + (-self.__format_operand(x))

    def __mul__(self, x):
        """
        Implements support for spectral power distribution multiplication.

        Parameters
        ----------
        x : numeric or array_like or SpectralPowerDistribution
            Variable to multiply.

        Returns
        -------
        SpectralPowerDistribution
            Variable multiplied spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.__add__, SpectralPowerDistribution.__sub__,
        SpectralPowerDistribution.__div__

        Notes
        -----
        -   Reimplements the :meth:`object.__mul__` method.

        Warning
        -------
        The multiplication operation happens in place.

        Examples
        --------
        Multiplying a single *numeric* variable:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd * 10
        >>> spd.values
        array([ 496.7,  695.9,  817.3,  881.9])

        Multiplying an *array_like* variable:

        >>> spd * [1, 2, 3, 4]
        >>> spd.values
        array([  496.7,  1391.8,  2451.9,  3527.6])

        Multiplying a :class:`SpectralPowerDistribution` class variable:

        >>> spd_alternate = colour.SpectralPowerDistribution('Spd', data)
        >>> spd * spd_alternate
        >>> spd.values
        array([  24671.089,   96855.362,  200393.787,  311099.044])
        """

        self.__data = dict(zip(self.wavelengths,
                               self.values * self.__format_operand(x)))

        return self

    def __div__(self, x):
        """
        Implements support for spectral power distribution division.

        Parameters
        ----------
        x : numeric or array_like or SpectralPowerDistribution
            Variable to divide.

        Returns
        -------
        SpectralPowerDistribution
            Variable divided spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.__add__, SpectralPowerDistribution.__sub__,
        SpectralPowerDistribution.__mul__

        Notes
        -----
        -   Reimplements the :meth:`object.__div__` method.

        Warning
        -------
        The division operation happens in place.

        Examples
        --------
        Dividing a single *numeric* variable:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd * 10
        >>> spd.values
        array([ 4.967,  6.959,  8.173,  8.819])

        Dividing an *array_like* variable:

        >>> spd * [1, 2, 3, 4]
        >>> spd.values
        array([ 4.967     ,  3.4795    ,  2.72433333,  2.20475   ])

        Dividing a :class:`SpectralPowerDistribution` class variable:

        >>> spd_alternate = colour.SpectralPowerDistribution('Spd', data)
        >>> spd * spd_alternate
        >>> spd.values
        array([ 0.1       ,  0.05      ,  0.03333333,  0.025     ])
        """

        self.__data = dict(zip(self.wavelengths,
                               self.values * (1 / self.__format_operand(x))))

        return self

    # Python 3 compatibility.
    __truediv__ = __div__

    def get(self, wavelength, default=None):
        """
        Returns the value for given wavelength :math:`\lambda`.

        Parameters
        ----------
        wavelength : numeric
            Wavelength :math:`\lambda` to retrieve the value.
        default : None or numeric, optional
            Wavelength :math:`\lambda` default value.

        Returns
        -------
        numeric
            Wavelength :math:`\lambda` value.

        See Also
        --------
        SpectralPowerDistribution.__getitem__

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd.get(510)
        49.67
        >>> spd.get(511)
        None
        """

        try:
            return self.__getitem__(wavelength)
        except KeyError as error:
            return default

    def is_uniform(self):
        """
        Returns if the spectral power distribution has uniformly spaced data.

        Returns
        -------
        bool
            Is uniform.

        See Also
        --------
        SpectralPowerDistribution.shape

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd.is_uniform()
        True

        Breaking the steps by introducing a new wavelength :math:`\lambda`
        value:

        >>> spd[511] = 3.1415
        >>> spd.is_uniform()
        False
        """

        return is_uniform(self.wavelengths)

    def extrapolate(self,
                    shape,
                    method='Constant',
                    left=None,
                    right=None):
        """
        Extrapolates the spectral power distribution following *CIE 15:2004*
        recommendation.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for extrapolation.
        method : unicode, optional
            ('Linear', 'Constant'),
            Extrapolation method.
        left : numeric, optional
            Value to return for low extrapolation range.
        right : numeric, optional
            Value to return for high extrapolation range.

        Returns
        -------
        SpectralPowerDistribution
            Extrapolated spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.align

        References
        ----------

        .. [2]  `CIE 015:2004 Colorimetry, 3rd edition: 7.2.2.1 Extrapolation
                <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.pdf>`_
        .. [3]  `CIE 167:2005 Recommended Practice for Tabulating Spectral Data
                for Use in Colour Computations: 10. EXTRAPOLATION
                <http://div1.cie.co.at/?i_ca_id=551&pubid=47>`_
        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd.extrapolate(colour.SpectralShape(400, 700)).shape
        SpectralShape(400, 700, 10)
        >>> spd[400]
        49.67
        >>> spd[700]
        88.19
        """

        extrapolator = Extrapolator1d(
            LinearInterpolator1d(self.wavelengths,
                                 self.values),
            method=method,
            left=left,
            right=right)

        spd_shape = self.shape
        for i in np.arange(spd_shape.start,
                           shape.start - spd_shape.steps,
                           -spd_shape.steps):
            self[i] = extrapolator(float(i))
        for i in np.arange(spd_shape.end,
                           shape.end + spd_shape.steps,
                           spd_shape.steps):
            self[i] = extrapolator(float(i))

        return self

    def interpolate(self, shape=SpectralShape(), method=None):
        """
        Interpolates the spectral power distribution following
        *CIE 167:2005* recommendations: the method developed by
        *Sprague (1880)* should be used for interpolating functions having a
        uniformly spaced independent variable and a *Cubic Spline* method for
        non-uniformly spaced independent variable.

        Parameters
        ----------
        shape : SpectralShape, optional
            Spectral shape used for interpolation.
        method : unicode, optional
            ('Sprague', 'Cubic Spline', 'Linear'),
            Enforce given interpolation method.

        Returns
        -------
        SpectralPowerDistribution
            Interpolated spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.align

        Notes
        -----
        -   Interpolation will be conducted over boundaries range, if you need
            to extend the range of the spectral power distribution use the
            :meth:`SpectralPowerDistribution.extrapolate` or
            :meth:`SpectralPowerDistribution.align` methods.
        -   *Sprague* interpolator cannot be used for interpolating
            functions having a non-uniformly spaced independent variable.

        Warning
        -------
        -   If *scipy* is not unavailable the *Cubic Spline* method will
            fallback to legacy *Linear* interpolation.
        -   *Linear* interpolator requires at least 2 wavelengths
            :math:`\lambda_n` for interpolation.
        -   *Cubic Spline* interpolator requires at least 3 wavelengths
            :math:`\lambda_n` for interpolation.
        -   *Sprague* interpolator requires at least 6 wavelengths
            :math:`\lambda_n` for interpolation.

        References
        ----------
        .. [4]  `CIE 167:2005 Recommended Practice for Tabulating Spectral Data
                for Use in Colour Computations: 9. INTERPOLATION
                <http://div1.cie.co.at/?i_ca_id=551&pubid=47>`_

        Examples
        --------
        Uniform data is using *Sprague* interpolation by default:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19, 550: 86.26, 560: 77.18}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd.interpolate(colour.SpectralShape(steps=1))
        >>> spd[515]
        60.312180023923446

        Non uniform data is using *Cubic Spline* interpolation by default:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19, 550: 86.26, 560: 77.18}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd[511] = 31.41
        >>> spd.interpolate(colour.SpectralShape(steps=1))
        >>> spd[515]
        21.479222237517757

        Enforcing *Linear* interpolation:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19, 550: 86.26, 560: 77.18}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd.interpolate(colour.SpectralShape(steps=1), method='Linear')
        >>> spd[515]
        59.63
        """

        spd_shape = self.shape
        boundaries = zip((shape.start, shape.end, shape.steps),
                         (spd_shape.start, spd_shape.end, spd_shape.end))
        boundaries = [x[0] if x[0] is not None else x[1] for x in boundaries]
        shape = SpectralShape(*boundaries)

        # Defining proper interpolation bounds.
        # TODO: Provide support for fractional steps like 0.1, etc...
        shape.start = max(shape.start, math.ceil(spd_shape.start))
        shape.end = min(shape.end, math.floor(spd_shape.end))

        wavelengths, values = self.wavelengths, self.values
        is_uniform = self.is_uniform()

        if is_string(method):
            method = method.lower()

        if method is None:
            if is_uniform:
                interpolator = SpragueInterpolator(wavelengths, values)
            else:
                interpolator = SplineInterpolator(wavelengths, values)
        elif method == 'sprague':
            if is_uniform:
                interpolator = SpragueInterpolator(wavelengths, values)
            else:
                raise RuntimeError(
                    ('"Sprague" interpolator can only be used for '
                     'interpolating functions having a uniformly spaced '
                     'independent variable!'))
        elif method == 'cubic spline':
            interpolator = SplineInterpolator(wavelengths, values)
        elif method == 'linear':
            interpolator = LinearInterpolator1d(wavelengths, values)
        else:
            raise ValueError(
                'Undefined "{0}" interpolator!'.format(method))

        self.__data = dict([(wavelength, float(interpolator(wavelength)))
                            for wavelength in shape])
        return self

    def align(self,
              shape,
              method='Constant',
              left=None,
              right=None):
        """
        Aligns the spectral power distribution to given spectral shape:
        Interpolates first then extrapolates to fit the given range.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for alignment.
        method : unicode, optional
            ('Linear', 'Constant'),
            Extrapolation method.
        left : numeric, optional
            Value to return for low extrapolation range.
        right : numeric, optional
            Value to return for high extrapolation range.

        Returns
        -------
        SpectralPowerDistribution
            Aligned spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.extrapolate,
        SpectralPowerDistribution.interpolate

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19, 550: 86.26, 560: 77.18}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd.align(colour.SpectralShape(505, 565, 1))
        >>> spd.wavelengths
        array([ 505.,  506.,  507.,  508.,  509.,  510.,  511.,  512.,  513.,
                514.,  515.,  516.,  517.,  518.,  519.,  520.,  521.,  522.,
                523.,  524.,  525.,  526.,  527.,  528.,  529.,  530.,  531.,
                532.,  533.,  534.,  535.,  536.,  537.,  538.,  539.,  540.,
                541.,  542.,  543.,  544.,  545.,  546.,  547.,  548.,  549.,
                550.,  551.,  552.,  553.,  554.,  555.,  556.,  557.,  558.,
                559.,  560.,  561.,  562.,  563.,  564.,  565.])
        >>> spd.values
        array([ 49.67      ,  49.67      ,  49.67      ,  49.67      ,
                49.67      ,  49.67      ,  51.83411622,  53.98564678,
                56.12294647,  58.23661971,  60.31218002,  62.33270959,
                64.28151876,  66.14480559,  67.91431533,  69.59      ,
                71.17599588,  72.6627938 ,  74.04657568,  75.33297102,
                76.53395428,  77.66474212,  78.74069075,  79.77419322,
                80.77157675,  81.73      ,  82.64075188,  83.507872  ,
                84.33263338,  85.109696  ,  85.82929687,  86.47944   ,
                87.04808638,  87.525344  ,  87.90565788,  88.19      ,
                88.38583474,  88.49756341,  88.52589063,  88.46965703,
                88.32664605,  88.09439066,  87.77098023,  87.35586725,
                86.85067414,  86.26      ,  85.59116999,  84.85034304,
                84.04348015,  83.17711104,  82.25838741,  81.29513608,
                80.29591222,  79.27005259,  78.22772869,  77.18      ,
                77.18      ,  77.18      ,  77.18      ,  77.18      ,  77.18])
        """

        self.interpolate(shape)
        self.extrapolate(shape, method, left, right)

        return self

    def zeros(self, shape=SpectralShape()):
        """
        Zeros fills the spectral power distribution: Missing values will be
        replaced with zeros to fit the defined range.

        Parameters
        ----------
        shape : SpectralShape, optional
            Spectral shape used for zeros fill.

        Returns
        -------
        SpectralPowerDistribution
            Zeros filled spectral power distribution.

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19, 550: 86.26, 560: 77.18}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd.zeros(colour.SpectralShape(505, 565, 1))
        >>> spd.values
        array([  0.  ,   0.  ,   0.  ,   0.  ,   0.  ,  49.67,   0.  ,   0.  ,
                 0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,  69.59,
                 0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,
                 0.  ,  81.73,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,
                 0.  ,   0.  ,   0.  ,  88.19,   0.  ,   0.  ,   0.  ,   0.  ,
                 0.  ,   0.  ,   0.  ,   0.  ,   0.  ,  86.26,   0.  ,   0.  ,
                 0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,  77.18,
                 0.  ,   0.  ,   0.  ,   0.  ,   0.  ])
        """

        spd_shape = self.shape
        boundaries = zip((shape.start, shape.end, shape.steps),
                         (spd_shape.start, spd_shape.end, spd_shape.end))
        boundaries = [x[0] if x[0] is not None else x[1] for x in boundaries]
        shape = SpectralShape(*boundaries)

        self.__data = dict(
            [(wavelength, self.get(wavelength, 0)) for wavelength in shape])

        return self

    def normalise(self, factor=1):
        """
        Normalises the spectral power distribution with given normalization
        factor.

        Parameters
        ----------
        factor : numeric, optional
            Normalization factor

        Returns
        -------
        SpectralPowerDistribution
            Normalised spectral power distribution.

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> spd.normalise()
        >>> spd.values
        array([ 0.56321578,  0.78909173,  0.92674906,  1.        ])
        """

        return (self * (1 / max(self.values))) * factor

    def clone(self):
        """
        Clones the spectral power distribution.

        Most of the :class:`SpectralPowerDistribution` class operations are
        conducted in-place. The :meth:`SpectralPowerDistribution.clone` method
        provides a convenient way to copy the spectral power distribution to a
        new object.

        Returns
        -------
        SpectralPowerDistribution
            Cloned spectral power distribution.

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = colour.SpectralPowerDistribution('Spd', data)
        >>> print(id(spd))
        >>> spd_clone = spd.clone()
        >>> print(id(spd_clone))
        4414965968
        4412951568
        """

        return copy.deepcopy(self)


class TriSpectralPowerDistribution(object):
    """
    Defines the base object for colour matching functions.

    A compound of three :class:`SpectralPowerDistribution` is used to store
    the underlying axis data.

    Parameters
    ----------
    name : str or unicode
        Tri-spectral power distribution name.
    data : dict
        Tri-spectral power distribution data.
    mapping : dict
        Tri-spectral power distribution attributes mapping.
    labels : dict
        Tri-spectral power distribution axis labels mapping.

    Attributes
    ----------
    name
    mapping
    data
    labels
    x
    y
    z
    wavelengths
    values
    items
    shape

    Methods
    -------
    __init__
    __getitem__
    __setitem__
    __iter__
    __contains__
    __len__
    __eq__
    __ne__
    __add__
    __sub__
    __mul__
    __div__
    __truediv__
    get
    is_uniform
    extrapolate
    interpolate
    align
    zeros
    normalise
    clone

    See Also
    --------
    colour.colorimetry.cmfs.LMS_ConeFundamentals,
    colour.colorimetry.cmfs.RGB_ColourMatchingFunctions,
    colour.colorimetry.cmfs.XYZ_ColourMatchingFunctions

    Examples
    --------
    """

    def __init__(self, name, data, mapping, labels):
        self.__name = None
        self.name = name
        self.__mapping = mapping
        self.__data = None
        self.data = data
        self.__labels = labels

    @property
    def name(self):
        """
        Property for **self.__name** private attribute.

        Returns
        -------
        str or unicode
            self.__name.
        """

        return self.__name

    @name.setter
    def name(self, value):
        """
        Setter for **self.__name** private attribute.

        Parameters
        ----------
        value : str or unicode
            Attribute value.
        """

        if value is not None:
            assert type(value) in (str, unicode), \
                '"{0}" attribute: "{1}" type is not "str" or "unicode"!'.format(
                    'name', value)
        self.__name = value

    @property
    def mapping(self):
        """
        Property for **self.__mapping** private attribute.

        Returns
        -------
        dict
            self.__mapping.
        """

        return self.__mapping

    @mapping.setter
    def mapping(self, value):
        """
        Setter for **self.__mapping** private attribute.

        Parameters
        ----------
        value : dict
            Attribute value.
        """

        if value is not None:
            assert type(value) is dict, \
                '"{0}" attribute: "{1}" type is not "dict"!'.format(
                    'mapping', value)
            for axis in ('x', 'y', 'z'):
                assert axis in value.keys(), \
                    '"{0}" attribute: "{1}" axis label is missing!'.format(
                        'mapping', axis)
        self.__mapping = value

    @property
    def data(self):
        """
        Property for **self.__data** private attribute.

        Returns
        -------
        dict
            self.__data.
        """

        return self.__data

    @data.setter
    def data(self, value):
        """
        Setter for **self.__data** private attribute.

        Parameters
        ----------
        value : dict
            Attribute value.
        """

        if value is not None:
            assert type(value) is dict, \
                '"{0}" attribute: "{1}" type is not "dict"!'.format(
                    'data', value)
            for axis in ('x', 'y', 'z'):
                assert self.__mapping.get(axis) in value.keys(), \
                    '"{0}" attribute: "{1}" axis is missing!'.format(
                        'data', axis)

            data = {}
            for axis in ('x', 'y', 'z'):
                data[axis] = SpectralPowerDistribution(
                    self.__mapping.get(axis),
                    value.get(self.__mapping.get(axis)))

            np.testing.assert_almost_equal(
                data['x'].wavelengths,
                data['y'].wavelengths,
                err_msg=('"{0}" attribute: "{1}" and "{2}" wavelengths are '
                         'different!').format('data',
                                              self.__mapping.get('x'),
                                              self.__mapping.get('y')))
            np.testing.assert_almost_equal(
                data['x'].wavelengths,
                data['z'].wavelengths,
                err_msg=('"{0}" attribute: "{1}" and "{2}" wavelengths are '
                         'different!').format('data',
                                              self.__mapping.get('x'),
                                              self.__mapping.get('z')))

            self.__data = data
        else:
            self.__data = None

    @property
    def labels(self):
        """
        Property for **self.__labels** private attribute.

        Returns
        -------
        dict
            self.__labels.
        """

        return self.__labels

    @labels.setter
    def labels(self, value):
        """
        Setter for **self.__labels** private attribute.

        Parameters
        ----------
        value : dict
            Attribute value.
        """

        if value is not None:
            assert type(value) is dict, \
                '"{0}" attribute: "{1}" type is not "dict"!'.format(
                    'labels', value)
            for axis in ('x', 'y', 'z'):
                assert axis in value.keys(), \
                    '"{0}" attribute: "{1}" axis label is missing!'.format(
                        'labels', axis)
        self.__labels = value

    @property
    def x(self):
        """
        Property for **self.x** attribute.

        Returns
        -------
        SpectralPowerDistribution
            Spectral power distribution for *x* axis.

        Warning
        -------
        :attr:`TriSpectralPowerDistribution.x` is read only.
        """

        return self.__data.get('x')

    @x.setter
    def x(self, value):
        """
        Setter for **self.x** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('x'))

    @property
    def y(self):
        """
        Property for **self.y** attribute.

        Returns
        -------
        SpectralPowerDistribution
            Spectral power distribution for *y* axis.

        Warning
        -------
        :attr:`TriSpectralPowerDistribution.y` is read only.
        """

        return self.__data.get('y')

    @y.setter
    def y(self, value):
        """
        Setter for **self.y** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('y'))

    @property
    def z(self):
        """
        Property for **self.z** attribute.

        Returns
        -------
        SpectralPowerDistribution
            Spectral power distribution for *z* axis.

        Warning
        -------
        :attr:`TriSpectralPowerDistribution.z` is read only.
        """

        return self.__data.get('z')

    @z.setter
    def z(self, value):
        """
        Setter for **self.z** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('z'))

    @property
    def wavelengths(self):
        """
        Property for **self.wavelengths** attribute.

        Returns
        -------
        ndarray
            Tri-spectral power distribution wavelengths :math:`\lambda_n`.

        Warning
        -------
        :attr:`TriSpectralPowerDistribution.wavelengths` is read only.
        """

        return self.x.wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        """
        Setter for **self.wavelengths** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError(
            '"{0}" attribute is read only!'.format('wavelengths'))

    @property
    def values(self):
        """
        Property for **self.values** attribute.

        Returns
        -------
        ndarray
            Tri-spectral power distribution wavelengths :math:`\lambda_n`
            values.

        Warning
        -------
        :attr:`TriSpectralPowerDistribution.values` is read only.
        """

        return np.array([self.get(wavelength)
                         for wavelength in self.wavelengths])

    @values.setter
    def values(self, value):
        """
        Setter for **self.values** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('values'))

    @property
    def items(self):
        """
        Property for **self.items** attribute. This is a convenient attribute
        used to iterate over the tri-spectral power distribution.

        Returns
        -------
        generator
            Tri-spectral power distribution data generator.
        """

        return self.__iter__()

    @items.setter
    def items(self, value):
        """
        Setter for **self.items** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('items'))


    @property
    def shape(self):
        """
        Property for **self.shape** attribute.

        Returns the shape of the tri-spectral power distribution in the form of
        a :class:`SpectralShape` class instance.

        Returns
        -------
        SpectralShape
            Tri-spectral power distribution shape.

        See Also
        --------
        SpectralPowerDistribution.is_uniform,
        TriSpectralPowerDistribution.is_uniform

        Warning
        -------
        :attr:`TriSpectralPowerDistribution.shape` is read only.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd.shape
        SpectralShape(510, 540, 10)
        """

        return self.x.shape

    @shape.setter
    def shape(self, value):
        """
        Setter for **self.shape** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('shape'))

    def __hash__(self):
        """
        Returns the spectral power distribution hash value.

        Returns
        -------
        int
            Object hash.

        Notes
        -----
        -   Reimplements the :meth:`object.__hash__` method.

        Warning
        -------
        See :meth:`SpectralPowerDistribution.__hash__` method warning section.

        References
        ----------
        .. [5]  http://stackoverflow.com/a/16162138/931625
                (Last accessed 8 August 2014)
        """

        return hash((frozenset(self.__data.get('x')),
                     frozenset(self.__data.get('y')),
                     frozenset(self.__data.get('z'))))

    def __getitem__(self, wavelength):
        """
        Returns the values for given wavelength :math:`\lambda`.

        Parameters
        ----------
        wavelength: numeric
            Wavelength :math:`\lambda` to retrieve the values.

        Returns
        -------
        ndarray, (3,)
            Wavelength :math:`\lambda` values.

        See Also
        --------
        TriSpectralPowerDistribution.get

        Notes
        -----
        -   Reimplements the :meth:`object.__getitem__` method.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd[510]
        array([ 49.67,  90.56,  12.43])
        """

        return np.array((self.x[wavelength],
                         self.y[wavelength],
                         self.z[wavelength]))

    def __setitem__(self, wavelength, value):
        """
        Sets the wavelength :math:`\lambda` with given value.

        Parameters
        ----------
        wavelength : numeric
            Wavelength :math:`\lambda` to set.
        value : array_like
            Value for wavelength :math:`\lambda`.

        Notes
        -----
        -   Reimplements the :meth:`object.__setitem__` method.

        Examples
        --------
        >>> x_bar = {}
        >>> y_bar = {}
        >>> z_bar = {}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd[510] = (49.6700, 49.6700, 49.6700)
        >>> tri_spd.values
        array([[ 49.67,  49.67,  49.67]])
        """

        x, y, z = np.ravel(value)

        self.x.__setitem__(wavelength, x)
        self.y.__setitem__(wavelength, y)
        self.z.__setitem__(wavelength, z)

    def __iter__(self):
        """
        Returns a generator for the tri-spectral power distribution data.

        Returns
        -------
        generator
            Tri-spectral power distribution data generator.

        Notes
        -----
        -   Reimplements the :meth:`object.__iter__` method.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> for wavelength, value in tri_spd:
        >>>     print(wavelength, value)
        (510, array([ 49.67,  90.56,  12.43]))
        (520, array([ 69.59,  87.34,  23.15]))
        (530, array([ 81.73,  45.76,  67.98]))
        (540, array([ 88.19,  23.45,  90.28]))
        """

        return itertools.izip(self.wavelengths, self.values)

    def __contains__(self, wavelength):
        """
        Returns if the tri-spectral power distribution contains the given
        wavelength :math:`\lambda`.

        Parameters
        ----------
        wavelength : numeric
            Wavelength :math:`\lambda`.

        Returns
        -------
        bool
            Is wavelength :math:`\lambda` in the tri-spectral power
            distribution.

        Notes
        -----
        -   Reimplements the :meth:`object.__contains__` method.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> 510 in tri_spd
        True

        """

        return wavelength in self.x

    def __len__(self):
        """
        Returns the tri-spectral power distribution wavelengths
        :math:`\lambda_n` count.

        Returns
        -------
        int
            Tri-Spectral power distribution wavelengths :math:`\lambda_n` count.

        Notes
        -----
        -   Reimplements the :meth:`object.__len__` method.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> len(tri_spd)
        4
        """

        return len(self.x)

    def __eq__(self, tri_spd):
        """
        Returns the tri-spectral power distribution equality with given other
        tri-spectral power distribution.

        Parameters
        ----------
        spd : TriSpectralPowerDistribution
            Tri-spectral power distribution to compare for equality.

        Returns
        -------
        bool
            Tri-spectral power distribution equality.

        Notes
        -----
        -   Reimplements the :meth:`object.__eq__` method.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data1 = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> data2 = {'x_bar': y_bar, 'y_bar': x_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd1 = colour.TriSpectralPowerDistribution('Tri Spd', data1, mpg, lbl)
        >>> tri_spd2 = colour.TriSpectralPowerDistribution('Tri Spd', data2, mpg, lbl)
        >>> tri_spd3 = colour.TriSpectralPowerDistribution('Tri Spd', data1, mpg, lbl)
        >>> tri_spd1 == tri_spd2
        False
        >>> tri_spd1 == tri_spd3
        True
        """

        if not isinstance(tri_spd, self.__class__):
            return False

        equality = True
        for axis in self.__mapping:
            equality *= getattr(self, axis) == getattr(tri_spd, axis)

        return bool(equality)

    def __ne__(self, tri_spd):
        """
        Returns the tri-spectral power distribution inequality with given other
        tri-spectral power distribution.

        Parameters
        ----------
        spd : TriSpectralPowerDistribution
            Tri-spectral power distribution to compare for inequality.

        Returns
        -------
        bool
            Tri-spectral power distribution inequality.

        Notes
        -----
        -   Reimplements the :meth:`object.__eq__` method.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data1 = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> data2 = {'x_bar': y_bar, 'y_bar': x_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd1 = colour.TriSpectralPowerDistribution('Tri Spd', data1, mpg, lbl)
        >>> tri_spd2 = colour.TriSpectralPowerDistribution('Tri Spd', data2, mpg, lbl)
        >>> tri_spd3 = colour.TriSpectralPowerDistribution('Tri Spd', data1, mpg, lbl)
        >>> tri_spd1 != tri_spd2
        True
        >>> tri_spd1 != tri_spd3
        False
        """

        return not (self == tri_spd)

    def __format_operand(self, x):
        """
        Formats given :math:`x` variable operand to *numeric* or *ndarray*.

        This method is a convenient method to prepare the given :math:`x`
        variable for the arithmetic operations below.

        Parameters
        ----------
        x : numeric or ndarray or TriSpectralPowerDistribution
            Variable to format.

        Returns
        -------
        numeric or ndarray
            Formatted operand.
        """

        if issubclass(type(x), TriSpectralPowerDistribution):
            x = x.values
        elif is_iterable(x):
            x = to_ndarray(x)

        return x

    def __add__(self, x):
        """
        Implements support for tri-spectral power distribution addition.

        Parameters
        ----------
        x : numeric or array_like or TriSpectralPowerDistribution
            Variable to add.

        Returns
        -------
        TriSpectralPowerDistribution
            Variable added tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.__sub__,
        TriSpectralPowerDistribution.__mul__,
        TriSpectralPowerDistribution.__div__

        Notes
        -----
        -   Reimplements the :meth:`object.__add__` method.

        Warning
        -------
        The addition operation happens in place.

        Examples
        --------
        Adding a single *numeric* variable:

        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd + 10
        >>> tri_spd.values
        array([[  59.67,  100.56,   22.43],
               [  79.59,   97.34,   33.15],
               [  91.73,   55.76,   77.98],
               [  98.19,   33.45,  100.28]])

        Adding an *array_like* variable:

        >>> tri_spd + [(1, 2, 3)] * 4
        >>> tri_spd.values
        array([[  60.67,  102.56,   25.43],
               [  80.59,   99.34,   36.15],
               [  92.73,   57.76,   80.98],
               [  99.19,   35.45,  103.28]])

        Adding a :class:`TriSpectralPowerDistribution` class variable:

        >>> data1 = {'x_bar': z_bar, 'y_bar': x_bar, 'z_bar': y_bar}
        >>> tri_spd_alternate = colour.TriSpectralPowerDistribution('Tri Spd', data1, mpg, lbl)
        >>> tri_spd + tri_spd_alternate
        >>> tri_spd.values
        array([[  73.1 ,  152.23,  115.99],
               [ 103.74,  168.93,  123.49],
               [ 160.71,  139.49,  126.74],
               [ 189.47,  123.64,  126.73]])
        """

        values = self.values + self.__format_operand(x)

        for i, axis in enumerate(('x', 'y', 'z')):
            self.__data[axis].data = dict(zip(self.wavelengths, values[:, i]))

        return self

    def __sub__(self, x):
        """
        Implements support for tri-spectral power distribution subtraction.

        Parameters
        ----------
        x : numeric or array_like or TriSpectralPowerDistribution
            Variable to subtract.

        Returns
        -------
        TriSpectralPowerDistribution
            Variable subtracted tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.__add__,
        TriSpectralPowerDistribution.__mul__,
        TriSpectralPowerDistribution.__div__

        Notes
        -----
        -   Reimplements the :meth:`object.__sub__` method.

        Warning
        -------
        The subtraction operation happens in place.

        Examples
        --------
        Subtracting a single *numeric* variable:

        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd - 10
        >>> tri_spd.values
        array([[ 39.67,  80.56,   2.43],
               [ 59.59,  77.34,  13.15],
               [ 71.73,  35.76,  57.98],
               [ 78.19,  13.45,  80.28]])

        Subtracting an *array_like* variable:

        >>> tri_spd - [(1, 2, 3)] * 4
        >>> tri_spd.values
        array([[ 38.67,  78.56,  -0.57],
               [ 58.59,  75.34,  10.15],
               [ 70.73,  33.76,  54.98],
               [ 77.19,  11.45,  77.28]])

        Subtracting a :class:`TriSpectralPowerDistribution` class variable:

        >>> data1 = {'x_bar': z_bar, 'y_bar': x_bar, 'z_bar': y_bar}
        >>> tri_spd_alternate = colour.TriSpectralPowerDistribution('Tri Spd', data1, mpg, lbl)
        >>> tri_spd - tri_spd_alternate
        >>> tri_spd.values
        array([[ 26.24,  28.89, -91.13],
               [ 35.44,   5.75, -77.19],
               [  2.75, -47.97,   9.22],
               [-13.09, -76.74,  53.83]])
        """

        return self + (-self.__format_operand(x))

    def __mul__(self, x):
        """
        Implements support for tri-spectral power distribution multiplication.

        Parameters
        ----------
        x : numeric or array_like or TriSpectralPowerDistribution
            Variable to multiply.

        Returns
        -------
        TriSpectralPowerDistribution
            Variable multiplied tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.__add__,
        TriSpectralPowerDistribution.__sub__,
        TriSpectralPowerDistribution.__div__

        Notes
        -----
        -   Reimplements the :meth:`object.__mul__` method.

        Warning
        -------
        The multiplication operation happens in place.

        Examples
        --------
        Multiplying a single *numeric* variable:

        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd * 10
        >>> tri_spd.values
        array([[ 496.7,  905.6,  124.3],
               [ 695.9,  873.4,  231.5],
               [ 817.3,  457.6,  679.8],
               [ 881.9,  234.5,  902.8]])

        Multiplying an *array_like* variable:

        >>> tri_spd * [(1, 2, 3)] * 4
        >>> tri_spd.values
        array([[  1986.8,   7244.8,   1491.6],
               [  2783.6,   6987.2,   2778. ],
               [  3269.2,   3660.8,   8157.6],
               [  3527.6,   1876. ,  10833.6]])

        Multiplying a :class:`TriSpectralPowerDistribution` class variable:

        >>> data1 = {'x_bar': z_bar, 'y_bar': x_bar, 'z_bar': y_bar}
        >>> tri_spd_alternate = colour.TriSpectralPowerDistribution('Tri Spd', data1, mpg, lbl)
        >>> tri_spd * tri_spd_alternate
        >>> tri_spd.values
        array([[  24695.924,  359849.216,  135079.296],
               [  64440.34 ,  486239.248,  242630.52 ],
               [ 222240.216,  299197.184,  373291.776],
               [ 318471.728,  165444.44 ,  254047.92 ]])
        """

        values = self.values * self.__format_operand(x)

        for i, axis in enumerate(('x', 'y', 'z')):
            self.__data[axis].data = dict(zip(self.wavelengths, values[:, i]))

        return self

    def __div__(self, x):
        """
        Implements support for tri-spectral power distribution division.

        Parameters
        ----------
        x : numeric or array_like or TriSpectralPowerDistribution
            Variable to divide.

        Returns
        -------
        TriSpectralPowerDistribution
            Variable divided tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.__add__,
        TriSpectralPowerDistribution.__sub__,
        TriSpectralPowerDistribution.__mul__

        Notes
        -----
        -   Reimplements the :meth:`object.__mul__` method.

        Warning
        -------
        The division operation happens in place.

        Examples
        --------
        Dividing a single *numeric* variable:

        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd / 10
        >>> tri_spd.values
        array([[ 4.967,  9.056,  1.243],
               [ 6.959,  8.734,  2.315],
               [ 8.173,  4.576,  6.798],
               [ 8.819,  2.345,  9.028]])

        Dividing an *array_like* variable:

        >>> tri_spd / [(1, 2, 3)] * 4
        >>> tri_spd.values
        array([[ 19.868     ,  18.112     ,   1.65733333],
               [ 27.836     ,  17.468     ,   3.08666667],
               [ 32.692     ,   9.152     ,   9.064     ],
               [ 35.276     ,   4.69      ,  12.03733333]])

        Dividing a :class:`TriSpectralPowerDistribution` class variable:

        >>> data1 = {'x_bar': z_bar, 'y_bar': x_bar, 'z_bar': y_bar}
        >>> tri_spd_alternate = colour.TriSpectralPowerDistribution('Tri Spd', data1, mpg, lbl)
        >>> tri_spd / tri_spd_alternate
        >>> tri_spd.values
        array([[ 1.59839099,  0.36464667,  0.01830094],
               [ 1.20241901,  0.25101308,  0.03534081],
               [ 0.48090615,  0.11197847,  0.19807692],
               [ 0.39073992,  0.05318063,  0.51331912]])
        """

        return self * (1 / self.__format_operand(x))

    # Python 3 compatibility.
    __truediv__ = __div__

    def get(self, wavelength, default=None):
        """
        Returns the values for given wavelength :math:`\lambda`.

        Parameters
        ----------
        wavelength : numeric
            Wavelength :math:`\lambda` to retrieve the values.
        default : None or numeric, optional
            Wavelength :math:`\lambda` default values.

        Returns
        -------
        numeric
            Wavelength :math:`\lambda` values.

        See Also
        --------
        TriSpectralPowerDistribution.__getitem__

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd[510]
        array([ 49.67,  90.56,  12.43])
        >>> tri_spd.get(511)
        None
        """

        try:
            return self.__getitem__(wavelength)
        except KeyError as error:
            return default

    def is_uniform(self):
        """
        Returns if the tri-spectral power distribution has uniformly spaced
        data.

        Returns
        -------
        bool
            Is uniform.

        See Also
        --------
        TriSpectralPowerDistribution.shape

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd.is_uniform()
        True

        Breaking the steps by introducing new wavelength :math:`\lambda`
        values:

        >>> tri_spd[511] = (49.6700, 49.6700, 49.6700)
        >>> tri_spd.is_uniform()
        False
        """

        for i in self.__mapping.keys():
            if not getattr(self, i).is_uniform():
                return False
        return True

    def extrapolate(self,
                    shape,
                    method='Constant',
                    left=None,
                    right=None):
        """
        Extrapolates the tri-spectral power distribution following
        *CIE 15:2004* recommendation.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for extrapolation.
        method : unicode, optional
            ('Linear', 'Constant'),
            Extrapolation method.
        left : numeric, optional
            Value to return for low extrapolation range.
        right : numeric, optional
            Value to return for high extrapolation range.

        Returns
        -------
        TriSpectralPowerDistribution
            Extrapolated tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.align

        References
        ----------

        .. [6]  `CIE 015:2004 Colorimetry, 3rd edition: 7.2.2.1 Extrapolation
                <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.pdf>`_
        .. [7]  `CIE 167:2005 Recommended Practice for Tabulating Spectral Data
                for Use in Colour Computations: 10. EXTRAPOLATION
                <http://div1.cie.co.at/?i_ca_id=551&pubid=47>`_
        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd.extrapolate(colour.SpectralShape(400, 700)).shape
        SpectralShape(400, 700, 10)
        >>> tri_spd[400]
        array([ 49.67,  90.56,  12.43])
        >>> tri_spd[700]
        array([ 88.19,  23.45,  90.28])
        """

        for i in self.__mapping.keys():
            getattr(self, i).extrapolate(shape, method, left, right)

        return self

    def interpolate(self, shape=SpectralShape(), method=None):
        """
        Interpolates the tri-spectral power distribution following
        *CIE 167:2005* recommendations: the method developed by
        *Sprague (1880)* should be used for interpolating functions having a
        uniformly spaced independent variable and a *Cubic Spline* method for
        non-uniformly spaced independent variable.

        Parameters
        ----------
        shape : SpectralShape, optional
            Spectral shape used for interpolation.
        method : unicode, optional
            ('Sprague', 'Cubic Spline', 'Linear'),
            Enforce given interpolation method.

        Returns
        -------
        TriSpectralPowerDistribution
            Interpolated tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.align

        Notes
        -----
        -   See :meth:`SpectralPowerDistribution.interpolate` method
            notes section.

        Warning
        -------
        See :meth:`SpectralPowerDistribution.interpolate` method warning
        section.

        References
        ----------
        .. [4]  `CIE 167:2005 Recommended Practice for Tabulating Spectral Data
                for Use in Colour Computations: 9. INTERPOLATION
                <http://div1.cie.co.at/?i_ca_id=551&pubid=47>`_

        Examples
        --------
        Uniform data is using *Sprague* interpolation by default:

        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19, 550: 89.76, 560: 90.28}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45, 550: 15.34, 560: 10.11}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28, 550: 91.61, 560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd.interpolate(colour.SpectralShape(steps=1))
        >>> tri_spd[515]
        array([ 60.30332087,  93.27163315,  13.86051361])

        Non uniform data is using *Cubic Spline* interpolation by default:

        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19, 550: 89.76, 560: 90.28}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45, 550: 15.34, 560: 10.11}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28, 550: 91.61, 560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd[511] = (31.41, 95.27, 15.06)
        >>> tri_spd.interpolate(colour.SpectralShape(steps=1))
        >>> tri_spd[515]
        array([  21.47104053,  100.64300155,   18.8165196 ])

        Enforcing *Linear* interpolation:

        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19, 550: 89.76, 560: 90.28}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45, 550: 15.34, 560: 10.11}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28, 550: 91.61, 560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd.interpolate(colour.SpectralShape(steps=1), method='Linear')
        >>> tri_spd[515]
        array([ 59.63,  88.95,  17.79])
        """

        for i in self.__mapping.keys():
            getattr(self, i).interpolate(shape, method)

        return self

    def align(self,
              shape,
              method='Constant',
              left=None,
              right=None):
        """
        Aligns the tri-spectral power distribution to given shape: Interpolates
        first then extrapolates to fit the given range.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for alignment.
        method : unicode, optional
            ('Linear', 'Constant'),
            Extrapolation method.
        left : numeric, optional
            Value to return for low extrapolation range.
        right : numeric, optional
            Value to return for high extrapolation range.

        Returns
        -------
        TriSpectralPowerDistribution
            Aligned tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.extrapolate,
        TriSpectralPowerDistribution.interpolate

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19, 550: 89.76, 560: 90.28}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45, 550: 15.34, 560: 10.11}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28, 550: 91.61, 560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd.align(colour.SpectralShape(505, 565, 1))
        >>> tri_spd.wavelengths
        array([ 505.,  506.,  507.,  508.,  509.,  510.,  511.,  512.,  513.,
                514.,  515.,  516.,  517.,  518.,  519.,  520.,  521.,  522.,
                523.,  524.,  525.,  526.,  527.,  528.,  529.,  530.,  531.,
                532.,  533.,  534.,  535.,  536.,  537.,  538.,  539.,  540.,
                541.,  542.,  543.,  544.,  545.,  546.,  547.,  548.,  549.,
                550.,  551.,  552.,  553.,  554.,  555.,  556.,  557.,  558.,
                559.,  560.,  561.,  562.,  563.,  564.,  565.])
        >>> tri_spd.values
        array([[ 49.67      ,  90.56      ,  12.43      ],
               [ 49.67      ,  90.56      ,  12.43      ],
               [ 49.67      ,  90.56      ,  12.43      ],
               [ 49.67      ,  90.56      ,  12.43      ],
               [ 49.67      ,  90.56      ,  12.43      ],
               [ 49.67      ,  90.56      ,  12.43      ],
               [ 51.83259386,  91.29949284,  12.53771846],
               [ 53.98419529,  91.95023878,  12.72331935],
               [ 56.12054529,  92.53954635,  12.96516791],
               [ 58.23153952,  93.01500371,  13.31237772],
               [ 60.30332087,  93.27163315,  13.86051361],
               [ 62.32037198,  93.17904559,  14.72729447],
               [ 64.26760777,  92.60859514,  16.02829619],
               [ 66.13246798,  91.46053359,  17.85265447],
               [ 67.90700972,  89.69116496,  20.23876774],
               [ 69.59      ,  87.34      ,  23.15      ],
               [ 71.18373788,  84.48680331,  26.51504699],
               [ 72.68000567,  81.0666018 ,  30.39648521],
               [ 74.07534838,  77.07662549,  34.7958422 ],
               [ 75.37403437,  72.61538702,  39.61788586],
               [ 76.58560089,  67.84907147,  44.70268055],
               [ 77.72239954,  62.97792612,  49.85764325],
               [ 78.79714184,  58.20265032,  54.88959973],
               [ 79.82044471,  53.69078522,  59.63684065],
               [ 80.798376  ,  49.54310369,  64.00117776],
               [ 81.73      ,  45.76      ,  67.98      ],
               [ 82.60936063,  42.26785344,  71.64608931],
               [ 83.439232  ,  39.10608   ,  74.976976  ],
               [ 84.22200712,  36.30637281,  77.94505894],
               [ 84.956896  ,  33.85464   ,  80.552     ],
               [ 85.64101563,  31.70511719,  82.82035156],
               [ 86.27048   ,  29.79448   ,  84.785184  ],
               [ 86.84149013,  28.05595656,  86.48571319],
               [ 87.351424  ,  26.43344   ,  87.956928  ],
               [ 87.79992663,  24.89560094,  89.22121781],
               [ 88.19      ,  23.45      ,  90.28      ],
               [ 88.52650365,  22.14240911,  91.10391339],
               [ 88.80908034,  20.99452341,  91.65380356],
               [ 89.03932796,  20.00217875,  91.93334992],
               [ 89.22228172,  19.14733703,  91.9858818 ],
               [ 89.36529549,  18.40281792,  91.88110029],
               [ 89.4769231 ,  17.73703066,  91.70180004],
               [ 89.56579964,  17.11870586,  91.53059104],
               [ 89.63952276,  16.52162725,  91.43662048],
               [ 89.70353398,  15.92936351,  91.46229449],
               [ 89.76      ,  15.34      ,  91.61      ],
               [ 89.80940416,  14.76591774,  91.85286166],
               [ 89.85788907,  14.21291904,  92.20917374],
               [ 89.90963075,  13.6795969 ,  92.69296649],
               [ 89.96529707,  13.16135104,  93.29883774],
               [ 90.02324985,  12.65198116,  94.00787866],
               [ 90.08074679,  12.14528008,  94.79359954],
               [ 90.13514354,  11.63662697,  95.62785553],
               [ 90.18509566,  11.12458059,  96.48677244],
               [ 90.23176065,  10.61247244,  97.35667247],
               [ 90.28      ,  10.11      ,  98.24      ],
               [ 90.28      ,  10.11      ,  98.24      ],
               [ 90.28      ,  10.11      ,  98.24      ],
               [ 90.28      ,  10.11      ,  98.24      ],
               [ 90.28      ,  10.11      ,  98.24      ],
               [ 90.28      ,  10.11      ,  98.24      ]])
        """

        for i in self.__mapping.keys():
            getattr(self, i).align(shape, method, left, right)

        return self

    def zeros(self, shape=SpectralShape()):
        """
        Zeros fills the tri-spectral power distribution: Missing values will be
        replaced with zeros to fit the defined range.

        Parameters
        ----------
        shape : SpectralShape, optional
            Spectral shape used for zeros fill.

        Returns
        -------
        TriSpectralPowerDistribution
            Zeros filled tri-spectral power distribution.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19, 550: 89.76, 560: 90.28}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45, 550: 15.34, 560: 10.11}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28, 550: 91.61, 560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd.zeros(colour.SpectralShape(505, 565, 1))
        >>> tri_spd.values
        array([[  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [ 49.67,  90.56,  12.43],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [ 69.59,  87.34,  23.15],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [ 81.73,  45.76,  67.98],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [ 88.19,  23.45,  90.28],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [ 89.76,  15.34,  91.61],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [ 90.28,  10.11,  98.24],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ],
               [  0.  ,   0.  ,   0.  ]])
        """

        for i in self.__mapping.keys():
            getattr(self, i).zeros(shape)

        return self

    def normalise(self, factor=1):
        """
        Normalises the tri-spectral power distribution with given normalization
        factor.

        Parameters
        ----------
        factor : numeric, optional
            Normalization factor

        Returns
        -------
        TriSpectralPowerDistribution
            Normalised tri- spectral power distribution.

        Notes
        -----
        -   The implementation uses the maximum value for all axis.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19, 550: 89.76, 560: 90.28}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45, 550: 15.34, 560: 10.11}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28, 550: 91.61, 560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> tri_spd.normalise()
        >>> tri_spd.values
        array([[ 0.50559853,  0.9218241 ,  0.12652687],
               [ 0.70836726,  0.88904723,  0.23564739],
               [ 0.83194218,  0.46579805,  0.69197883],
               [ 0.89769951,  0.23870114,  0.91897394],
               [ 0.91368078,  0.15614821,  0.93251221],
               [ 0.91897394,  0.10291124,  1.        ]])
        """

        maximum = max(np.ravel(self.values))
        for i in self.__mapping.keys():
            getattr(self, i) * (1 / maximum) * factor

        return self

    def clone(self):
        """
        Clones the tri-spectral power distribution.

        Most of the :class:`TriSpectralPowerDistribution` class operations are
        conducted in-place. The :meth:`TriSpectralPowerDistribution.clone`
        method provides a convenient way to copy the tri-spectral power
        distribution to a new object.

        Returns
        -------
        TriSpectralPowerDistribution
            Cloned tri-spectral power distribution.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19, 550: 89.76, 560: 90.28}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45, 550: 15.34, 560: 10.11}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28, 550: 91.61, 560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mpg = lbl = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = colour.TriSpectralPowerDistribution('Tri Spd', data, mpg, lbl)
        >>> print(id(tri_spd))
        >>> tri_spd_clone = tri_spd.clone()
        >>> print(id(tri_spd_clone))
        4412008784
        4411975312
        """

        return copy.deepcopy(self)


def constant_spd(k,
                 shape=DEFAULT_SPECTRAL_SHAPE):
    """
    Returns a spectral power distribution of given spectral shape filled with
    constant :math:`k` values.

    Parameters
    ----------
    k : numeric
        Constant :math:`k` to fill the spectral power distribution with.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.

    Returns
    -------
    SpectralPowerDistribution
        Constant :math:`k` to filled spectral power distribution.

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`DEFAULT_SPECTRAL_SHAPE` attribute.

    Examples
    --------
    >>> spd = colour.constant_spd(100)
    >>> spd.shape
    SpectralShape(360, 830, 1)
    >>> spd[400]
    100.0
    """

    wavelengths = shape.range()
    values = np.full(len(wavelengths), k)

    name = '{0} Constant'.format(k)
    return SpectralPowerDistribution(name, dict(zip(wavelengths, values)))


def zeros_spd(shape=DEFAULT_SPECTRAL_SHAPE):
    """
    Returns a spectral power distribution of given spectral shape filled with
    zeros.

    Parameters
    ----------
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.

    Returns
    -------
    SpectralPowerDistribution
        Zeros filled spectral power distribution.

    See Also
    --------
    constant_spd

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`DEFAULT_SPECTRAL_SHAPE` attribute.

    Examples
    --------
    >>> spd = colour.zeros_spd()
    >>> spd.shape
    SpectralShape(360, 830, 1)
    >>> spd[400]
    0.0
    """

    return constant_spd(0, shape)


def ones_spd(shape=DEFAULT_SPECTRAL_SHAPE):
    """
    Returns a spectral power distribution of given spectral shape filled with
    ones.

    Parameters
    ----------
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.

    Returns
    -------
    SpectralPowerDistribution
        Ones filled spectral power distribution.

    See Also
    --------
    constant_spd

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`DEFAULT_SPECTRAL_SHAPE` attribute.

    Examples
    --------
    >>> spd = colour.ones_spd()
    >>> spd.shape
    SpectralShape(360, 830, 1)
    >>> spd[400]
    1.0
    """

    return constant_spd(1, shape)