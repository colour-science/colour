#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spectrum
========

Defines the classes handling spectral data computation:

-   :class:`SpectralMapping`
-   :class:`SpectralShape`
-   :class:`SpectralPowerDistribution`
-   :class:`TriSpectralPowerDistribution`

See Also
--------
`Spectrum Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/spectrum.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np
import operator
import pprint
from six.moves import zip

from colour.algebra import (
    Extrapolator,
    LinearInterpolator,
    SpragueInterpolator,
    CubicSplineInterpolator,
    PchipInterpolator)
from colour.utilities import (
    ArbitraryPrecisionMapping,
    is_iterable,
    is_numeric,
    is_string,
    is_uniform,
    interval,
    tsplit,
    tstack,
    warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['DEFAULT_WAVELENGTH_DECIMALS',
           'SpectralMapping',
           'SpectralShape',
           'SpectralPowerDistribution',
           'TriSpectralPowerDistribution',
           'DEFAULT_SPECTRAL_SHAPE',
           'constant_spd',
           'zeros_spd',
           'ones_spd']

DEFAULT_WAVELENGTH_DECIMALS = 10
"""
Default wavelength precision decimals.

DEFAULT_WAVELENGTH_DECIMALS : int
"""


class SpectralMapping(ArbitraryPrecisionMapping):
    """
    Defines the base mapping for spectral data.

    It enables usage of floating point wavelengths as keys by rounding them at
    a specific decimals count.

    Parameters
    ----------
    data : dict or SpectralMapping, optional
        Spectral data in a *dict* or *SpectralMapping* as follows:
        {wavelength :math:`\lambda_{i}`: value,
        wavelength :math:`\lambda_{i+1}`: value,
        ...,
        wavelength :math:`\lambda_{i+n}`: value}
    wavelength_decimals : int, optional
        Decimals count the keys will be rounded at.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        Key / Value pairs to store into the mapping at initialisation.

    Attributes
    ----------
    wavelength_decimals

    Examples
    --------
    >>> data1 = {380.1999999998: 0.000039, 380.2000000000: 0.000039}
    >>> mapping = SpectralMapping(data1, wavelength_decimals=10)
    >>> # Doctests skip for Python 2.x compatibility.
    >>> tuple(mapping.keys())  # doctest: +SKIP
    (380.1999999..., 380.2)
    >>> mapping = SpectralMapping(data1, wavelength_decimals=7)
    >>> # Doctests skip for Python 2.x compatibility.
    >>> tuple(mapping.keys())  # doctest: +SKIP
    (380.2,)
    """

    def __init__(self,
                 data=None,
                 wavelength_decimals=DEFAULT_WAVELENGTH_DECIMALS,
                 **kwargs):
        super(SpectralMapping, self).__init__(
            data, wavelength_decimals, **kwargs)

    @property
    def wavelength_decimals(self):
        """
        Property for **self.key_decimals** attribute.

        Returns
        -------
        unicode
            self.key_decimals.
        """

        return self.key_decimals

    @wavelength_decimals.setter
    def wavelength_decimals(self, value):
        """
        Setter for **self.key_decimals** attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, int), (
                '"{0}" attribute: "{1}" is not a "int" instance!').format(
                'wavelength_decimals', value)
        self.key_decimals = value


class SpectralShape(object):
    """
    Defines the base object for spectral power distribution shape.

    Parameters
    ----------
    start : numeric, optional
        Wavelength :math:`\lambda_{i}` range start in nm.
    end : numeric, optional
        Wavelength :math:`\lambda_{i}` range end in nm.
    interval : numeric, optional
        Wavelength :math:`\lambda_{i}` range interval.

    Attributes
    ----------
    start
    end
    interval
    boundaries

    Methods
    -------
    __str__
    __repr__
    __iter__
    __contains__
    __len__
    __eq__
    __ne__
    range

    Examples
    --------
    >>> # Doctests skip for Python 2.x compatibility.
    >>> SpectralShape(360, 830, 1)  # doctest: +SKIP
    SpectralShape(360, 830, 1)
    """

    def __init__(self, start=None, end=None, interval=None):
        # Attribute storing the spectral shape range for caching purpose.
        self._range = None

        self._start = None
        self._end = None
        self._interval = None
        self.start = start
        self.end = end
        self.interval = interval

    @property
    def start(self):
        """
        Property for **self._start** private attribute.

        Returns
        -------
        numeric
            self._start.
        """

        return self._start

    @start.setter
    def start(self, value):
        """
        Setter for **self._start** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" is not a "numeric"!'.format(
                    'start', value))

            value = round(value, DEFAULT_WAVELENGTH_DECIMALS)

            if self._end is not None:
                assert value < self._end, (
                    '"{0}" attribute value must be strictly less than '
                    '"{1}"!'.format('start', self._end))

        # Invalidating the *range* cache.
        if value != self._start:
            self._range = None

        self._start = value

    @property
    def end(self):
        """
        Property for **self._end** private attribute.

        Returns
        -------
        numeric
            self._end.
        """

        return self._end

    @end.setter
    def end(self, value):
        """
        Setter for **self._end** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" is not a "numeric"!'.format(
                    'end', value))

            value = round(value, DEFAULT_WAVELENGTH_DECIMALS)

            if self._start is not None:
                assert value > self._start, (
                    '"{0}" attribute value must be strictly greater than '
                    '"{1}"!'.format('end', self._start))

        # Invalidating the *range* cache.
        if value != self._end:
            self._range = None

        self._end = value

    @property
    def interval(self):
        """
        Property for **self._interval** private attribute.

        Returns
        -------
        numeric
            self._interval.
        """

        return self._interval

    @interval.setter
    def interval(self, value):
        """
        Setter for **self._interval** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" is not a "numeric"!'.format(
                    'interval', value))

            value = round(value, DEFAULT_WAVELENGTH_DECIMALS)

        # Invalidating the *range* cache.
        if value != self._interval:
            self._range = None

        self._interval = value

    @property
    def boundaries(self):
        """
        Property for **self._start** and **self._end** private attributes.

        Returns
        -------
        tuple
            self._start, self._end.
        """

        return self._start, self._end

    @boundaries.setter
    def boundaries(self, value):
        """
        Setter for **self._boundaries** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            assert is_iterable(value), (
                '"{0}" attribute: "{1}" is not an "iterable"!'.format(
                    'boundaries', value))

            assert len(value) == 2, (
                '"{0}" attribute: "{1}" must have exactly '
                'two elements!'.format('boundaries', value))

            start, end = value
            self.start = start
            self.end = end

    def __str__(self):
        """
        Returns a nice formatted string representation.

        Returns
        -------
        unicode
            Nice formatted string representation.
        """

        return '({0}, {1}, {2})'.format(
            self._start, self._end, self._interval)

    def __repr__(self):
        """
        Returns a formatted string representation.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return 'SpectralShape({0}, {1}, {2})'.format(
            self._start, self._end, self._interval)

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
        >>> for wavelength in shape: print(wavelength)
        0.0
        1.0
        2.0
        3.0
        4.0
        5.0
        6.0
        7.0
        8.0
        9.0
        10.0
        """

        return iter(self.range())

    def __contains__(self, wavelength):
        """
        Returns if the spectral shape contains given wavelength
        :math:`\lambda`.

        Parameters
        ----------
        wavelength : numeric or array_like
            Wavelength :math:`\lambda`.

        Returns
        -------
        bool
            Is wavelength :math:`\lambda` contained in the spectral shape.

        Warning
        -------
        *wavelength* argument is tested to be contained in the spectral shape
        within the tolerance defined by :attr:`colour.constants.common.EPSILON`
        attribute value.

        Notes
        -----
        -   Reimplements the :meth:`object.__contains__` method.

        Examples
        --------
        >>> 0.5 in SpectralShape(0, 10, 0.1)
        True
        >>> 0.6 in SpectralShape(0, 10, 0.1)
        True
        >>> 0.51 in SpectralShape(0, 10, 0.1)
        False
        >>> np.array([0.5, 0.6]) in SpectralShape(0, 10, 0.1)
        True
        >>> np.array([0.51, 0.6]) in SpectralShape(0, 10, 0.1)
        False
        """

        return np.all(np.in1d(wavelength, self.range()))

    def __len__(self):
        """
        Returns the spectral shape wavelength :math:`\lambda_n` count.

        Returns
        -------
        int
            Spectral shape wavelength :math:`\lambda_n` count.

        Notes
        -----
        -   Reimplements the :meth:`object.__len__` method.

        Examples
        --------
        >>> len(SpectralShape(0, 10, 0.1))
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

        return (isinstance(shape, self.__class__) and
                np.array_equal(self.range(), shape.range()))

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

        Raises
        ------
        RuntimeError
            If one of spectral shape *start*, *end* or *interval* attributes is
            not defined.

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

        if None in (self._start, self._end, self._interval):
            raise RuntimeError(('One of the spectral shape "start", "end" or '
                                '"interval" attributes is not defined!'))

        if self._range is None:
            samples = round((self._interval + self._end - self._start) /
                            self._interval)
            range_, current_interval = np.linspace(
                self._start, self._end, samples, retstep=True)
            self._range = np.around(range_, DEFAULT_WAVELENGTH_DECIMALS)

            if current_interval != self._interval:
                self._interval = current_interval
                warning(('"{0}" shape could not be honored, using '
                         '"{1}"!').format(
                    (self._start, self._end, self._interval), self))

        return self._range


class SpectralPowerDistribution(object):
    """
    Defines the base object for spectral data computations.

    Parameters
    ----------
    name : unicode
        Spectral power distribution name.
    data : dict or SpectralMapping
        Spectral power distribution data in a *dict* or
        *SpectralMapping* as follows:
        {wavelength :math:`\lambda_{i}`: value,
        wavelength :math:`\lambda_{i+1}`: value,
        ...,
        wavelength :math:`\lambda_{i+n}`: value}
    title : unicode, optional
        Spectral power distribution title for figures.

    Notes
    -----
    -   Underlying spectral data is stored within a `colour.SpectralMapping`
        class mapping which implies that wavelengths keys will be rounded.

    Attributes
    ----------
    name
    data
    title
    wavelengths
    values
    items
    shape

    Methods
    -------
    __str__
    __repr__
    __hash__
    __init__
    __getitem__
    __setitem__
    __iter__
    __contains__
    __len__
    __eq__
    __ne__
    __add__
    __iadd__
    __sub__
    __isub__
    __mul__
    __imul__
    __div__
    __idiv__
    __pow__
    __ipow__
    get
    is_uniform
    extrapolate
    interpolate
    align
    trim_wavelengths
    zeros
    normalise
    clone

    Examples
    --------
    >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
    >>> spd = SpectralPowerDistribution('Sample', data)
    >>> # Doctests skip for Python 2.x compatibility.
    >>> spd.wavelengths  # doctest: +SKIP
    array([510, 520, 530, 540])
    >>> spd.values
    array([ 49.67,  69.59,  81.73,  88.19])
    >>> spd.shape  # doctest: +SKIP
    SpectralShape(510, 540, 10)
    """

    def __init__(self, name, data, title=None):
        self._name = None
        self.name = name
        self._data = None
        self.data = data
        self._title = None
        self.title = title

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
            assert is_string(value), (
                ('"{0}" attribute: "{1}" is not a '
                 '"string" like object!').format('name', value))
        self._name = value

    @property
    def data(self):
        """
        Property for **self._data** private attribute.

        Returns
        -------
        SpectralMapping
            self._data.
        """

        return self._data

    @data.setter
    def data(self, value):
        """
        Setter for **self._data** private attribute.

        Parameters
        ----------
        value : dict or SpectralMapping
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, (dict, SpectralMapping)), (
                '"{0}" attribute: "{1}" is not a "dict" or "SpectralMapping" '
                'instance!'.format('data', value))
        self._data = SpectralMapping(value)

    @property
    def title(self):
        """
        Property for **self._title** private attribute.

        Returns
        -------
        unicode
            self._title.
        """

        if self._title is not None:
            return self._title
        else:
            return self._name

    @title.setter
    def title(self, value):
        """
        Setter for **self._title** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert is_string(value), (
                ('"{0}" attribute: "{1}" is not a '
                 '"string" like object!').format('title', value))
        self._title = value

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

        return np.array(sorted(self._data.keys()))

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

        return self[self.wavelengths]

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
        ndarray
            Spectral power distribution data.
        """

        return np.array(list(self.__iter__()))

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
            different interval, in that case
            :attr:`SpectralPowerDistribution.shape` returns the *minimum*
            interval size.

        Warning
        -------
        :attr:`SpectralPowerDistribution.shape` is read only.

        Examples
        --------
        Uniform spectral power distribution:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> SpectralPowerDistribution(  # doctest: +ELLIPSIS
        ...     'Sample', data).shape
        SpectralShape(510..., 540..., 10...)

        Non uniform spectral power distribution:

        >>> data = {512.3: 49.67, 524.5: 69.59, 532.4: 81.73, 545.7: 88.19}
        >>> # Doctests ellipsis for Python 2.x compatibility.
        >>> SpectralPowerDistribution(  # doctest: +ELLIPSIS
        ...     'Sample', data).shape
        SpectralShape(512.3, 545.7, 7...)
        """

        wavelengths = self.wavelengths

        return SpectralShape(min(wavelengths),
                             max(wavelengths),
                             min(interval(wavelengths)))

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

    def __str__(self):
        """
        Returns a pretty formatted string representation of the spectral power
        distribution.

        Returns
        -------
        unicode
            Pretty formatted string representation.

        See Also
        --------
        SpectralPowerDistribution.__repr__

        Notes
        -----
        -   Reimplements the :meth:`object.__str__` method.

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> print(  # doctest: +ELLIPSIS
        ...     SpectralPowerDistribution('Sample', data))
        SpectralPowerDistribution('Sample', (510..., 540..., 10...))
        """

        return '{0}(\'{1}\', {2})'.format(self.__class__.__name__,
                                          self._name,
                                          str(self.shape))

    def __repr__(self):
        """
        Returns a formatted string representation of the spectral power
        distribution.

        Returns
        -------
        unicode
            Formatted string representation.

        See Also
        --------
        SpectralPowerDistribution.__str__

        Notes
        -----
        -   Reimplements the :meth:`object.__repr__` method.

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> SpectralPowerDistribution('Sample', data)  # doctest: +ELLIPSIS
        SpectralPowerDistribution(
            'Sample',
            {510...: 49.67, 520...: 69.59, 530...: 81.73, 540...: 88.19})
        """

        return '{0}(\n    \'{1}\',\n    {2})'.format(
            self.__class__.__name__,
            self._name,
            pprint.pformat(dict(self.data)).replace('\n', '\n    '))

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
        .. [1]  Hettinger, R. (n.d.). Python hashable dicts. Retrieved August
                08, 2014, from http://stackoverflow.com/a/16162138/931625
        """

        return hash(frozenset(self._data))

    def __getitem__(self, wavelength):
        """
        Returns the value for given wavelength :math:`\lambda`.

        Parameters
        ----------
        wavelength: numeric, array_like or slice
            Wavelength :math:`\lambda` to retrieve the value.

        Returns
        -------
        numeric or ndarray
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
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> # Doctests ellipsis for Python 2.x compatibility.
        >>> spd[510]  # doctest: +ELLIPSIS
        array(49.67...)
        >>> spd[np.array([510, 520])]
        array([ 49.67,  69.59])
        >>> spd[:]
        array([ 49.67,  69.59,  81.73,  88.19])
        """

        if isinstance(wavelength, slice):
            return self.values[wavelength]
        else:
            wavelength = np.asarray(wavelength)

            value = [self.data[x] for x in np.ravel(wavelength)]
            value = np.reshape(value, wavelength.shape)

            return value

    def __setitem__(self, wavelength, value):
        """
        Sets the wavelength :math:`\lambda` with given value.

        Parameters
        ----------
        wavelength : numeric, array_like or slice
            Wavelength :math:`\lambda` to set.
        value : numeric or array_like
            Value for wavelength :math:`\lambda`.

        Warning
        -------
        *value* parameter is resized to match *wavelength* parameter size.

        Notes
        -----
        -   Reimplements the :meth:`object.__setitem__` method.

        Examples
        --------
        >>> spd = SpectralPowerDistribution('Sample', {})
        >>> spd[510] = 49.67
        >>> spd.values
        array([ 49.67])
        >>> spd[np.array([520, 530])] = np.array([69.59, 81.73])
        >>> spd.values
        array([ 49.67,  69.59,  81.73])
        >>> spd[np.array([540, 550])] = 88.19
        >>> spd.values
        array([ 49.67,  69.59,  81.73,  88.19,  88.19])
        >>> spd[:] = 49.67
        >>> spd.values
        array([ 49.67,  49.67,  49.67,  49.67,  49.67])
        """

        if is_numeric(wavelength) or is_iterable(wavelength):
            wavelengths = np.ravel(wavelength)
        elif isinstance(wavelength, slice):
            wavelengths = self.wavelengths[wavelength]
        else:
            raise NotImplementedError(
                '"{0}" type is not supported for indexing!'.format(
                    type(wavelength)))

        values = np.resize(value, wavelengths.shape)
        for i in range(len(wavelengths)):
            self._data.__setitem__(wavelengths[i], values[i])

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
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> # Doctests ellipsis for Python 2.x compatibility.
        >>> for wavelength, value in spd:  # doctest: +SKIP
        ...     print(wavelength, value)
        (510, 49.6...)
        (520, 69.5...)
        (530, 81.7...)
        (540, 88.1...)
        """

        return iter(sorted(self._data.items()))

    def __contains__(self, wavelength):
        """
        Returns if the spectral power distribution contains given wavelength
        :math:`\lambda`.

        Parameters
        ----------
        wavelength : numeric or array_like
            Wavelength :math:`\lambda`.

        Returns
        -------
        bool
            Is wavelength :math:`\lambda` contained in the spectral power
            distribution.

        Warning
        -------
        *wavelength* argument is tested to be contained in the spectral power
        distribution within the tolerance defined by
        :attr:`colour.constants.common.EPSILON` attribute value.

        Notes
        -----
        -   Reimplements the :meth:`object.__contains__` method.

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> 510 in spd
        True
        >>> np.array([510, 520]) in spd
        True
        >>> np.array([510, 520, 521]) in spd
        False
        """

        return np.all(np.in1d(wavelength, self.wavelengths))

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
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> len(spd)
        4
        """

        return len(self._data)

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
        >>> spd1 = SpectralPowerDistribution('Sample', data1)
        >>> spd2 = SpectralPowerDistribution('Sample', data2)
        >>> spd3 = SpectralPowerDistribution('Sample', data2)
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
        >>> spd1 = SpectralPowerDistribution('Sample', data1)
        >>> spd2 = SpectralPowerDistribution('Sample', data2)
        >>> spd3 = SpectralPowerDistribution('Sample', data2)
        >>> spd1 != spd2
        True
        >>> spd2 != spd3
        False
        """

        return not (self == spd)

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
        SpectralPowerDistribution.__iadd__

        Notes
        -----
        -   Reimplements the :meth:`object.__add__` method.

        Examples
        --------
        Adding a single *numeric* variable:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> spd = spd + 10
        >>> spd.values
        array([ 59.67,  79.59,  91.73,  98.19])

        Adding an *array_like* variable:

        >>> spd = spd + [1, 2, 3, 4]
        >>> spd.values
        array([  60.67,   81.59,   94.73,  102.19])

        Adding a :class:`SpectralPowerDistribution` class variable:

        >>> spd_alternate = SpectralPowerDistribution('Sample', data)
        >>> spd = spd + spd_alternate
        >>> spd.values
        array([ 110.34,  151.18,  176.46,  190.38])
        """

        return self._arithmetical_operation(x, operator.add)

    def __iadd__(self, x):
        """
        Implements support for in-place spectral power distribution addition.

        Usage is similar to the regular *addition* operation but make use of
        the *augmented assignement* operator such as: `spd += 10` instead of
        `spd + 10`.

        Parameters
        ----------
        x : numeric or array_like or SpectralPowerDistribution
            Variable to in-place add.

        Returns
        -------
        SpectralPowerDistribution
            Variable in-place added spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.__add__

        Notes
        -----
        -   Reimplements the :meth:`object.__iadd__` method.
        """

        return self._arithmetical_operation(x, operator.add, True)

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
        SpectralPowerDistribution.__isub__

        Notes
        -----
        -   Reimplements the :meth:`object.__sub__` method.

        Examples
        --------
        Subtracting a single *numeric* variable:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> spd = spd - 10
        >>> spd.values
        array([ 39.67,  59.59,  71.73,  78.19])

        Subtracting an *array_like* variable:

        >>> spd = spd - [1, 2, 3, 4]
        >>> spd.values
        array([ 38.67,  57.59,  68.73,  74.19])

        Subtracting a :class:`SpectralPowerDistribution` class variable:

        >>> spd_alternate = SpectralPowerDistribution('Sample', data)
        >>> spd = spd - spd_alternate
        >>> spd.values
        array([-11., -12., -13., -14.])
        """

        return self._arithmetical_operation(x, operator.sub)

    def __isub__(self, x):
        """
        Implements support for in-place spectral power distribution
        subtraction.

        Usage is similar to the regular *subtraction* operation but make use of
        the *augmented assignement* operator such as: `spd -= 10` instead of
        `spd - 10`.

        Parameters
        ----------
        x : numeric or array_like or SpectralPowerDistribution
            Variable to in-place subtract.

        Returns
        -------
        SpectralPowerDistribution
            Variable in-place subtracted spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.__sub__

        Notes
        -----
        -   Reimplements the :meth:`object.__isub__` method.
        """

        return self._arithmetical_operation(x, operator.sub, True)

    def __mul__(self, x):
        """
        Implements support for spectral power distribution multiplication.

        Parameters
        ----------
        x : numeric or array_like or SpectralPowerDistribution
            Variable to multiply by.

        Returns
        -------
        SpectralPowerDistribution
            Variable multiplied spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.__imul__

        Notes
        -----
        -   Reimplements the :meth:`object.__mul__` method.

        Examples
        --------
        Multiplying a single *numeric* variable:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> spd = spd * 10
        >>> spd.values
        array([ 496.7,  695.9,  817.3,  881.9])

        Multiplying an *array_like* variable:

        >>> spd = spd * [1, 2, 3, 4]
        >>> spd.values
        array([  496.7,  1391.8,  2451.9,  3527.6])

        Multiplying a :class:`SpectralPowerDistribution` class variable:

        >>> spd_alternate = SpectralPowerDistribution('Sample', data)
        >>> spd = spd * spd_alternate
        >>> spd.values
        array([  24671.089,   96855.362,  200393.787,  311099.044])
        """

        return self._arithmetical_operation(x, operator.mul)

    def __imul__(self, x):
        """
        Implements support for in-place spectral power distribution
        multiplication.

        Usage is similar to the regular *multiplication* operation but make use
        of the *augmented assignement* operator such as: `spd *= 10` instead of
        `spd * 10`.

        Parameters
        ----------
        x : numeric or array_like or SpectralPowerDistribution
            Variable to in-place multiply by.

        Returns
        -------
        SpectralPowerDistribution
            Variable in-place multiplied spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.__mul__

        Notes
        -----
        -   Reimplements the :meth:`object.__imul__` method.
        """

        return self._arithmetical_operation(x, operator.mul, True)

    def __div__(self, x):
        """
        Implements support for spectral power distribution division.

        Parameters
        ----------
        x : numeric or array_like or SpectralPowerDistribution
            Variable to divide by.

        Returns
        -------
        SpectralPowerDistribution
            Variable divided spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.__idiv__

        Notes
        -----
        -   Reimplements the :meth:`object.__div__` method.

        Examples
        --------
        Dividing a single *numeric* variable:

        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> spd = spd / 10
        >>> spd.values
        array([ 4.967,  6.959,  8.173,  8.819])

        Dividing an *array_like* variable:

        >>> spd = spd / [1, 2, 3, 4]
        >>> spd.values
        array([ 4.967     ,  3.4795    ,  2.72433333,  2.20475   ])

        Dividing a :class:`SpectralPowerDistribution` class variable:

        >>> spd_alternate = SpectralPowerDistribution('Sample', data)
        >>> spd = spd / spd_alternate
        >>> spd.values  # doctest: +ELLIPSIS
        array([ 0.1       ,  0.05      ,  0.0333333...,  0.025     ])
        """

        return self._arithmetical_operation(x, operator.truediv)

    def __idiv__(self, x):
        """
        Implements support for in-place spectral power distribution division.

        Usage is similar to the regular *division* operation but make use of
        the *augmented assignement* operator such as: `spd /= 10` instead of
        `spd / 10`.

        Parameters
        ----------
        x : numeric or array_like or SpectralPowerDistribution
            Variable to in-place divide by.

        Returns
        -------
        SpectralPowerDistribution
            Variable in-place divided spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.__div__

        Notes
        -----
        -   Reimplements the :meth:`object.__idiv_` method.
        """

        return self._arithmetical_operation(x, operator.truediv, True)

    # Python 3 compatibility.
    __itruediv__ = __idiv__
    __truediv__ = __div__

    def __pow__(self, x):
        """
        Implements support for spectral power distribution exponentiation.

        Parameters
        ----------
        x : numeric or array_like or SpectralPowerDistribution
            Variable to exponentiate by.

        Returns
        -------
        SpectralPowerDistribution
            Spectral power distribution raised by power of x.

        See Also
        --------
        SpectralPowerDistribution.__ipow__

        Notes
        -----
        -   Reimplements the :meth:`object.__pow__` method.

        Examples
        --------
        Exponentiation by a single *numeric* variable:

        >>> data = {510: 1.67, 520: 2.59, 530: 3.73, 540: 4.19}
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> spd = spd ** 2
        >>> spd.values
        array([  2.7889,   6.7081,  13.9129,  17.5561])

        Exponentiation by an *array_like* variable:

        >>> spd = spd ** [1, 2, 3, 4]
        >>> spd.values  # doctest: +ELLIPSIS
        array([  2.7889000...e+00,   4.4998605...e+01,   2.6931031...e+03,
                 9.4997501...e+04])

        Exponentiation by a :class:`SpectralPowerDistribution` class variable:

        >>> spd_alternate = SpectralPowerDistribution('Sample', data)
        >>> spd = spd ** spd_alternate
        >>> spd.values  # doctest: +ELLIPSIS
        array([  5.5446356...e+00,   1.9133109...e+04,   6.2351033...e+12,
                 7.1880990...e+20])
        """

        return self._arithmetical_operation(x, operator.pow)

    def __ipow__(self, x):
        """
        Implements support for in-place spectral power distribution
        exponentiation.

        Usage is similar to the regular *exponentiation* operation but make use
        of the *augmented assignement* operator such as: `spd **= 10` instead
        of `spd ** 10`.

        Parameters
        ----------
        x : numeric or array_like or SpectralPowerDistribution
            Variable to in-place exponentiate by.

        Returns
        -------
        SpectralPowerDistribution
            Variable in-place exponentiated spectral power distribution.

        See Also
        --------
        SpectralPowerDistribution.__pow__

        Notes
        -----
        -   Reimplements the :meth:`object.__ipow__` method.
        """

        return self._arithmetical_operation(x, operator.pow, True)

    def _arithmetical_operation(self, x, operation, in_place=False):
        """
        Performs given arithmetical operation on :math:`x` variable, the
        operation can be either performed on a spectral power distribution
        clone or in-place.

        Parameters
        ----------
        x : numeric or ndarray or SpectralPowerDistribution
            Operand.
        operation : object
            Operation to perform.
        in_place : bool, optional
            Operation happens in place.

        Returns
        -------
        SpectralPowerDistribution
            Spectral power distribution.
        """

        if issubclass(type(x), SpectralPowerDistribution):
            x = x.values
        elif is_iterable(x):
            x = np.atleast_1d(x)

        data = SpectralMapping(
            zip(self.wavelengths, operation(self.values, x)))

        if in_place:
            self._data = data
            return self
        else:
            clone = self.clone()
            clone.data = data
            return clone

    def get(self, wavelength, default=np.nan):
        """
        Returns the value for given wavelength :math:`\lambda`.

        Parameters
        ----------
        wavelength : numeric or ndarray
            Wavelength :math:`\lambda` to retrieve the value.
        default : nan or numeric, optional
            Wavelength :math:`\lambda` default value.

        Returns
        -------
        numeric or ndarray
            Wavelength :math:`\lambda` value.

        See Also
        --------
        SpectralPowerDistribution.__getitem__

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> # Doctests ellipsis for Python 2.x compatibility.
        >>> spd.get(510)  # doctest: +ELLIPSIS
        array(49.67...)
        >>> spd.get(511)
        array(nan)
        >>> spd.get(np.array([510, 520]))
        array([ 49.67,  69.59])
        """

        wavelength = np.asarray(wavelength)

        value = [self.data.get(x, default) for x in np.ravel(wavelength)]
        value = np.reshape(value, wavelength.shape)

        return value

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
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> spd.is_uniform()
        True

        Breaking the interval by introducing a new wavelength :math:`\lambda`
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
            **{'Constant', 'Linear'}**,,
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
        .. [2]  CIE TC 1-48. (2004). Extrapolation. In CIE 015:2004
                Colorimetry, 3rd Edition (p. 24). ISBN:978-3-901-90633-6
        .. [3]  CIE TC 1-38. (2005). EXTRAPOLATION. In CIE 167:2005
                Recommended Practice for Tabulating Spectral Data for Use in
                Colour Computations (pp. 19–20). ISBN:978-3-901-90641-1

        Examples
        --------
        >>> data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> spd.extrapolate(  # doctest: +ELLIPSIS
        ...     SpectralShape(400, 700)).shape
        SpectralShape(400..., 700..., 10...)
        >>> spd[400]  # doctest: +ELLIPSIS
        array(49.67...)
        >>> spd[700]  # doctest: +ELLIPSIS
        array(88.1...)
        """

        extrapolator = Extrapolator(
            LinearInterpolator(self.wavelengths, self.values),
            method=method, left=left, right=right)

        spd_shape = self.shape
        for i in np.arange(spd_shape.start,
                           shape.start - spd_shape.interval,
                           -spd_shape.interval):
            self[i] = extrapolator(np.float_(i))
        for i in np.arange(spd_shape.end,
                           shape.end + spd_shape.interval,
                           spd_shape.interval):
            self[i] = extrapolator(np.float_(i))

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
            **{None, 'Cubic Spline', 'Linear', 'Pchip', 'Sprague'}**,
            Enforce given interpolation method.

        Returns
        -------
        SpectralPowerDistribution
            Interpolated spectral power distribution.

        Raises
        ------
        RuntimeError
            If *Sprague (1880)* interpolation method is forced with a
            non-uniformly spaced independent variable.
        ValueError
            If the interpolation method is not defined.

        See Also
        --------
        SpectralPowerDistribution.align

        Notes
        -----
        -   Interpolation will be conducted over boundaries range, if you need
            to extend the range of the spectral power distribution use the
            :meth:`SpectralPowerDistribution.extrapolate` or
            :meth:`SpectralPowerDistribution.align` methods.
        -   *Sprague (1880)* interpolator cannot be used for interpolating
            functions having a non-uniformly spaced independent variable.

        Warning
        -------
        -   If *scipy* is not unavailable the *Cubic Spline* method will
            fallback to legacy *Linear* interpolation.
        -   *Cubic Spline* interpolator requires at least 3 wavelengths
            :math:`\lambda_n` for interpolation.
        -   *Linear* interpolator requires at least 2 wavelengths
            :math:`\lambda_n` for interpolation.
        -   *Pchip* interpolator requires at least 2 wavelengths
            :math:`\lambda_n` for interpolation.
        -   *Sprague (1880)* interpolator requires at least 6 wavelengths
            :math:`\lambda_n` for interpolation.

        References
        ----------
        .. [4]  CIE TC 1-38. (2005). 9. INTERPOLATION. In CIE 167:2005
                Recommended Practice for Tabulating Spectral Data for Use in
                Colour Computations (pp. 14–19). ISBN:978-3-901-90641-1

        Examples
        --------
        Uniform data is using *Sprague (1880)* interpolation by default:

        >>> data = {
        ...     510: 49.67,
        ...     520: 69.59,
        ...     530: 81.73,
        ...     540: 88.19,
        ...     550: 86.26,
        ...     560: 77.18}
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> print(spd.interpolate(SpectralShape(interval=1)))
        SpectralPowerDistribution('Sample', (510.0, 560.0, 1.0))
        >>> spd[515]  # doctest: +ELLIPSIS
        array(60.3121800...)

        Non uniform data is using *Cubic Spline* interpolation by default:

        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> spd[511] = 31.41
        >>> print(spd.interpolate(SpectralShape(interval=1)))
        SpectralPowerDistribution('Sample', (510.0, 560.0, 1.0))
        >>> spd[515]  # doctest: +ELLIPSIS
        array(21.4792222...)

        Enforcing *Linear* interpolation:

        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> print(spd.interpolate(
        ...     SpectralShape(interval=1), method='Linear'))
        SpectralPowerDistribution('Sample', (510.0, 560.0, 1.0))
        >>> spd[515]  # doctest: +ELLIPSIS
        array(59.63...)

        Enforcing *Pchip* interpolation:

        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> print(spd.interpolate(
        ...     SpectralShape(interval=1), method='Pchip'))
        SpectralPowerDistribution('Sample', (510.0, 560.0, 1.0))
        >>> spd[515]  # doctest: +ELLIPSIS
        array(60.7204982...)
        """

        spd_shape = self.shape
        boundaries = zip((shape.start, shape.end, shape.interval),
                         (spd_shape.start, spd_shape.end, spd_shape.interval))
        boundaries = [x[0] if x[0] is not None else x[1] for x in boundaries]
        shape = SpectralShape(*boundaries)

        # Defining proper interpolation bounds.
        # TODO: Provide support for fractional interval like 0.1, etc...
        shape.start = max(shape.start, np.ceil(spd_shape.start))
        shape.end = min(shape.end, np.floor(spd_shape.end))

        wavelengths, values = tsplit(self.items)
        uniform = self.is_uniform()

        if is_string(method):
            method = method.lower()

        if method is None:
            if uniform:
                interpolator = SpragueInterpolator
            else:
                interpolator = CubicSplineInterpolator
        elif method == 'cubic spline':
            interpolator = CubicSplineInterpolator
        elif method == 'linear':
            interpolator = LinearInterpolator
        elif method == 'pchip':
            interpolator = PchipInterpolator
        elif method == 'sprague':
            if not uniform:
                warning(('"Sprague" interpolator should only be used for '
                         'interpolating functions having a uniformly spaced '
                         'independent variable!'))

            interpolator = SpragueInterpolator
        else:
            raise ValueError(
                'Undefined "{0}" interpolator!'.format(method))

        interpolator = interpolator(wavelengths, values)
        wavelengths = shape.range()

        self._data = SpectralMapping(
            zip(wavelengths, interpolator(wavelengths)))

        return self

    def align(self,
              shape,
              interpolation_method=None,
              extrapolation_method='Constant',
              extrapolation_left=None,
              extrapolation_right=None):
        """
        Aligns the spectral power distribution to given spectral shape:
        Interpolates first then extrapolates to fit the given range.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for alignment.
        interpolation_method : unicode, optional
            **{None, 'Cubic Spline', 'Linear', 'Pchip', 'Sprague'}**,
            Enforce given interpolation method.
        extrapolation_method : unicode, optional
            **{'Constant', 'Linear'}**,
            Extrapolation method.
        extrapolation_left : numeric, optional
            Value to return for low extrapolation range.
        extrapolation_right : numeric, optional
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
        >>> data = {
        ...     510: 49.67,
        ...     520: 69.59,
        ...     530: 81.73,
        ...     540: 88.19,
        ...     550: 86.26,
        ...     560: 77.18}
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> print(spd.align(SpectralShape(505, 565, 1)))
        SpectralPowerDistribution('Sample', (505.0, 565.0, 1.0))
        >>> # Doctests skip for Python 2.x compatibility.
        >>> spd.wavelengths  # doctest: +SKIP
        array([505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517,
               518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530,
               531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543,
               544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556,
               557, 558, 559, 560, 561, 562, 563, 564, 565])
        >>> spd.values  # doctest: +ELLIPSIS
        array([ 49.67     ...,  49.67     ...,  49.67     ...,  49.67     ...,
                49.67     ...,  49.67     ...,  51.8341162...,  53.9856467...,
                56.1229464...,  58.2366197...,  60.3121800...,  62.3327095...,
                64.2815187...,  66.1448055...,  67.9143153...,  69.59     ...,
                71.1759958...,  72.6627938...,  74.0465756...,  75.3329710...,
                76.5339542...,  77.6647421...,  78.7406907...,  79.7741932...,
                80.7715767...,  81.73     ...,  82.6407518...,  83.507872 ...,
                84.3326333...,  85.109696 ...,  85.8292968...,  86.47944  ...,
                87.0480863...,  87.525344 ...,  87.9056578...,  88.19     ...,
                88.3858347...,  88.4975634...,  88.5258906...,  88.4696570...,
                88.3266460...,  88.0943906...,  87.7709802...,  87.3558672...,
                86.8506741...,  86.26     ...,  85.5911699...,  84.8503430...,
                84.0434801...,  83.1771110...,  82.2583874...,  81.2951360...,
                80.2959122...,  79.2700525...,  78.2277286...,  77.18     ...,
                77.18     ...,  77.18     ...,  77.18     ...,  77.18     ...])
        """

        self.interpolate(shape, interpolation_method)
        self.extrapolate(shape,
                         extrapolation_method,
                         extrapolation_left,
                         extrapolation_right)

        return self

    def trim_wavelengths(self, shape):
        """
        Trims the spectral power distribution wavelengths to given spectral
        shape.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for trimming.

        Returns
        -------
        SpectralPowerDistribution
            Trimed spectral power distribution.

        Examples
        --------
        >>> data = {
        ...     510: 49.67,
        ...     520: 69.59,
        ...     530: 81.73,
        ...     540: 88.19,
        ...     550: 86.26,
        ...     560: 77.18}
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> print(spd.trim_wavelengths(  # doctest: +SKIP
        ...     SpectralShape(520, 550, 10)))
        SpectralPowerDistribution('Sample', (520.0, 550.0, 10.0))
        >>> # Doctests skip for Python 2.x compatibility.
        >>> spd.wavelengths  # doctest: +SKIP
        array([ 520.,  530.,  540.,  550.])
        """

        for wavelength in set(self.wavelengths).difference(set(shape.range())):
            del self._data[wavelength]

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

        Raises
        ------
        RuntimeError
            If the spectral power distribution cannot be zeros filled.

        Examples
        --------
        >>> data = {
        ...     510: 49.67,
        ...     520: 69.59,
        ...     530: 81.73,
        ...     540: 88.19,
        ...     550: 86.26,
        ...     560: 77.18}
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> print(spd.zeros(SpectralShape(505, 565, 1)))
        SpectralPowerDistribution('Sample', (505.0, 565.0, 1.0))
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
        boundaries = zip((shape.start, shape.end, shape.interval),
                         (spd_shape.start, spd_shape.end, spd_shape.interval))
        boundaries = [x[0] if x[0] is not None else x[1] for x in boundaries]
        shape = SpectralShape(*boundaries)

        data = SpectralMapping(
            [(wavelength, self.get(wavelength, 0))
             for wavelength in shape])

        values_s = max(self.shape.start, shape.start)
        values_e = min(self.shape.end, shape.end)
        values = [self[wavelength] for wavelength in self.wavelengths
                  if values_s <= wavelength <= values_e]
        if not np.all(np.in1d(values, list(data.values()))):
            raise RuntimeError(('"{0}" cannot be zeros filled using "{1}" '
                                'shape!').format(self, shape))
        else:
            self._data = data

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
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> print(spd.normalise())  # doctest: +ELLIPSIS
        SpectralPowerDistribution('Sample', (510..., 540..., 10...))
        >>> spd.values  # doctest: +ELLIPSIS
        array([ 0.5632157...,  0.7890917...,  0.9267490...,  1.        ])
        """

        self *= 1 / max(self.values) * factor

        return self

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
        >>> spd = SpectralPowerDistribution('Sample', data)
        >>> print(spd)  # doctest: +ELLIPSIS
        SpectralPowerDistribution('Sample', (510..., 540..., 10...))
        >>> spd_clone = spd.clone()
        >>> print(spd_clone)  # doctest: +ELLIPSIS
        SpectralPowerDistribution('Sample (...)', (510..., 540..., 10...))
        """

        clone = SpectralPowerDistribution(
            self.name, self.data.data, self.title)

        clone.name = '{0} ({1})'.format(clone.name, id(clone))

        if self._title is None:
            clone.title = self._name

        return clone


class TriSpectralPowerDistribution(object):
    """
    Defines the base object for colour matching functions.

    A compound of three :class:`SpectralPowerDistribution` is used to store
    the underlying axis data.

    Parameters
    ----------
    name : unicode
        Tri-spectral power distribution name.
    data : dict
        Tri-spectral power distribution data.
    mapping : dict
        Tri-spectral power distribution attributes mapping.
    title : unicode, optional
        Tri-spectral power distribution title for figures.
    labels : dict, optional
        Tri-spectral power distribution axis labels mapping for figures.

    Attributes
    ----------
    name
    mapping
    data
    title
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
    __str__
    __repr__
    __hash__
    __init__
    __getitem__
    __setitem__
    __iter__
    __contains__
    __len__
    __eq__
    __ne__
    __add__
    __iadd__
    __sub__
    __isub__
    __mul__
    __imul__
    __div__
    __idiv__
    __pow__
    __ipow__
    get
    is_uniform
    extrapolate
    interpolate
    align
    trim_wavelengths
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
    >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
    >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
    >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
    >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
    >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
    >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
    >>> # Doctests skip for Python 2.x compatibility.
    >>> tri_spd.wavelengths  # doctest: +SKIP
    array([510, 520, 530, 540])
    >>> tri_spd.values
    array([[ 49.67,  90.56,  12.43],
           [ 69.59,  87.34,  23.15],
           [ 81.73,  45.76,  67.98],
           [ 88.19,  23.45,  90.28]])
    >>> # Doctests skip for Python 2.x compatibility.
    >>> tri_spd.shape  # doctest: +SKIP
    SpectralShape(510, 540, 10)
    """

    def __init__(self, name, data, mapping, title=None, labels=None):
        self._name = None
        self.name = name
        self._mapping = None
        self.mapping = mapping
        self._data = None
        self.data = data
        self._title = None
        self.title = title
        self._labels = None
        self.labels = labels

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
            assert is_string(value), (
                ('"{0}" attribute: "{1}" is not a '
                 '"string" like object!').format('name', value))
        self._name = value

    @property
    def mapping(self):
        """
        Property for **self._mapping** private attribute.

        Returns
        -------
        dict
            self._mapping.
        """

        return self._mapping

    @mapping.setter
    def mapping(self, value):
        """
        Setter for **self._mapping** private attribute.

        Parameters
        ----------
        value : dict
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, dict), (
                '"{0}" attribute: "{1}" is not a "dict" instance!'.format(
                    'mapping', value))
            for axis in ('x', 'y', 'z'):
                assert axis in value.keys(), (
                    '"{0}" attribute: "{1}" axis label is missing!'.format(
                        'mapping', axis))
        self._mapping = value

    @property
    def data(self):
        """
        Property for **self._data** private attribute.

        Returns
        -------
        dict
            self._data.
        """

        return self._data

    @data.setter
    def data(self, value):
        """
        Setter for **self._data** private attribute.

        Parameters
        ----------
        value : dict
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, dict), (
                '"{0}" attribute: "{1}" is not a "dict" instance!'.format(
                    'data', value))
            for axis in ('x', 'y', 'z'):
                assert self._mapping[axis] in value.keys(), (
                    '"{0}" attribute: "{1}" axis is missing!'.format(
                        'data', axis))

            data = {}
            for axis in ('x', 'y', 'z'):
                data[axis] = SpectralPowerDistribution(
                    self._mapping[axis],
                    value[self._mapping[axis]])

            wavelengths = data['x'].wavelengths
            np.testing.assert_array_equal(
                wavelengths,
                data['y'].wavelengths,
                err_msg=('"{0}" attribute: "x" and "y" wavelengths are '
                         'different!').format('data'))
            np.testing.assert_array_equal(
                wavelengths,
                data['z'].wavelengths,
                err_msg=('"{0}" attribute: "x" and "z" wavelengths are '
                         'different!').format('data'))

            self._data = data
        else:
            self._data = None

    @property
    def title(self):
        """
        Property for **self._title** private attribute.

        Returns
        -------
        unicode
            self._title.
        """

        if self._title is not None:
            return self._title
        else:
            return self._name

    @title.setter
    def title(self, value):
        """
        Setter for **self._title** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert is_string(value), (
                ('"{0}" attribute: "{1}" is not a '
                 '"string" like object!').format('title', value))
        self._title = value

    @property
    def labels(self):
        """
        Property for **self._labels** private attribute.

        Returns
        -------
        dict
            self._labels.
        """

        if self._labels is not None:
            return self._labels
        else:
            return self._mapping

    @labels.setter
    def labels(self, value):
        """
        Setter for **self._labels** private attribute.

        Parameters
        ----------
        value : dict
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, dict), (
                '"{0}" attribute: "{1}" is not a "dict" instance!'.format(
                    'labels', value))
            for axis in ('x', 'y', 'z'):
                assert axis in value.keys(), (
                    '"{0}" attribute: "{1}" axis label is missing!'.format(
                        'labels', axis))
        self._labels = value

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

        return self._data['x']

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

        return self._data['y']

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

        return self._data['z']

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

        return self[self.wavelengths]

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
        ndarray
            Tri-spectral power distribution data.
        """

        return np.array(list(self.__iter__()))

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
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> tri_spd.shape  # doctest: +ELLIPSIS
        SpectralShape(510..., 540..., 10...)
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

    def __str__(self):
        """
        Returns a pretty formatted string representation of the tri-spectral
        power distribution.

        Returns
        -------
        unicode
            Pretty formatted string representation.

        See Also
        --------
        TriSpectralPowerDistribution.__repr__

        Notes
        -----
        -   Reimplements the :meth:`object.__str__` method.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping  = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> print(TriSpectralPowerDistribution(  # doctest: +ELLIPSIS
        ...     'Observer', data, mapping))
        TriSpectralPowerDistribution('Observer', (510..., 540..., 10...))
        """

        return '{0}(\'{1}\', {2})'.format(self.__class__.__name__,
                                          self._name,
                                          str(self.shape))

    def __repr__(self):
        """
        Returns a formatted string representation of the tri-spectral power
        distribution.

        Returns
        -------
        unicode
            Formatted string representation.

        See Also
        --------
        TriSpectralPowerDistribution.__str__

        Notes
        -----
        -   Reimplements the :meth:`object.__repr__` method.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping  = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> TriSpectralPowerDistribution(  # doctest: +ELLIPSIS
        ...     'Observer', data, mapping)
        TriSpectralPowerDistribution(
            'Observer',
            {...'x_bar': \
{510...: 49.67, 520...: 69.59, 530...: 81.73, 540...: 88.19},
             ...'y_bar': \
{510...: 90.56, 520...: 87.34, 530...: 45.76, 540...: 23.45},
             ...'z_bar': \
{510...: 12.43, 520...: 23.15, 530...: 67.98, 540...: 90.28}},
            {...'x': ...'x_bar', ...'y': ...'y_bar', ...'z': ...'z_bar'},
            None,
            None)
        """

        data = {'x_bar': dict(self.x.data),
                'y_bar': dict(self.y.data),
                'z_bar': dict(self.z.data)}

        return '{0}(\n    \'{1}\',\n    {2},\n    {3},\n    {4},' \
               '\n    {5})'.format(
                self.__class__.__name__,
                self._name,
                pprint.pformat(data).replace('\n', '\n    '),
                pprint.pformat(self.mapping),
                ('\'{0}\''.format(self._title)
                 if self._title is not None else
                 self._title),
                pprint.pformat(self._labels))

    def __hash__(self):
        """
        Returns the spectral power distribution hash value. [1]_

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
        """

        return hash((frozenset(self._data['x']),
                     frozenset(self._data['y']),
                     frozenset(self._data['z'])))

    def __getitem__(self, wavelength):
        """
        Returns the values for given wavelength :math:`\lambda`.

        Parameters
        ----------
        wavelength: numeric, array_like or slice
            Wavelength :math:`\lambda` to retrieve the values.

        Returns
        -------
        ndarray
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
        >>> mapping  = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> tri_spd[510]
        array([ 49.67,  90.56,  12.43])
        >>> tri_spd[np.array([510, 520])]
        array([[ 49.67,  90.56,  12.43],
               [ 69.59,  87.34,  23.15]])
        >>> tri_spd[:]
        array([[ 49.67,  90.56,  12.43],
               [ 69.59,  87.34,  23.15],
               [ 81.73,  45.76,  67.98],
               [ 88.19,  23.45,  90.28]])
        """

        value = tstack(
            (self.x[wavelength], self.y[wavelength], self.z[wavelength]))

        return value

    def __setitem__(self, wavelength, value):
        """
        Sets the wavelength :math:`\lambda` with given value.

        Parameters
        ----------
        wavelength : numeric, array_like or slice
            Wavelength :math:`\lambda` to set.
        value : array_like
            Value for wavelength :math:`\lambda`.

        Warning
        -------
        *value* parameter is resized to match *wavelength* parameter size.

        Notes
        -----
        -   Reimplements the :meth:`object.__setitem__` method.

        Examples
        --------
        >>> x_bar = {}
        >>> y_bar = {}
        >>> z_bar = {}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> tri_spd[510] = np.array([49.67, 49.67, 49.67])
        >>> tri_spd.values
        array([[ 49.67,  49.67,  49.67]])
        >>> tri_spd[np.array([520, 530])] = np.array(
        ... [[69.59, 69.59, 69.59],
        ...  [81.73, 81.73, 81.73]])
        >>> tri_spd.values
        array([[ 49.67,  49.67,  49.67],
               [ 69.59,  69.59,  69.59],
               [ 81.73,  81.73,  81.73]])
        >>> tri_spd[np.array([540, 550])] = 88.19
        >>> tri_spd.values
        array([[ 49.67,  49.67,  49.67],
               [ 69.59,  69.59,  69.59],
               [ 81.73,  81.73,  81.73],
               [ 88.19,  88.19,  88.19],
               [ 88.19,  88.19,  88.19]])
        >>> tri_spd[:] = 49.67
        >>> tri_spd.values
        array([[ 49.67,  49.67,  49.67],
               [ 49.67,  49.67,  49.67],
               [ 49.67,  49.67,  49.67],
               [ 49.67,  49.67,  49.67],
               [ 49.67,  49.67,  49.67]])
        """

        if is_numeric(wavelength) or is_iterable(wavelength):
            wavelengths = np.ravel(wavelength)
        elif isinstance(wavelength, slice):
            wavelengths = self.wavelengths[wavelength]
        else:
            raise NotImplementedError(
                '"{0}" type is not supported for indexing!'.format(
                    type(wavelength)))

        value = np.resize(value, (wavelengths.shape[0], 3))

        self.x[wavelengths] = value[..., 0]
        self.y[wavelengths] = value[..., 1]
        self.z[wavelengths] = value[..., 2]

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
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> # Doctests skip for Python 2.x compatibility.
        >>> for wavelength, value in tri_spd:  # doctest: +SKIP
        ...     print(wavelength, value)
        (510, array([ 49.67,  90.56,  12.43]))
        (520, array([ 69.59,  87.34,  23.15]))
        (530, array([ 81.73,  45.76,  67.98]))
        (540, array([ 88.19,  23.45,  90.28]))
        """

        return iter(zip(self.wavelengths, self.values))

    def __contains__(self, wavelength):
        """
        Returns if the tri-spectral power distribution contains given
        wavelength :math:`\lambda`.

        Parameters
        ----------
        wavelength : numeric or array_like
            Wavelength :math:`\lambda`.

        Returns
        -------
        bool
            Is wavelength :math:`\lambda` contained in the tri-spectral power
            distribution.

        Warning
        -------
        *wavelength* argument is tested to be contained in the tri-spectral
        power distribution within the tolerance defined by
        :attr:`colour.constants.common.EPSILON` attribute value.

        Notes
        -----
        -   Reimplements the :meth:`object.__contains__` method.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> 510 in tri_spd
        True
        >>> np.array([510, 520]) in tri_spd
        True
        >>> np.array([510, 520, 521]) in tri_spd
        False
        """

        return wavelength in self.x

    def __len__(self):
        """
        Returns the tri-spectral power distribution wavelengths
        :math:`\lambda_n` count.

        Returns
        -------
        int
            Tri-Spectral power distribution wavelengths :math:`\lambda_n`
            count.

        Notes
        -----
        -   Reimplements the :meth:`object.__len__` method.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
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
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd1 = TriSpectralPowerDistribution('Observer', data1, mapping)
        >>> tri_spd2 = TriSpectralPowerDistribution('Observer', data2, mapping)
        >>> tri_spd3 = TriSpectralPowerDistribution('Observer', data1, mapping)
        >>> tri_spd1 == tri_spd2
        False
        >>> tri_spd1 == tri_spd3
        True
        """

        if not isinstance(tri_spd, self.__class__):
            return False

        equality = True
        for axis in self._mapping:
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
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd1 = TriSpectralPowerDistribution('Observer', data1, mapping)
        >>> tri_spd2 = TriSpectralPowerDistribution('Observer', data2, mapping)
        >>> tri_spd3 = TriSpectralPowerDistribution('Observer', data1, mapping)
        >>> tri_spd1 != tri_spd2
        True
        >>> tri_spd1 != tri_spd3
        False
        """

        return not (self == tri_spd)

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
        TriSpectralPowerDistribution.__iadd__

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
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> tri_spd = tri_spd + 10
        >>> tri_spd.values
        array([[  59.67,  100.56,   22.43],
               [  79.59,   97.34,   33.15],
               [  91.73,   55.76,   77.98],
               [  98.19,   33.45,  100.28]])

        Adding an *array_like* variable:

        >>> tri_spd = tri_spd + [(1, 2, 3)] * 4
        >>> tri_spd.values
        array([[  60.67,  102.56,   25.43],
               [  80.59,   99.34,   36.15],
               [  92.73,   57.76,   80.98],
               [  99.19,   35.45,  103.28]])

        Adding a :class:`TriSpectralPowerDistribution` class variable:

        >>> data1 = {'x_bar': z_bar, 'y_bar': x_bar, 'z_bar': y_bar}
        >>> tri_spd1 = TriSpectralPowerDistribution('Observer', data1, mapping)
        >>> tri_spd = tri_spd + tri_spd1
        >>> tri_spd.values
        array([[  73.1 ,  152.23,  115.99],
               [ 103.74,  168.93,  123.49],
               [ 160.71,  139.49,  126.74],
               [ 189.47,  123.64,  126.73]])
        """

        return self._arithmetical_operation(x, operator.add)

    def __iadd__(self, x):
        """
        Implements support for in-place tri-spectral power distribution
        addition.

        Usage is similar to the regular *addition* operation but make use of
        the *augmented assignement* operator such as: `tri_spd += 10` instead
        of `tri_spd + 10`.

        Parameters
        ----------
        x : numeric or array_like or TriSpectralPowerDistribution
            Variable to in-place add.

        Returns
        -------
        TriSpectralPowerDistribution
            Variable in-place added tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.__add__

        Notes
        -----
        -   Reimplements the :meth:`object.__iadd__` method.
        """

        return self._arithmetical_operation(x, operator.add, True)

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
        TriSpectralPowerDistribution.__isub__

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
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> tri_spd = tri_spd - 10
        >>> tri_spd.values
        array([[ 39.67,  80.56,   2.43],
               [ 59.59,  77.34,  13.15],
               [ 71.73,  35.76,  57.98],
               [ 78.19,  13.45,  80.28]])

        Subtracting an *array_like* variable:

        >>> tri_spd = tri_spd - [(1, 2, 3)] * 4
        >>> tri_spd.values
        array([[ 38.67,  78.56,  -0.57],
               [ 58.59,  75.34,  10.15],
               [ 70.73,  33.76,  54.98],
               [ 77.19,  11.45,  77.28]])

        Subtracting a :class:`TriSpectralPowerDistribution` class variable:

        >>> data1 = {'x_bar': z_bar, 'y_bar': x_bar, 'z_bar': y_bar}
        >>> tri_spd1 = TriSpectralPowerDistribution('Observer', data1, mapping)
        >>> tri_spd = tri_spd - tri_spd1
        >>> tri_spd.values
        array([[ 26.24,  28.89, -91.13],
               [ 35.44,   5.75, -77.19],
               [  2.75, -47.97,   9.22],
               [-13.09, -76.74,  53.83]])
        """

        return self._arithmetical_operation(x, operator.sub)

    def __isub__(self, x):
        """
        Implements support for in-place tri-spectral power distribution
        subtraction.

        Usage is similar to the regular *subtraction* operation but make use of
        the *augmented assignement* operator such as: `tri_spd -= 10` instead
        of `tri_spd - 10`.

        Parameters
        ----------
        x : numeric or array_like or TriSpectralPowerDistribution
            Variable to in-place subtract.

        Returns
        -------
        TriSpectralPowerDistribution
            Variable in-place subtracted tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.__sub__

        Notes
        -----
        -   Reimplements the :meth:`object.__isub__` method.
        """

        return self._arithmetical_operation(x, operator.sub, True)

    def __mul__(self, x):
        """
        Implements support for tri-spectral power distribution multiplication.

        Parameters
        ----------
        x : numeric or array_like or TriSpectralPowerDistribution
            Variable to multiply by.

        Returns
        -------
        TriSpectralPowerDistribution
            Variable multiplied tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.__imul__

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
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> tri_spd = tri_spd * 10
        >>> tri_spd.values
        array([[ 496.7,  905.6,  124.3],
               [ 695.9,  873.4,  231.5],
               [ 817.3,  457.6,  679.8],
               [ 881.9,  234.5,  902.8]])

        Multiplying an *array_like* variable:

        >>> tri_spd = tri_spd * [(1, 2, 3)] * 4
        >>> tri_spd.values
        array([[  1986.8,   7244.8,   1491.6],
               [  2783.6,   6987.2,   2778. ],
               [  3269.2,   3660.8,   8157.6],
               [  3527.6,   1876. ,  10833.6]])

        Multiplying a :class:`TriSpectralPowerDistribution` class variable:

        >>> data1 = {'x_bar': z_bar, 'y_bar': x_bar, 'z_bar': y_bar}
        >>> tri_spd1 = TriSpectralPowerDistribution('Observer', data1, mapping)
        >>> tri_spd = tri_spd * tri_spd1
        >>> tri_spd.values
        array([[  24695.924,  359849.216,  135079.296],
               [  64440.34 ,  486239.248,  242630.52 ],
               [ 222240.216,  299197.184,  373291.776],
               [ 318471.728,  165444.44 ,  254047.92 ]])
        """

        return self._arithmetical_operation(x, operator.mul)

    def __imul__(self, x):
        """
        Implements support for in-place tri-spectral power distribution
        multiplication.

        Usage is similar to the regular *multiplication* operation but make use
        of the *augmented assignement* operator such as: `tri_spd *= 10`
        instead of `tri_spd * 10`.

        Parameters
        ----------
        x : numeric or array_like or TriSpectralPowerDistribution
            Variable to in-place multiply by.

        Returns
        -------
        TriSpectralPowerDistribution
            Variable in-place multiplied tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.__mul__

        Notes
        -----
        -   Reimplements the :meth:`object.__imul__` method.
        """

        return self._arithmetical_operation(x, operator.mul, True)

    def __div__(self, x):
        """
        Implements support for tri-spectral power distribution division.

        Parameters
        ----------
        x : numeric or array_like or TriSpectralPowerDistribution
            Variable to divide by.

        Returns
        -------
        TriSpectralPowerDistribution
            Variable divided tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.__idiv__

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
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> tri_spd = tri_spd / 10
        >>> tri_spd.values
        array([[ 4.967,  9.056,  1.243],
               [ 6.959,  8.734,  2.315],
               [ 8.173,  4.576,  6.798],
               [ 8.819,  2.345,  9.028]])

        Dividing an *array_like* variable:

        >>> tri_spd = tri_spd / [(1, 2, 3)] * 4
        >>> tri_spd.values  # doctest: +ELLIPSIS
        array([[ 19.868     ,  18.112     ,   1.6573333...],
               [ 27.836     ,  17.468     ,   3.0866666...],
               [ 32.692     ,   9.152     ,   9.064    ...],
               [ 35.276     ,   4.69      ,  12.0373333...]])

        Dividing a :class:`TriSpectralPowerDistribution` class variable:

        >>> data1 = {'x_bar': z_bar, 'y_bar': x_bar, 'z_bar': y_bar}
        >>> tri_spd1 = TriSpectralPowerDistribution('Observer', data1, mapping)
        >>> tri_spd = tri_spd / tri_spd1
        >>> tri_spd.values  # doctest: +ELLIPSIS
        array([[ 1.5983909...,  0.3646466...,  0.0183009...],
               [ 1.2024190...,  0.2510130...,  0.0353408...],
               [ 0.4809061...,  0.1119784...,  0.1980769...],
               [ 0.3907399...,  0.0531806...,  0.5133191...]])
        """

        return self._arithmetical_operation(x, operator.truediv)

    def __idiv__(self, x):
        """
        Implements support for in-place tri-spectral power distribution
        division.

        Usage is similar to the regular *division* operation but make use of
        the *augmented assignement* operator such as: `tri_spd /= 10` instead
        of `tri_spd / 10`.

        Parameters
        ----------
        x : numeric or array_like or TriSpectralPowerDistribution
            Variable to in-place divide by.

        Returns
        -------
        TriSpectralPowerDistribution
            Variable in-place divided tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.__div__

        Notes
        -----
        -   Reimplements the :meth:`object.__idiv_` method.
        """

        return self._arithmetical_operation(x, operator.truediv, True)

    # Python 3 compatibility.
    __itruediv__ = __idiv__
    __truediv__ = __div__

    def __pow__(self, x):
        """
        Implements support for tri-spectral power distribution exponentiation.

        Parameters
        ----------
        x : numeric or array_like or TriSpectralPowerDistribution
            Variable to exponentiate by.

        Returns
        -------
        TriSpectralPowerDistribution
            TriSpectral power distribution raised by power of x.

        See Also
        --------
        TriSpectralPowerDistribution.__ipow__,

        Notes
        -----
        -   Reimplements the :meth:`object.__pow__` method.

        Warning
        -------
        The power operation happens in place.

        Examples
        --------
        Exponentiation by a single *numeric* variable:

        >>> x_bar = {510: 1.67, 520: 1.59, 530: 1.73, 540: 1.19}
        >>> y_bar = {510: 1.56, 520: 1.34, 530: 1.76, 540: 1.45}
        >>> z_bar = {510: 1.43, 520: 1.15, 530: 1.98, 540: 1.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> tri_spd = tri_spd ** 1.1
        >>> tri_spd.values  # doctest: +ELLIPSIS
        array([[ 1.7578755...,  1.6309365...,  1.4820731...],
               [ 1.6654700...,  1.3797972...,  1.1661854...],
               [ 1.8274719...,  1.8623612...,  2.1199797...],
               [ 1.2108815...,  1.5048901...,  1.3119913...]])

        Exponentiation by an *array_like* variable:

        >>> tri_spd = tri_spd ** ([(1, 2, 3)] * 4)
        >>> tri_spd.values  # doctest: +ELLIPSIS
        array([[ 1.7578755...,  2.6599539...,  3.2554342...],
               [ 1.6654700...,  1.9038404...,  1.5859988...],
               [ 1.8274719...,  3.4683895...,  9.5278547...],
               [ 1.2108815...,  2.2646943...,  2.2583585...]])

        Exponentiation by a :class:`TriSpectralPowerDistribution`
        class variable:

        >>> data1 = {'x_bar': z_bar, 'y_bar': x_bar, 'z_bar': y_bar}
        >>> tri_spd1 = TriSpectralPowerDistribution('Observer', data1, mapping)
        >>> tri_spd = tri_spd ** tri_spd1
        >>> tri_spd.values  # doctest: +ELLIPSIS
        array([[  2.2404384...,   5.1231818...,   6.3047797...],
               [  1.7979075...,   2.7836369...,   1.8552645...],
               [  3.2996236...,   8.5984706...,  52.8483490...],
               [  1.2775271...,   2.6452177...,   3.2583647...]])
        """

        return self._arithmetical_operation(x, operator.pow)

    def __ipow__(self, x):
        """
        Implements support for in-place tri-spectral power distribution
        exponentiation.

        Usage is similar to the regular *exponentiation* operation but make use
        of the *augmented assignement* operator such as: `tri_spd **= 10`
        instead of `tri_spd ** 10`.

        Parameters
        ----------
        x : numeric or array_like or TriSpectralPowerDistribution
            Variable to in-place exponentiate by.

        Returns
        -------
        TriSpectralPowerDistribution
            Variable in-place exponentiated tri-spectral power distribution.

        See Also
        --------
        TriSpectralPowerDistribution.__pow__

        Notes
        -----
        -   Reimplements the :meth:`object.__ipow__` method.
        """

        return self._arithmetical_operation(x, operator.pow, True)

    def _arithmetical_operation(self, x, operation, in_place=False):
        """
        Performs given arithmetical operation on :math:`x` variable, the
        operation can be either performed on a tri-spectral power distribution
        clone or in-place.

        Parameters
        ----------
        x : numeric or ndarray or TriSpectralPowerDistribution
            Operand.
        operation : object
            Operation to perform.
        in_place : bool, optional
            Operation happens in place.

        Returns
        -------
        TriSpectralPowerDistribution
            Tri-spectral power distribution.
        """

        if issubclass(type(x), TriSpectralPowerDistribution):
            x = x.values
        elif is_iterable(x):
            x = np.atleast_1d(x)

        data = {}
        values = operation(self.values, x)
        for i, axis in enumerate(('x', 'y', 'z')):
            data[self._mapping[axis]] = SpectralMapping(
                zip(self.wavelengths, values[..., i]))

        if in_place:
            self.data = data
            return self
        else:
            clone = self.clone()
            clone.data = data
            return clone

    def get(self, wavelength, default=np.nan):
        """
        Returns the values for given wavelength :math:`\lambda`.

        Parameters
        ----------
        wavelength : numeric or array_like
            Wavelength :math:`\lambda` to retrieve the values.
        default : nan, numeric or array_like, optional
            Wavelength :math:`\lambda` default values.

        Returns
        -------
        numeric or array_like
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
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> tri_spd.get(510)
        array([ 49.67,  90.56,  12.43])
        >>> tri_spd.get(np.array([510, 520]))
        array([[ 49.67,  90.56,  12.43],
               [ 69.59,  87.34,  23.15]])
        >>> tri_spd.get(511)
        array([ nan,  nan,  nan])
        >>> tri_spd.get(np.array([510, 520]))
        array([[ 49.67,  90.56,  12.43],
               [ 69.59,  87.34,  23.15]])
        """

        wavelength = np.asarray(wavelength)

        default = np.resize(default, 3)
        value = tstack([self.x.get(wavelength, default[0]),
                        self.y.get(wavelength, default[1]),
                        self.z.get(wavelength, default[2])])

        value = np.reshape(value, wavelength.shape + (3,))

        return value

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
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> tri_spd.is_uniform()
        True

        Breaking the interval by introducing new wavelength :math:`\lambda`
        values:

        >>> tri_spd[511] = np.array([49.6700, 49.6700, 49.6700])
        >>> tri_spd.is_uniform()
        False
        """

        for i in self._mapping.keys():
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
        *CIE 15:2004* recommendation. [2]_ [3]_

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for extrapolation.
        method : unicode, optional
            **{'Constant', 'Linear'}**,
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

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> tri_spd.extrapolate(  # doctest: +ELLIPSIS
        ...     SpectralShape(400, 700)).shape
        SpectralShape(400..., 700..., 10...)
        >>> tri_spd[400]
        array([ 49.67,  90.56,  12.43])
        >>> tri_spd[700]
        array([ 88.19,  23.45,  90.28])
        """

        for i in self._mapping.keys():
            getattr(self, i).extrapolate(shape, method, left, right)

        return self

    def interpolate(self, shape=SpectralShape(), method=None):
        """
        Interpolates the tri-spectral power distribution following
        *CIE 167:2005* recommendations: the method developed by
        *Sprague (1880)* should be used for interpolating functions having a
        uniformly spaced independent variable and a *Cubic Spline* method for
        non-uniformly spaced independent variable. [4]_

        Parameters
        ----------
        shape : SpectralShape, optional
            Spectral shape used for interpolation.
        method : unicode, optional
            **{None, 'Cubic Spline', 'Linear', 'Pchip', 'Sprague'}**,
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

        Examples
        --------
        Uniform data is using *Sprague (1880)* interpolation by default:

        >>> x_bar = {
        ...     510: 49.67,
        ...     520: 69.59,
        ...     530: 81.73,
        ...     540: 88.19,
        ...     550: 89.76,
        ...     560: 90.28}
        >>> y_bar = {
        ...     510: 90.56,
        ...     520: 87.34,
        ...     530: 45.76,
        ...     540: 23.45,
        ...     550: 15.34,
        ...     560: 10.11}
        >>> z_bar = {
        ...     510: 12.43,
        ...     520: 23.15,
        ...     530: 67.98,
        ...     540: 90.28,
        ...     550: 91.61,
        ...     560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> print(tri_spd.interpolate(SpectralShape(interval=1)))
        TriSpectralPowerDistribution('Observer', (510.0, 560.0, 1.0))
        >>> tri_spd[515]  # doctest: +ELLIPSIS
        array([ 60.3033208...,  93.2716331...,  13.8605136...])

        Non uniform data is using *Cubic Spline* interpolation by default:

        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> tri_spd[511] = np.array([31.41, 95.27, 15.06])
        >>> print(tri_spd.interpolate(SpectralShape(interval=1)))
        TriSpectralPowerDistribution('Observer', (510.0, 560.0, 1.0))
        >>> tri_spd[515]  # doctest: +ELLIPSIS
        array([  21.4710405...,  100.6430015...,   18.8165196...])

        Enforcing *Linear* interpolation:

        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> print(tri_spd.interpolate(  # doctest: +ELLIPSIS
        ...     SpectralShape(interval=1), method='Linear'))
        TriSpectralPowerDistribution('Observer', (510.0, 560.0, 1.0))
        >>> tri_spd[515]  # doctest: +ELLIPSIS
        array([ 59.63...,  88.95...,  17.79...])

        Enforcing *Pchip* interpolation:

        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> print(tri_spd.interpolate(  # doctest: +ELLIPSIS
        ...     SpectralShape(interval=1), method='Pchip'))
        TriSpectralPowerDistribution('Observer', (510.0, 560.0, 1.0))
        >>> tri_spd[515]  # doctest: +ELLIPSIS
        array([ 60.7204982...,  89.6971406...,  15.6271845...])
        """

        for i in self._mapping.keys():
            getattr(self, i).interpolate(shape, method)

        return self

    def align(self,
              shape,
              interpolation_method=None,
              extrapolation_method='Constant',
              extrapolation_left=None,
              extrapolation_right=None):
        """
        Aligns the tri-spectral power distribution to given shape: Interpolates
        first then extrapolates to fit the given range.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for alignment.
        interpolation_method : unicode, optional
            **{None, 'Cubic Spline', 'Linear', 'Pchip', 'Sprague'}**,
            Enforce given interpolation method.
        extrapolation_method : unicode, optional
            **{'Constant', 'Linear'}**,
            Extrapolation method.
        extrapolation_left : numeric, optional
            Value to return for low extrapolation range.
        extrapolation_right : numeric, optional
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
        >>> x_bar = {
        ...     510: 49.67,
        ...     520: 69.59,
        ...     530: 81.73,
        ...     540: 88.19,
        ...     550: 89.76,
        ...     560: 90.28}
        >>> y_bar = {
        ...     510: 90.56,
        ...     520: 87.34,
        ...     530: 45.76,
        ...     540: 23.45,
        ...     550: 15.34,
        ...     560: 10.11}
        >>> z_bar = {
        ...     510: 12.43,
        ...     520: 23.15,
        ...     530: 67.98,
        ...     540: 90.28,
        ...     550: 91.61,
        ...     560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> print(tri_spd.align(SpectralShape(505, 565, 1)))
        TriSpectralPowerDistribution('Observer', (505.0, 565.0, 1.0))
        >>> # Doctests skip for Python 2.x compatibility.
        >>> tri_spd.wavelengths  # doctest: +SKIP
        array([505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517,
               518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530,
               531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543,
               544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556,
               557, 558, 559, 560, 561, 562, 563, 564, 565])
        >>> tri_spd.values  # doctest: +ELLIPSIS
        array([[ 49.67     ...,  90.56     ...,  12.43     ...],
               [ 49.67     ...,  90.56     ...,  12.43     ...],
               [ 49.67     ...,  90.56     ...,  12.43     ...],
               [ 49.67     ...,  90.56     ...,  12.43     ...],
               [ 49.67     ...,  90.56     ...,  12.43     ...],
               [ 49.67     ...,  90.56     ...,  12.43     ...],
               [ 51.8325938...,  91.2994928...,  12.5377184...],
               [ 53.9841952...,  91.9502387...,  12.7233193...],
               [ 56.1205452...,  92.5395463...,  12.9651679...],
               [ 58.2315395...,  93.0150037...,  13.3123777...],
               [ 60.3033208...,  93.2716331...,  13.8605136...],
               [ 62.3203719...,  93.1790455...,  14.7272944...],
               [ 64.2676077...,  92.6085951...,  16.0282961...],
               [ 66.1324679...,  91.4605335...,  17.8526544...],
               [ 67.9070097...,  89.6911649...,  20.2387677...],
               [ 69.59     ...,  87.34     ...,  23.15     ...],
               [ 71.1837378...,  84.4868033...,  26.5150469...],
               [ 72.6800056...,  81.0666018...,  30.3964852...],
               [ 74.0753483...,  77.0766254...,  34.7958422...],
               [ 75.3740343...,  72.6153870...,  39.6178858...],
               [ 76.5856008...,  67.8490714...,  44.7026805...],
               [ 77.7223995...,  62.9779261...,  49.8576432...],
               [ 78.7971418...,  58.2026503...,  54.8895997...],
               [ 79.8204447...,  53.6907852...,  59.6368406...],
               [ 80.798376 ...,  49.5431036...,  64.0011777...],
               [ 81.73     ...,  45.76     ...,  67.98     ...],
               [ 82.6093606...,  42.2678534...,  71.6460893...],
               [ 83.439232 ...,  39.10608  ...,  74.976976 ...],
               [ 84.2220071...,  36.3063728...,  77.9450589...],
               [ 84.956896 ...,  33.85464  ...,  80.552    ...],
               [ 85.6410156...,  31.7051171...,  82.8203515...],
               [ 86.27048  ...,  29.79448  ...,  84.785184 ...],
               [ 86.8414901...,  28.0559565...,  86.4857131...],
               [ 87.351424 ...,  26.43344  ...,  87.956928 ...],
               [ 87.7999266...,  24.8956009...,  89.2212178...],
               [ 88.19     ...,  23.45     ...,  90.28     ...],
               [ 88.5265036...,  22.1424091...,  91.1039133...],
               [ 88.8090803...,  20.9945234...,  91.6538035...],
               [ 89.0393279...,  20.0021787...,  91.9333499...],
               [ 89.2222817...,  19.1473370...,  91.9858818...],
               [ 89.3652954...,  18.4028179...,  91.8811002...],
               [ 89.4769231...,  17.7370306...,  91.7018000...],
               [ 89.5657996...,  17.1187058...,  91.5305910...],
               [ 89.6395227...,  16.5216272...,  91.4366204...],
               [ 89.7035339...,  15.9293635...,  91.4622944...],
               [ 89.76     ...,  15.34     ...,  91.61     ...],
               [ 89.8094041...,  14.7659177...,  91.8528616...],
               [ 89.8578890...,  14.2129190...,  92.2091737...],
               [ 89.9096307...,  13.6795969...,  92.6929664...],
               [ 89.9652970...,  13.1613510...,  93.2988377...],
               [ 90.0232498...,  12.6519811...,  94.0078786...],
               [ 90.0807467...,  12.1452800...,  94.7935995...],
               [ 90.1351435...,  11.6366269...,  95.6278555...],
               [ 90.1850956...,  11.1245805...,  96.4867724...],
               [ 90.2317606...,  10.6124724...,  97.3566724...],
               [ 90.28     ...,  10.11     ...,  98.24     ...],
               [ 90.28     ...,  10.11     ...,  98.24     ...],
               [ 90.28     ...,  10.11     ...,  98.24     ...],
               [ 90.28     ...,  10.11     ...,  98.24     ...],
               [ 90.28     ...,  10.11     ...,  98.24     ...],
               [ 90.28     ...,  10.11     ...,  98.24     ...]])
        """

        for i in self._mapping.keys():
            getattr(self, i).align(shape,
                                   interpolation_method,
                                   extrapolation_method,
                                   extrapolation_left,
                                   extrapolation_right)

        return self

    def trim_wavelengths(self, shape):
        """
        Trims the tri-spectral power distribution wavelengths to given shape.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for trimming.

        Returns
        -------
        TriSpectralPowerDistribution
            Trimmed tri-spectral power distribution.

        Examples
        --------
        >>> x_bar = {
        ...     510: 49.67,
        ...     520: 69.59,
        ...     530: 81.73,
        ...     540: 88.19,
        ...     550: 89.76,
        ...     560: 90.28}
        >>> y_bar = {
        ...     510: 90.56,
        ...     520: 87.34,
        ...     530: 45.76,
        ...     540: 23.45,
        ...     550: 15.34,
        ...     560: 10.11}
        >>> z_bar = {
        ...     510: 12.43,
        ...     520: 23.15,
        ...     530: 67.98,
        ...     540: 90.28,
        ...     550: 91.61,
        ...     560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> # Doctests skip for Python 2.x compatibility.
        >>> print(tri_spd.trim_wavelengths(  # doctest: +SKIP
        ...     SpectralShape(520, 550, 10)))
        TriSpectralPowerDistribution('Observer', (520.0, 550.0, 10.0))
        >>> tri_spd.wavelengths  # doctest: +SKIP
        array([ 520.,  530.,  540.,  550.])
        """

        for i in self._mapping.keys():
            getattr(self, i).trim_wavelengths(shape)

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
        >>> x_bar = {
        ...     510: 49.67,
        ...     520: 69.59,
        ...     530: 81.73,
        ...     540: 88.19,
        ...     550: 89.76,
        ...     560: 90.28}
        >>> y_bar = {
        ...     510: 90.56,
        ...     520: 87.34,
        ...     530: 45.76,
        ...     540: 23.45,
        ...     550: 15.34,
        ...     560: 10.11}
        >>> z_bar = {
        ...     510: 12.43,
        ...     520: 23.15,
        ...     530: 67.98,
        ...     540: 90.28,
        ...     550: 91.61,
        ...     560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> print(tri_spd.zeros(SpectralShape(505, 565, 1)))
        TriSpectralPowerDistribution('Observer', (505.0, 565.0, 1.0))
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

        for i in self._mapping.keys():
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
        >>> x_bar = {
        ...     510: 49.67,
        ...     520: 69.59,
        ...     530: 81.73,
        ...     540: 88.19,
        ...     550: 89.76,
        ...     560: 90.28}
        >>> y_bar = {
        ...     510: 90.56,
        ...     520: 87.34,
        ...     530: 45.76,
        ...     540: 23.45,
        ...     550: 15.34,
        ...     560: 10.11}
        >>> z_bar = {
        ...     510: 12.43,
        ...     520: 23.15,
        ...     530: 67.98,
        ...     540: 90.28,
        ...     550: 91.61,
        ...     560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> print(tri_spd.normalise())  # doctest: +ELLIPSIS
        TriSpectralPowerDistribution('Observer', (510..., 560..., 10...))
        >>> tri_spd.values  # doctest: +ELLIPSIS
        array([[ 0.5055985...,  0.9218241...,  0.1265268...],
               [ 0.7083672...,  0.8890472...,  0.2356473...],
               [ 0.8319421...,  0.4657980...,  0.6919788...],
               [ 0.8976995...,  0.2387011...,  0.9189739...],
               [ 0.9136807...,  0.1561482...,  0.9325122...],
               [ 0.9189739...,  0.1029112...,  1.       ...]])
        """

        maximum = np.max(self.values)
        for i in self._mapping.keys():
            operator.imul(getattr(self, i), (1 / maximum) * factor)

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
        >>> x_bar = {
        ...     510: 49.67,
        ...     520: 69.59,
        ...     530: 81.73,
        ...     540: 88.19,
        ...     550: 89.76,
        ...     560: 90.28}
        >>> y_bar = {
        ...     510: 90.56,
        ...     520: 87.34,
        ...     530: 45.76,
        ...     540: 23.45,
        ...     550: 15.34,
        ...     560: 10.11}
        >>> z_bar = {
        ...     510: 12.43,
        ...     520: 23.15,
        ...     530: 67.98,
        ...     540: 90.28,
        ...     550: 91.61,
        ...     560: 98.24}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> mapping = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
        >>> tri_spd = TriSpectralPowerDistribution('Observer', data, mapping)
        >>> print(tri_spd)  # doctest: +ELLIPSIS
        TriSpectralPowerDistribution('Observer', (510..., 560..., 10...))
        >>> tri_spd_clone = tri_spd.clone()
        >>> print(tri_spd_clone)  # doctest: +ELLIPSIS
        TriSpectralPowerDistribution('Observer (...)', (510..., 560..., 10...))
        """

        data = {self.mapping['x']: self.x.data.data,
                self.mapping['y']: self.y.data.data,
                self.mapping['z']: self.z.data.data}
        mapping = self.mapping.copy()
        labels = self.labels.copy()
        clone = TriSpectralPowerDistribution(
            self.name, data, mapping, self.title, labels)

        clone.name = '{0} ({1})'.format(clone.name, id(clone))

        if self._title is None:
            clone.title = self._name

        return clone


DEFAULT_SPECTRAL_SHAPE = SpectralShape(360, 780, 1)
"""
Default spectral shape using *ASTM E308–15* practise shape.

DEFAULT_SPECTRAL_SHAPE : SpectralShape
"""


def constant_spd(k, shape=DEFAULT_SPECTRAL_SHAPE):
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
    >>> spd = constant_spd(100)
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[400]
    array(100.0)
    """

    wavelengths = shape.range()
    values = np.full(len(wavelengths), k, np.float_)

    name = '{0} Constant'.format(k)
    return SpectralPowerDistribution(
        name, SpectralMapping(zip(wavelengths, values)))


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
    >>> spd = zeros_spd()
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[400]
    array(0.0)
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
    >>> spd = ones_spd()
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[400]
    array(1.0)
    """

    return constant_spd(1, shape)
