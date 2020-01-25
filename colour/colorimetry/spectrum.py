# -*- coding: utf-8 -*-
"""
Spectrum
========

Defines the classes and objects handling spectral data computations:

-   :class:`colour.DEFAULT_SPECTRAL_SHAPE`
-   :class:`colour.SpectralShape`
-   :class:`colour.SpectralDistribution`
-   :class:`colour.MultiSpectralDistributions`
-   :func:`colour.colorimetry.sds_and_multi_sds_to_sds`
-   :func:`colour.colorimetry.sds_and_multi_sds_to_multi_sds`

See Also
--------
`Spectrum Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/spectrum.ipynb>`_

References
----------
-   :cite:`CIETC1-382005e` : CIE TC 1-38. (2005). 9. INTERPOLATION. In
    CIE 167:2005 Recommended Practice for Tabulating Spectral Data for Use in
    Colour Computations (pp. 14-19). ISBN:978-3-901-90641-1
-   :cite:`CIETC1-382005g` : CIE TC 1-38. (2005). EXTRAPOLATION. In
    CIE 167:2005 Recommended Practice for Tabulating Spectral Data for Use in
    Colour Computations (pp. 19-20). ISBN:978-3-901-90641-1
-   :cite:`CIETC1-482004l` : CIE TC 1-48. (2004). Extrapolation. In
    CIE 015:2004 Colorimetry, 3rd Edition (p. 24). ISBN:978-3-901-90633-6
"""

from __future__ import division, unicode_literals

import numpy as np
from six.moves import zip

from colour.algebra import (Extrapolator, CubicSplineInterpolator,
                            SpragueInterpolator)
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.continuous import Signal, MultiSignals
from colour.utilities import (as_float, as_int, first_item, is_iterable,
                              is_numeric, is_string, is_uniform, interval,
                              runtime_warning, tstack)
from colour.utilities.deprecation import ObjectRemoved, ObjectRenamed

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'SpectralShape', 'DEFAULT_SPECTRAL_SHAPE', 'SpectralDistribution',
    'MultiSpectralDistributions', 'sds_and_multi_sds_to_sds',
    'sds_and_multi_sds_to_multi_sds'
]


class SpectralShape(object):
    """
    Defines the base object for spectral distribution shape.

    Parameters
    ----------
    start : numeric, optional
        Wavelength :math:`\\lambda_{i}` range start in nm.
    end : numeric, optional
        Wavelength :math:`\\lambda_{i}` range end in nm.
    interval : numeric, optional
        Wavelength :math:`\\lambda_{i}` range interval.

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
    __hash__
    __iter__
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

    def __init__(self, start=None, end=None, interval=None):
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
        Getter and setter property for the spectral shape start.

        Parameters
        ----------
        value : numeric
            Value to set the spectral shape start with.

        Returns
        -------
        numeric
            Spectral shape start.
        """

        return self._start

    @start.setter
    def start(self, value):
        """
        Setter for the **self.start** property.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" is not a "numeric"!'.format(
                    'start', value))

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
        Getter and setter property for the spectral shape end.

        Parameters
        ----------
        value : numeric
            Value to set the spectral shape end with.

        Returns
        -------
        numeric
            Spectral shape end.
        """

        return self._end

    @end.setter
    def end(self, value):
        """
        Setter for the **self.end** property.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" is not a "numeric"!'.format(
                    'end', value))

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
        Getter and setter property for the spectral shape interval.

        Parameters
        ----------
        value : numeric
            Value to set the spectral shape interval with.

        Returns
        -------
        numeric
            Spectral shape interval.
        """

        return self._interval

    @interval.setter
    def interval(self, value):
        """
        Setter for the **self.interval** property.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" is not a "numeric"!'.format(
                    'interval', value))

        # Invalidating the *range* cache.
        if value != self._interval:
            self._range = None

        self._interval = value

    @property
    def boundaries(self):
        """
        Getter and setter property for the spectral shape boundaries.

        Parameters
        ----------
        value : array_like
            Value to set the spectral shape boundaries with.

        Returns
        -------
        tuple
            Spectral shape boundaries.
        """

        return self._start, self._end

    @boundaries.setter
    def boundaries(self, value):
        """
        Setter for the **self.boundaries** property.
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
        Returns a formatted string representation of the spectral shape.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return '({0}, {1}, {2})'.format(self._start, self._end, self._interval)

    def __repr__(self):
        """
        Returns an evaluable string representation of the spectral shape.

        Returns
        -------
        unicode
            Evaluable string representation.
        """

        return 'SpectralShape({0}, {1}, {2})'.format(self._start, self._end,
                                                     self._interval)

    def __hash__(self):
        """
        Returns the spectral shape hash.

        Returns
        -------
        int
            Object hash.
        """

        return hash(repr(self))

    def __iter__(self):
        """
        Returns a generator for the spectral shape data.

        Returns
        -------
        generator
            Spectral shape data generator.

        Examples
        --------
        >>> shape = SpectralShape(0, 10, 1)
        >>> for wavelength in shape:
        ...     print(wavelength)
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
        :math:`\\lambda`.

        Parameters
        ----------
        wavelength : numeric or array_like
            Wavelength :math:`\\lambda`.

        Returns
        -------
        bool
            Is wavelength :math:`\\lambda` contained in the spectral shape.

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

        tolerance = 10 ** -np.finfo(DEFAULT_FLOAT_DTYPE).precision

        return np.all(
            np.in1d(
                np.around(wavelength / tolerance).astype(np.int64),
                np.around(self.range() / tolerance).astype(np.int64)))

    def __len__(self):
        """
        Returns the spectral shape wavelength :math:`\\lambda_n` count.

        Returns
        -------
        int
            Spectral shape wavelength :math:`\\lambda_n` count.

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

        Examples
        --------
        >>> SpectralShape(0, 10, 0.1) != SpectralShape(0, 10, 0.1)
        False
        >>> SpectralShape(0, 10, 0.1) != SpectralShape(0, 10, 1)
        True
        """

        return not (self == shape)

    def range(self, dtype=DEFAULT_FLOAT_DTYPE):
        """
        Returns an iterable range for the spectral shape.

        Parameters
        ----------
        dtype : type
            Data type used to generate the range.

        Returns
        -------
        ndarray
            Iterable range for the spectral distribution shape

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
            samples = as_int(
                round((self._interval + self._end - self._start) /
                      self._interval))
            range_, current_interval = np.linspace(
                self._start, self._end, samples, retstep=True, dtype=dtype)

            self._range = range_

            if current_interval != self._interval:
                self._interval = current_interval
                runtime_warning(
                    ('"{0}" shape could not be honoured, using '
                     '"{1}"!').format((self._start, self._end, self._interval),
                                      self))

        return self._range


DEFAULT_SPECTRAL_SHAPE = SpectralShape(360, 780, 1)
"""
Default spectral shape according to *ASTM E308-15* practise shape.

DEFAULT_SPECTRAL_SHAPE : SpectralShape
"""


class SpectralDistribution(Signal):
    """
    Defines the spectral distribution: the base object for spectral
    computations.

    The spectral distribution will be initialised according to *CIE 15:2004*
    recommendation: the method developed by *Sprague (1880)* will be used for
    interpolating functions having a uniformly spaced independent variable and
    the *Cubic Spline* method for non-uniformly spaced independent variable.
    Extrapolation is performed according to *CIE 167:2005* recommendation.

    Parameters
    ----------
    data : Series or Signal, SpectralDistribution or array_like or \
dict_like, optional
        Data to be stored in the spectral distribution.
    domain : array_like, optional
        Values to initialise the
        :attr:`colour.SpectralDistribution.wavelength` attribute with.
        If both ``data`` and ``domain`` arguments are defined, the latter will
        be used to initialise the
        :attr:`colour.SpectralDistribution.wavelength` attribute.

    Other Parameters
    ----------------
    name : unicode, optional
        Spectral distribution name.
    interpolator : object, optional
        Interpolator class type to use as interpolating function.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating the extrapolating function.
    strict_name : unicode, optional
        Spectral distribution name for figures, default to
        :attr:`colour.SpectralDistribution.name` attribute value.

    Attributes
    ----------
    strict_name
    wavelengths
    values
    shape

    Methods
    -------
    __init__
    extrapolate
    interpolate
    align
    trim
    normalise

    References
    ----------
    :cite:`CIETC1-382005e`, :cite:`CIETC1-382005g`, :cite:`CIETC1-482004l`

    Examples
    --------
    Instantiating a spectral distribution with a uniformly spaced independent
    variable:

    >>> from colour.utilities import numpy_print_options
    >>> data = {
    ...     500: 0.0651,
    ...     520: 0.0705,
    ...     540: 0.0772,
    ...     560: 0.0870,
    ...     580: 0.1128,
    ...     600: 0.1360
    ... }
    >>> with numpy_print_options(suppress=True):
    ...     SpectralDistribution(data)  # doctest: +ELLIPSIS
    SpectralDistribution([[ 500.    ,    0.0651],
                          [ 520.    ,    0.0705],
                          [ 540.    ,    0.0772],
                          [ 560.    ,    0.087 ],
                          [ 580.    ,    0.1128],
                          [ 600.    ,    0.136 ]],
                         interpolator=SpragueInterpolator,
                         interpolator_args={},
                         extrapolator=Extrapolator,
                         extrapolator_args={...})

    Instantiating a spectral distribution with a non-uniformly spaced
    independent variable:

    >>> data[510] = 0.31416
    >>> with numpy_print_options(suppress=True):
    ...     SpectralDistribution(data)  # doctest: +ELLIPSIS
    SpectralDistribution([[ 500.     ,    0.0651 ],
                          [ 510.     ,    0.31416],
                          [ 520.     ,    0.0705 ],
                          [ 540.     ,    0.0772 ],
                          [ 560.     ,    0.087  ],
                          [ 580.     ,    0.1128 ],
                          [ 600.     ,    0.136  ]],
                         interpolator=CubicSplineInterpolator,
                         interpolator_args={},
                         extrapolator=Extrapolator,
                         extrapolator_args={...})
    """

    def __init__(self, data=None, domain=None, **kwargs):
        domain, range_ = self.signal_unpack_data(data, domain)

        uniform = is_uniform(domain) if domain is not None else True

        # Initialising with *CIE 15:2004* and *CIE 167:2005* recommendations
        # defaults.
        kwargs['interpolator'] = kwargs.get(
            'interpolator', SpragueInterpolator
            if uniform else CubicSplineInterpolator)
        kwargs['interpolator_args'] = kwargs.get('interpolator_args', {})

        kwargs['extrapolator'] = kwargs.get('extrapolator', Extrapolator)
        kwargs['extrapolator_args'] = kwargs.get('extrapolator_args', {
            'method': 'Constant',
            'left': None,
            'right': None
        })

        super(SpectralDistribution, self).__init__(range_, domain, **kwargs)

        self._strict_name = None
        self.strict_name = kwargs.get('strict_name')

    @property
    def strict_name(self):
        """
        Getter and setter property for the spectral distribution strict name.

        Parameters
        ----------
        value : unicode
            Value to set the spectral distribution strict name with.

        Returns
        -------
        unicode
            Spectral distribution strict name.
        """

        if self._strict_name is not None:
            return self._strict_name
        else:
            return self._name

    @strict_name.setter
    def strict_name(self, value):
        """
        Setter for **self.strict_name** property.
        """

        if value is not None:
            assert is_string(value), (
                ('"{0}" attribute: "{1}" is not a "string" like object!'
                 ).format('strict_name', value))
            self._strict_name = value

    @property
    def wavelengths(self):
        """
        Getter and setter property for the spectral distribution wavelengths
        :math:`\\lambda_n`.

        Parameters
        ----------
        value : array_like
            Value to set the spectral distribution wavelengths
            :math:`\\lambda_n` with.

        Returns
        -------
        ndarray
            Spectral distribution wavelengths :math:`\\lambda_n`.
        """

        return self.domain

    @wavelengths.setter
    def wavelengths(self, value):
        """
        Setter for the **self.wavelengths** property.
        """

        self.domain = value

    @property
    def values(self):
        """
        Getter and setter property for the spectral distribution values.

        Parameters
        ----------
        value : array_like
            Value to set the spectral distribution wavelengths values
            with.

        Returns
        -------
        ndarray
            Spectral distribution values.
        """

        return self.range

    @values.setter
    def values(self, value):
        """
        Setter for the **self.values** property.
        """

        self.range = value

    @property
    def shape(self):
        """
        Getter and setter property for the spectral distribution shape.

        Returns
        -------
        SpectralShape
            Spectral distribution shape.

        Notes
        -----
        -   A spectral distribution with a non-uniformly spaced independent
            variable have multiple intervals, in that case
            :attr:`colour.SpectralDistribution.shape` attribute returns
            the *minimum* interval size.

        Warning
        -------
        :attr:`colour.SpectralDistribution.shape` attribute is read only.

        Examples
        --------
        Shape of a spectral distribution with a uniformly spaced independent
        variable:

        >>> data = {
        ...     500: 0.0651,
        ...     520: 0.0705,
        ...     540: 0.0772,
        ...     560: 0.0870,
        ...     580: 0.1128,
        ...     600: 0.1360
        ... }
        >>> SpectralDistribution(data).shape
        SpectralShape(500.0, 600.0, 20.0)

        Shape of a spectral distribution with a non-uniformly spaced
        independent variable:

        >>> data[510] = 0.31416
        >>> SpectralDistribution(data).shape
        SpectralShape(500.0, 600.0, 10.0)
        """

        wavelengths_interval = interval(self.wavelengths)
        if wavelengths_interval.size != 1:
            runtime_warning(('"{0}" spectral distribution is not uniform, '
                             'using minimum interval!'.format(self.name)))

        return SpectralShape(
            min(self.wavelengths), max(self.wavelengths),
            as_float(min(wavelengths_interval)))

    def extrapolate(self, shape, extrapolator=None, extrapolator_args=None):
        """
        Extrapolates the spectral distribution in-place according to
        *CIE 15:2004* and *CIE 167:2005* recommendations or given extrapolation
        arguments.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for extrapolation.
        extrapolator : object, optional
            Extrapolator class type to use as extrapolating function.
        extrapolator_args : dict_like, optional
            Arguments to use when instantiating the extrapolating function.

        Returns
        -------
        SpectralDistribution
            Extrapolated spectral distribution.

        References
        ----------
        :cite:`CIETC1-382005g`, :cite:`CIETC1-482004l`

        Examples
        --------
        >>> from colour.utilities import numpy_print_options
        >>> data = {
        ...     500: 0.0651,
        ...     520: 0.0705,
        ...     540: 0.0772,
        ...     560: 0.0870,
        ...     580: 0.1128,
        ...     600: 0.1360
        ... }
        >>> sd = SpectralDistribution(data)
        >>> sd.extrapolate(SpectralShape(400, 700)).shape
        SpectralShape(400.0, 700.0, 20.0)
        >>> with numpy_print_options(suppress=True):
        ...     print(sd)
        [[ 400.        0.0651]
         [ 420.        0.0651]
         [ 440.        0.0651]
         [ 460.        0.0651]
         [ 480.        0.0651]
         [ 500.        0.0651]
         [ 520.        0.0705]
         [ 540.        0.0772]
         [ 560.        0.087 ]
         [ 580.        0.1128]
         [ 600.        0.136 ]
         [ 620.        0.136 ]
         [ 640.        0.136 ]
         [ 660.        0.136 ]
         [ 680.        0.136 ]
         [ 700.        0.136 ]]
        """

        self_shape = self.shape
        wavelengths = np.hstack([
            np.arange(shape.start, self_shape.start, self_shape.interval),
            np.arange(self_shape.end + self_shape.interval,
                      shape.end + self_shape.interval, self_shape.interval)
        ])

        if extrapolator is None:
            extrapolator = Extrapolator

        if extrapolator_args is None:
            extrapolator_args = {
                'method': 'Constant',
                'left': None,
                'right': None
            }

        self_extrapolator = self.extrapolator
        self_extrapolator_args = self.extrapolator_args

        self.extrapolator = extrapolator
        self.extrapolator_args = extrapolator_args

        # The following self-assignment is written as intended and triggers the
        # extrapolation.
        self[wavelengths] = self[wavelengths]

        self.extrapolator = self_extrapolator
        self.extrapolator_args = self_extrapolator_args

        return self

    def interpolate(self, shape, interpolator=None, interpolator_args=None):
        """
        Interpolates the spectral distribution in-place according to
        *CIE 167:2005* recommendation or given interpolation arguments.

        Parameters
        ----------
        shape : SpectralShape, optional
            Spectral shape used for interpolation.
        interpolator : object, optional
            Interpolator class type to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when instantiating the interpolating function.

        Returns
        -------
        SpectralDistribution
            Interpolated spectral distribution.

        Notes
        -----
        -   Interpolation will be performed over boundaries range, if you need
            to extend the range of the spectral distribution use the
            :meth:`colour.SpectralDistribution.extrapolate` or
            :meth:`colour.SpectralDistribution.align` methods.

        Warning
        -------
        -   *Cubic Spline* interpolator requires at least 3 wavelengths
            :math:`\\lambda_n` for interpolation.
        -   *Sprague (1880)* interpolator requires at least 6 wavelengths
            :math:`\\lambda_n` for interpolation.

        References
        ----------
        :cite:`CIETC1-382005e`

        Examples
        --------
        Spectral distribution with a uniformly spaced independent variable uses
        *Sprague (1880)* interpolation:

        >>> from colour.utilities import numpy_print_options
        >>> data = {
        ...     500: 0.0651,
        ...     520: 0.0705,
        ...     540: 0.0772,
        ...     560: 0.0870,
        ...     580: 0.1128,
        ...     600: 0.1360
        ... }
        >>> sd = SpectralDistribution(data)
        >>> with numpy_print_options(suppress=True):
        ...     print(sd.interpolate(SpectralShape(interval=1)))
        ... # doctest: +ELLIPSIS
        [[ 500.            0.0651   ...]
         [ 501.            0.0653522...]
         [ 502.            0.0656105...]
         [ 503.            0.0658715...]
         [ 504.            0.0661328...]
         [ 505.            0.0663929...]
         [ 506.            0.0666509...]
         [ 507.            0.0669069...]
         [ 508.            0.0671613...]
         [ 509.            0.0674150...]
         [ 510.            0.0676692...]
         [ 511.            0.0679253...]
         [ 512.            0.0681848...]
         [ 513.            0.0684491...]
         [ 514.            0.0687197...]
         [ 515.            0.0689975...]
         [ 516.            0.0692832...]
         [ 517.            0.0695771...]
         [ 518.            0.0698787...]
         [ 519.            0.0701870...]
         [ 520.            0.0705   ...]
         [ 521.            0.0708155...]
         [ 522.            0.0711336...]
         [ 523.            0.0714547...]
         [ 524.            0.0717789...]
         [ 525.            0.0721063...]
         [ 526.            0.0724367...]
         [ 527.            0.0727698...]
         [ 528.            0.0731051...]
         [ 529.            0.0734423...]
         [ 530.            0.0737808...]
         [ 531.            0.0741203...]
         [ 532.            0.0744603...]
         [ 533.            0.0748006...]
         [ 534.            0.0751409...]
         [ 535.            0.0754813...]
         [ 536.            0.0758220...]
         [ 537.            0.0761633...]
         [ 538.            0.0765060...]
         [ 539.            0.0768511...]
         [ 540.            0.0772   ...]
         [ 541.            0.0775527...]
         [ 542.            0.0779042...]
         [ 543.            0.0782507...]
         [ 544.            0.0785908...]
         [ 545.            0.0789255...]
         [ 546.            0.0792576...]
         [ 547.            0.0795917...]
         [ 548.            0.0799334...]
         [ 549.            0.0802895...]
         [ 550.            0.0806671...]
         [ 551.            0.0810740...]
         [ 552.            0.0815176...]
         [ 553.            0.0820049...]
         [ 554.            0.0825423...]
         [ 555.            0.0831351...]
         [ 556.            0.0837873...]
         [ 557.            0.0845010...]
         [ 558.            0.0852763...]
         [ 559.            0.0861110...]
         [ 560.            0.087    ...]
         [ 561.            0.0879383...]
         [ 562.            0.0889300...]
         [ 563.            0.0899793...]
         [ 564.            0.0910876...]
         [ 565.            0.0922541...]
         [ 566.            0.0934760...]
         [ 567.            0.0947487...]
         [ 568.            0.0960663...]
         [ 569.            0.0974220...]
         [ 570.            0.0988081...]
         [ 571.            0.1002166...]
         [ 572.            0.1016394...]
         [ 573.            0.1030687...]
         [ 574.            0.1044972...]
         [ 575.            0.1059186...]
         [ 576.            0.1073277...]
         [ 577.            0.1087210...]
         [ 578.            0.1100968...]
         [ 579.            0.1114554...]
         [ 580.            0.1128   ...]
         [ 581.            0.1141333...]
         [ 582.            0.1154495...]
         [ 583.            0.1167424...]
         [ 584.            0.1180082...]
         [ 585.            0.1192452...]
         [ 586.            0.1204536...]
         [ 587.            0.1216348...]
         [ 588.            0.1227915...]
         [ 589.            0.1239274...]
         [ 590.            0.1250465...]
         [ 591.            0.1261531...]
         [ 592.            0.1272517...]
         [ 593.            0.1283460...]
         [ 594.            0.1294393...]
         [ 595.            0.1305340...]
         [ 596.            0.1316310...]
         [ 597.            0.1327297...]
         [ 598.            0.1338277...]
         [ 599.            0.1349201...]
         [ 600.            0.136    ...]]

        Spectral distribution with a no-uniformly spaced independent variable
        uses *Cubic Spline* interpolation:

        >>> sd = SpectralDistribution(data)
        >>> sd[510] = np.pi / 10
        >>> with numpy_print_options(suppress=True):
        ...     print(sd.interpolate(SpectralShape(interval=1)))
        ... # doctest: +ELLIPSIS
        [[ 500.            0.0651   ...]
         [ 501.            0.1365202...]
         [ 502.            0.1953263...]
         [ 503.            0.2423724...]
         [ 504.            0.2785126...]
         [ 505.            0.3046010...]
         [ 506.            0.3214916...]
         [ 507.            0.3300387...]
         [ 508.            0.3310962...]
         [ 509.            0.3255184...]
         [ 510.            0.3141592...]
         [ 511.            0.2978729...]
         [ 512.            0.2775135...]
         [ 513.            0.2539351...]
         [ 514.            0.2279918...]
         [ 515.            0.2005378...]
         [ 516.            0.1724271...]
         [ 517.            0.1445139...]
         [ 518.            0.1176522...]
         [ 519.            0.0926962...]
         [ 520.            0.0705   ...]
         [ 521.            0.0517370...]
         [ 522.            0.0363589...]
         [ 523.            0.0241365...]
         [ 524.            0.0148407...]
         [ 525.            0.0082424...]
         [ 526.            0.0041126...]
         [ 527.            0.0022222...]
         [ 528.            0.0023421...]
         [ 529.            0.0042433...]
         [ 530.            0.0076966...]
         [ 531.            0.0124729...]
         [ 532.            0.0183432...]
         [ 533.            0.0250785...]
         [ 534.            0.0324496...]
         [ 535.            0.0402274...]
         [ 536.            0.0481829...]
         [ 537.            0.0560870...]
         [ 538.            0.0637106...]
         [ 539.            0.0708246...]
         [ 540.            0.0772   ...]
         [ 541.            0.0826564...]
         [ 542.            0.0872086...]
         [ 543.            0.0909203...]
         [ 544.            0.0938549...]
         [ 545.            0.0960760...]
         [ 546.            0.0976472...]
         [ 547.            0.0986321...]
         [ 548.            0.0990942...]
         [ 549.            0.0990971...]
         [ 550.            0.0987043...]
         [ 551.            0.0979794...]
         [ 552.            0.0969861...]
         [ 553.            0.0957877...]
         [ 554.            0.0944480...]
         [ 555.            0.0930304...]
         [ 556.            0.0915986...]
         [ 557.            0.0902161...]
         [ 558.            0.0889464...]
         [ 559.            0.0878532...]
         [ 560.            0.087    ...]
         [ 561.            0.0864371...]
         [ 562.            0.0861623...]
         [ 563.            0.0861600...]
         [ 564.            0.0864148...]
         [ 565.            0.0869112...]
         [ 566.            0.0876336...]
         [ 567.            0.0885665...]
         [ 568.            0.0896945...]
         [ 569.            0.0910020...]
         [ 570.            0.0924735...]
         [ 571.            0.0940936...]
         [ 572.            0.0958467...]
         [ 573.            0.0977173...]
         [ 574.            0.0996899...]
         [ 575.            0.1017491...]
         [ 576.            0.1038792...]
         [ 577.            0.1060649...]
         [ 578.            0.1082906...]
         [ 579.            0.1105408...]
         [ 580.            0.1128   ...]
         [ 581.            0.1150526...]
         [ 582.            0.1172833...]
         [ 583.            0.1194765...]
         [ 584.            0.1216167...]
         [ 585.            0.1236884...]
         [ 586.            0.1256760...]
         [ 587.            0.1275641...]
         [ 588.            0.1293373...]
         [ 589.            0.1309798...]
         [ 590.            0.1324764...]
         [ 591.            0.1338114...]
         [ 592.            0.1349694...]
         [ 593.            0.1359349...]
         [ 594.            0.1366923...]
         [ 595.            0.1372262...]
         [ 596.            0.1375211...]
         [ 597.            0.1375614...]
         [ 598.            0.1373316...]
         [ 599.            0.1368163...]
         [ 600.            0.136    ...]]
        """

        self_shape = self.shape
        s_e_i = zip((shape.start, shape.end, shape.interval),
                    (self_shape.start, self_shape.end, self_shape.interval))
        shape = SpectralShape(
            *[x[0] if x[0] is not None else x[1] for x in s_e_i])
        # Defining proper interpolation bounds.
        # TODO: Provide support for fractional interval like 0.1, etc...
        if (round(self_shape.start) != self_shape.start or
                round(self_shape.end) != self_shape.end):
            runtime_warning(
                'Fractional bound encountered, rounding will occur!')

        shape.start = max(shape.start, np.ceil(self_shape.start))
        shape.end = min(shape.end, np.floor(self_shape.end))

        if interpolator is None:
            if self.is_uniform():
                interpolator = SpragueInterpolator
            else:
                interpolator = CubicSplineInterpolator

        if interpolator_args is None:
            interpolator_args = {}

        interpolator = interpolator(self.wavelengths, self.values,
                                    **interpolator_args)

        self.domain = shape.range()
        self.range = interpolator(self.domain)

        return self

    def align(self,
              shape,
              interpolator=None,
              interpolator_args=None,
              extrapolator=None,
              extrapolator_args=None):
        """
        Aligns the spectral distribution in-place to given spectral shape:
        Interpolates first then extrapolates to fit the given range.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for alignment.
        interpolator : object, optional
            Interpolator class type to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when instantiating the interpolating function.
        extrapolator : object, optional
            Extrapolator class type to use as extrapolating function.
        extrapolator_args : dict_like, optional
            Arguments to use when instantiating the extrapolating function.

        Returns
        -------
        SpectralDistribution
            Aligned spectral distribution.

        Examples
        --------
        >>> from colour.utilities import numpy_print_options
        >>> data = {
        ...     500: 0.0651,
        ...     520: 0.0705,
        ...     540: 0.0772,
        ...     560: 0.0870,
        ...     580: 0.1128,
        ...     600: 0.1360
        ... }
        >>> sd = SpectralDistribution(data)
        >>> with numpy_print_options(suppress=True):
        ...     print(sd.align(SpectralShape(505, 565, 1)))
        ... # doctest: +ELLIPSIS
        [[ 505.            0.0663929...]
         [ 506.            0.0666509...]
         [ 507.            0.0669069...]
         [ 508.            0.0671613...]
         [ 509.            0.0674150...]
         [ 510.            0.0676692...]
         [ 511.            0.0679253...]
         [ 512.            0.0681848...]
         [ 513.            0.0684491...]
         [ 514.            0.0687197...]
         [ 515.            0.0689975...]
         [ 516.            0.0692832...]
         [ 517.            0.0695771...]
         [ 518.            0.0698787...]
         [ 519.            0.0701870...]
         [ 520.            0.0705   ...]
         [ 521.            0.0708155...]
         [ 522.            0.0711336...]
         [ 523.            0.0714547...]
         [ 524.            0.0717789...]
         [ 525.            0.0721063...]
         [ 526.            0.0724367...]
         [ 527.            0.0727698...]
         [ 528.            0.0731051...]
         [ 529.            0.0734423...]
         [ 530.            0.0737808...]
         [ 531.            0.0741203...]
         [ 532.            0.0744603...]
         [ 533.            0.0748006...]
         [ 534.            0.0751409...]
         [ 535.            0.0754813...]
         [ 536.            0.0758220...]
         [ 537.            0.0761633...]
         [ 538.            0.0765060...]
         [ 539.            0.0768511...]
         [ 540.            0.0772   ...]
         [ 541.            0.0775527...]
         [ 542.            0.0779042...]
         [ 543.            0.0782507...]
         [ 544.            0.0785908...]
         [ 545.            0.0789255...]
         [ 546.            0.0792576...]
         [ 547.            0.0795917...]
         [ 548.            0.0799334...]
         [ 549.            0.0802895...]
         [ 550.            0.0806671...]
         [ 551.            0.0810740...]
         [ 552.            0.0815176...]
         [ 553.            0.0820049...]
         [ 554.            0.0825423...]
         [ 555.            0.0831351...]
         [ 556.            0.0837873...]
         [ 557.            0.0845010...]
         [ 558.            0.0852763...]
         [ 559.            0.0861110...]
         [ 560.            0.087    ...]
         [ 561.            0.0879383...]
         [ 562.            0.0889300...]
         [ 563.            0.0899793...]
         [ 564.            0.0910876...]
         [ 565.            0.0922541...]]
        """

        self.interpolate(shape, interpolator, interpolator_args)
        self.extrapolate(shape, extrapolator, extrapolator_args)

        return self

    def trim(self, shape):
        """
        Trims the spectral distribution wavelengths to given spectral shape.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for trimming.

        Returns
        -------
        SpectralDistribution
            Trimmed spectral distribution.

        Examples
        --------
        >>> from colour.utilities import numpy_print_options
        >>> data = {
        ...     500: 0.0651,
        ...     520: 0.0705,
        ...     540: 0.0772,
        ...     560: 0.0870,
        ...     580: 0.1128,
        ...     600: 0.1360
        ... }
        >>> sd = SpectralDistribution(data)
        >>> sd = sd.interpolate(SpectralShape(interval=1))
        >>> with numpy_print_options(suppress=True):
        ...     print(sd.trim(SpectralShape(520, 580, 5)))
        ... # doctest: +ELLIPSIS
        [[ 520.            0.0705   ...]
         [ 521.            0.0708155...]
         [ 522.            0.0711336...]
         [ 523.            0.0714547...]
         [ 524.            0.0717789...]
         [ 525.            0.0721063...]
         [ 526.            0.0724367...]
         [ 527.            0.0727698...]
         [ 528.            0.0731051...]
         [ 529.            0.0734423...]
         [ 530.            0.0737808...]
         [ 531.            0.0741203...]
         [ 532.            0.0744603...]
         [ 533.            0.0748006...]
         [ 534.            0.0751409...]
         [ 535.            0.0754813...]
         [ 536.            0.0758220...]
         [ 537.            0.0761633...]
         [ 538.            0.0765060...]
         [ 539.            0.0768511...]
         [ 540.            0.0772   ...]
         [ 541.            0.0775527...]
         [ 542.            0.0779042...]
         [ 543.            0.0782507...]
         [ 544.            0.0785908...]
         [ 545.            0.0789255...]
         [ 546.            0.0792576...]
         [ 547.            0.0795917...]
         [ 548.            0.0799334...]
         [ 549.            0.0802895...]
         [ 550.            0.0806671...]
         [ 551.            0.0810740...]
         [ 552.            0.0815176...]
         [ 553.            0.0820049...]
         [ 554.            0.0825423...]
         [ 555.            0.0831351...]
         [ 556.            0.0837873...]
         [ 557.            0.0845010...]
         [ 558.            0.0852763...]
         [ 559.            0.0861110...]
         [ 560.            0.087    ...]
         [ 561.            0.0879383...]
         [ 562.            0.0889300...]
         [ 563.            0.0899793...]
         [ 564.            0.0910876...]
         [ 565.            0.0922541...]
         [ 566.            0.0934760...]
         [ 567.            0.0947487...]
         [ 568.            0.0960663...]
         [ 569.            0.0974220...]
         [ 570.            0.0988081...]
         [ 571.            0.1002166...]
         [ 572.            0.1016394...]
         [ 573.            0.1030687...]
         [ 574.            0.1044972...]
         [ 575.            0.1059186...]
         [ 576.            0.1073277...]
         [ 577.            0.1087210...]
         [ 578.            0.1100968...]
         [ 579.            0.1114554...]
         [ 580.            0.1128   ...]]
        """

        start = max(shape.start, self.shape.start)
        end = min(shape.end, self.shape.end)

        indexes = np.where(
            np.logical_and(self.domain >= start, self.domain <= end))

        wavelengths = self.wavelengths[indexes]
        values = self.values[indexes]

        self.wavelengths = wavelengths
        self.values = values

        return self

    def normalise(self, factor=1):
        """
        Normalises the spectral distribution using given normalization factor.

        Parameters
        ----------
        factor : numeric, optional
            Normalization factor

        Returns
        -------
        SpectralDistribution
            Normalised spectral distribution.

        Examples
        --------
        >>> from colour.utilities import numpy_print_options
        >>> data = {
        ...     500: 0.0651,
        ...     520: 0.0705,
        ...     540: 0.0772,
        ...     560: 0.0870,
        ...     580: 0.1128,
        ...     600: 0.1360
        ... }
        >>> sd = SpectralDistribution(data)
        >>> with numpy_print_options(suppress=True):
        ...     print(sd.normalise())  # doctest: +ELLIPSIS
        [[ 500.            0.4786764...]
         [ 520.            0.5183823...]
         [ 540.            0.5676470...]
         [ 560.            0.6397058...]
         [ 580.            0.8294117...]
         [ 600.            1.       ...]]
        """

        self *= 1 / max(self.values) * factor

        return self

    # ------------------------------------------------------------------------#
    # ---              API Changes and Deprecation Management              ---#
    # ------------------------------------------------------------------------#
    @property
    def title(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        runtime_warning(
            str(
                ObjectRenamed('SpectralPowerDistribution.title',
                              'SpectralDistribution.strict_name')))

        return self.strict_name

    @title.setter
    def title(self, value):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        runtime_warning(
            str(
                ObjectRenamed('SpectralPowerDistribution.title',
                              'SpectralDistribution.strict_name')))

        self.strict_name = value

    @property
    def data(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(str(ObjectRemoved('SpectralDistribution.data')))

    @property
    def items(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(str(ObjectRemoved('SpectralDistribution.items')))

    def __iter__(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(
            str(ObjectRemoved('SpectralDistribution.__iter__')))

    def get(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(str(ObjectRemoved('SpectralDistribution.get')))

    def zeros(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(str(ObjectRemoved('SpectralDistribution.zeros')))

    def trim_wavelengths(self, shape):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        runtime_warning(
            str(
                ObjectRenamed('SpectralPowerDistribution.trim_wavelengths',
                              'SpectralDistribution.trim')))

        return self.trim(shape)

    def clone(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        runtime_warning(
            str(
                ObjectRenamed('SpectralPowerDistribution.clone',
                              'SpectralDistribution.copy')))

        return self.copy()


class MultiSpectralDistributions(MultiSignals):
    """
    Defines the multi-spectral distributions: the base object for multi
    spectral computations. It is used to model colour matching functions,
    display primaries, camera sensitivities, etc...

    The multi-spectral distributions will be initialised according to
    *CIE 15:2004* recommendation: the method developed by *Sprague (1880)* will
    be used for interpolating functions having a uniformly spaced independent
    variable and the *Cubic Spline* method for non-uniformly spaced independent
    variable. Extrapolation is performed according to *CIE 167:2005*
    recommendation.

    Parameters
    ----------
    data : Series or Dataframe or Signal or MultiSignals or \
MultiSpectralDistributions or array_like or dict_like, optional
        Data to be stored in the multi-spectral distributions.
    domain : array_like, optional
        Values to initialise the multiple :class:`colour.SpectralDistribution`
        class instances :attr:`colour.continuous.Signal.wavelengths` attribute
        with. If both ``data`` and ``domain`` arguments are defined, the latter
        will be used to initialise the
        :attr:`colour.continuous.Signal.wavelengths` attribute.
    labels : array_like, optional
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.

    Other Parameters
    ----------------
    name : unicode, optional
       Multi-spectral distributions name.
    interpolator : object, optional
        Interpolator class type to use as interpolating function for the
        :class:`colour.SpectralDistribution` class instances.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function of the
        :class:`colour.SpectralDistribution` class instances.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function for the
        :class:`colour.SpectralDistribution` class instances.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating the extrapolating function of the
        :class:`colour.SpectralDistribution` class instances.
    strict_labels : array_like, optional
        Multi-spectral distributions labels for figures, default to
        :attr:`colour.MultiSpectralDistributions.labels` attribute value.

    Attributes
    ----------
    strict_name
    strict_labels
    wavelengths
    values
    shape

    Methods
    -------
    extrapolate
    interpolate
    align
    trim
    normalise
    to_sds

    References
    ----------
    :cite:`CIETC1-382005e`, :cite:`CIETC1-382005g`, :cite:`CIETC1-482004l`

    Examples
    --------
    Instantiating the multi-spectral distributions with a uniformly spaced
    independent variable:

    >>> from colour.utilities import numpy_print_options
    >>> data = {
    ...     500: (0.004900, 0.323000, 0.272000),
    ...     510: (0.009300, 0.503000, 0.158200),
    ...     520: (0.063270, 0.710000, 0.078250),
    ...     530: (0.165500, 0.862000, 0.042160),
    ...     540: (0.290400, 0.954000, 0.020300),
    ...     550: (0.433450, 0.994950, 0.008750),
    ...     560: (0.594500, 0.995000, 0.003900)
    ... }
    >>> labels = ('x_bar', 'y_bar', 'z_bar')
    >>> with numpy_print_options(suppress=True):
    ...     MultiSpectralDistributions(data, labels=labels)
    ... # doctest: +ELLIPSIS
    MultiSpectral...([[ 500.     ,    0.0049 ,    0.323  ,    0.272  ],
                 ...  [ 510.     ,    0.0093 ,    0.503  ,    0.1582 ],
                 ...  [ 520.     ,    0.06327,    0.71   ,    0.07825],
                 ...  [ 530.     ,    0.1655 ,    0.862  ,    0.04216],
                 ...  [ 540.     ,    0.2904 ,    0.954  ,    0.0203 ],
                 ...  [ 550.     ,    0.43345,    0.99495,    0.00875],
                 ...  [ 560.     ,    0.5945 ,    0.995  ,    0.0039 ]],
                 ... labels=[...'x_bar', ...'y_bar', ...'z_bar'],
                 ... interpolator=SpragueInterpolator,
                 ... interpolator_args={},
                 ... extrapolator=Extrapolator,
                 ... extrapolator_args={...})

    Instantiating a spectral distribution with a non-uniformly spaced
    independent variable:

    >>> data[511] = (0.00314, 0.31416, 0.03142)
    >>> with numpy_print_options(suppress=True):
    ...     MultiSpectralDistributions(data, labels=labels)
    ... # doctest: +ELLIPSIS
    MultiSpectral...([[ 500.     ,    0.0049 ,    0.323  ,    0.272  ],
                 ...  [ 510.     ,    0.0093 ,    0.503  ,    0.1582 ],
                 ...  [ 511.     ,    0.00314,    0.31416,    0.03142],
                 ...  [ 520.     ,    0.06327,    0.71   ,    0.07825],
                 ...  [ 530.     ,    0.1655 ,    0.862  ,    0.04216],
                 ...  [ 540.     ,    0.2904 ,    0.954  ,    0.0203 ],
                 ...  [ 550.     ,    0.43345,    0.99495,    0.00875],
                 ...  [ 560.     ,    0.5945 ,    0.995  ,    0.0039 ]],
                 ... labels=[...'x_bar', ...'y_bar', ...'z_bar'],
                 ... interpolator=CubicSplineInterpolator,
                 ... interpolator_args={},
                 ... extrapolator=Extrapolator,
                 ... extrapolator_args={...})
    """

    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        signals = self.multi_signals_unpack_data(data, domain, labels)

        domain = signals[list(signals.keys())[0]].domain if signals else None
        uniform = is_uniform(domain) if domain is not None else True

        # Initialising with *CIE 15:2004* and *CIE 167:2005* recommendations
        # defaults.
        kwargs['interpolator'] = kwargs.get(
            'interpolator', SpragueInterpolator
            if uniform else CubicSplineInterpolator)
        kwargs['interpolator_args'] = kwargs.get('interpolator_args', {})

        kwargs['extrapolator'] = kwargs.get('extrapolator', Extrapolator)
        kwargs['extrapolator_args'] = kwargs.get('extrapolator_args', {
            'method': 'Constant',
            'left': None,
            'right': None
        })

        super(MultiSpectralDistributions, self).__init__(
            signals, domain, signal_type=SpectralDistribution, **kwargs)

        self._strict_name = None
        self.strict_name = kwargs.get('strict_name')
        self._strict_labels = None
        self.strict_labels = kwargs.get('strict_labels')

    @property
    def strict_name(self):
        """
        Getter and setter property for the multi-spectral distributions strict
        name.

        Parameters
        ----------
        value : unicode
            Value to set the multi-spectral distributions strict name with.

        Returns
        -------
        unicode
            Multi-spectral distributions strict name.
        """

        if self._strict_name is not None:
            return self._strict_name
        else:
            return self._name

    @strict_name.setter
    def strict_name(self, value):
        """
        Setter for **self.strict_name** property.
        """

        if value is not None:
            assert is_string(value), (
                ('"{0}" attribute: "{1}" is not a "string" like object!'
                 ).format('strict_name', value))
            self._strict_name = value

    @property
    def strict_labels(self):
        """
        Getter and setter property for the multi-spectral distributions strict
        labels.

        Parameters
        ----------
        value : array_like
            Value to set the multi-spectral distributions strict labels with.

        Returns
        -------
        array_like
            Multi-spectral distributions strict labels.
        """

        if self._strict_labels is not None:
            return self._strict_labels
        else:
            return self.labels

    @strict_labels.setter
    def strict_labels(self, value):
        """
        Setter for **self.strict_labels** property.
        """

        if value is not None:
            assert is_iterable(value), (
                '"{0}" attribute: "{1}" is not an "iterable" like object!'.
                format('strict_labels', value))

            assert len(value) == len(
                self.labels), ('"{0}" attribute: length must be "{1}"!'.format(
                    'strict_labels', len(self.labels)))
            self._strict_labels = value

    @property
    def wavelengths(self):
        """
        Getter and setter property for the multi-spectral distributions
        wavelengths :math:`\\lambda_n`.

        Parameters
        ----------
        value : array_like
            Value to set the multi-spectral distributions wavelengths
            :math:`\\lambda_n` with.

        Returns
        -------
        ndarray
            Multi-spectral distributions wavelengths :math:`\\lambda_n`.
        """

        return self.domain

    @wavelengths.setter
    def wavelengths(self, value):
        """
        Setter for the **self.wavelengths** property.
        """

        self.domain = value

    @property
    def values(self):
        """
        Getter and setter property for the multi-spectral distributions values.

        Parameters
        ----------
        value : array_like
            Value to set the multi-spectral distributions wavelengths values
            with.

        Returns
        -------
        ndarray
            Multi-spectral distributions values.
        """

        return self.range

    @values.setter
    def values(self, value):
        """
        Setter for the **self.values** property.
        """

        self.range = value

    @property
    def shape(self):
        """
        Getter and setter property for the multi-spectral distributions shape.

        Returns
        -------
        SpectralShape
            Multi-spectral distributions shape.

        Notes
        -----
        -   Multi-spectral distributions with a non-uniformly spaced
            independent variable have multiple intervals, in that case
            :attr:`colour.MultiSpectralDistributions.shape` attribute returns
            the *minimum* interval size.

        Warning
        -------
        :attr:`colour.MultiSpectralDistributions.shape` attribute is read only.

        Examples
        --------
        Shape of the multi-spectral distributions with a uniformly spaced
        independent variable:

        >>> from colour.utilities import numpy_print_options
        >>> data = {
        ...     500: (0.004900, 0.323000, 0.272000),
        ...     510: (0.009300, 0.503000, 0.158200),
        ...     520: (0.063270, 0.710000, 0.078250),
        ...     530: (0.165500, 0.862000, 0.042160),
        ...     540: (0.290400, 0.954000, 0.020300),
        ...     550: (0.433450, 0.994950, 0.008750),
        ...     560: (0.594500, 0.995000, 0.003900)
        ... }
        >>> MultiSpectralDistributions(data).shape
        SpectralShape(500.0, 560.0, 10.0)

        Shape of the multi-spectral distributions with a non-uniformly spaced
        independent variable:

        >>> data[511] = (0.00314, 0.31416, 0.03142)
        >>> MultiSpectralDistributions(data).shape
        SpectralShape(500.0, 560.0, 1.0)
        """

        if self.signals:
            return first_item(self._signals.values()).shape

    def extrapolate(self, shape, extrapolator=None, extrapolator_args=None):
        """
        Extrapolates the multi-spectral distributions in-place according to
        *CIE 15:2004* and *CIE 167:2005* recommendations or given extrapolation
        arguments.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for extrapolation.
        extrapolator : object, optional
            Extrapolator class type to use as extrapolating function.
        extrapolator_args : dict_like, optional
            Arguments to use when instantiating the extrapolating function.

        Returns
        -------
        MultiSpectralDistributions
            Extrapolated multi-spectral distributions.

        References
        ----------
        :cite:`CIETC1-382005g`, :cite:`CIETC1-482004l`

        Examples
        --------
        >>> from colour.utilities import numpy_print_options
        >>> data = {
        ...     500: (0.004900, 0.323000, 0.272000),
        ...     510: (0.009300, 0.503000, 0.158200),
        ...     520: (0.063270, 0.710000, 0.078250),
        ...     530: (0.165500, 0.862000, 0.042160),
        ...     540: (0.290400, 0.954000, 0.020300),
        ...     550: (0.433450, 0.994950, 0.008750),
        ...     560: (0.594500, 0.995000, 0.003900)
        ... }
        >>> msds = MultiSpectralDistributions(data)
        >>> msds.extrapolate(SpectralShape(400, 700)).shape
        SpectralShape(400.0, 700.0, 10.0)
        >>> with numpy_print_options(suppress=True):
        ...     print(msds)
        [[ 400.         0.0049     0.323      0.272  ]
         [ 410.         0.0049     0.323      0.272  ]
         [ 420.         0.0049     0.323      0.272  ]
         [ 430.         0.0049     0.323      0.272  ]
         [ 440.         0.0049     0.323      0.272  ]
         [ 450.         0.0049     0.323      0.272  ]
         [ 460.         0.0049     0.323      0.272  ]
         [ 470.         0.0049     0.323      0.272  ]
         [ 480.         0.0049     0.323      0.272  ]
         [ 490.         0.0049     0.323      0.272  ]
         [ 500.         0.0049     0.323      0.272  ]
         [ 510.         0.0093     0.503      0.1582 ]
         [ 520.         0.06327    0.71       0.07825]
         [ 530.         0.1655     0.862      0.04216]
         [ 540.         0.2904     0.954      0.0203 ]
         [ 550.         0.43345    0.99495    0.00875]
         [ 560.         0.5945     0.995      0.0039 ]
         [ 570.         0.5945     0.995      0.0039 ]
         [ 580.         0.5945     0.995      0.0039 ]
         [ 590.         0.5945     0.995      0.0039 ]
         [ 600.         0.5945     0.995      0.0039 ]
         [ 610.         0.5945     0.995      0.0039 ]
         [ 620.         0.5945     0.995      0.0039 ]
         [ 630.         0.5945     0.995      0.0039 ]
         [ 640.         0.5945     0.995      0.0039 ]
         [ 650.         0.5945     0.995      0.0039 ]
         [ 660.         0.5945     0.995      0.0039 ]
         [ 670.         0.5945     0.995      0.0039 ]
         [ 680.         0.5945     0.995      0.0039 ]
         [ 690.         0.5945     0.995      0.0039 ]
         [ 700.         0.5945     0.995      0.0039 ]]
        """

        for signal in self.signals.values():
            signal.extrapolate(shape, extrapolator, extrapolator_args)

        return self

    def interpolate(self, shape, interpolator=None, interpolator_args=None):
        """
        Interpolates the multi-spectral distributions in-place according to
        *CIE 167:2005* recommendation or given interpolation arguments.

        Parameters
        ----------
        shape : SpectralShape, optional
            Spectral shape used for interpolation.
        interpolator : object, optional
            Interpolator class type to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when instantiating the interpolating function.

        Returns
        -------
        MultiSpectralDistributions
            Interpolated multi-spectral distributions.

        Notes
        -----
        -   See :meth:`colour.SpectralDistribution.interpolate` method notes
            section.

        Warning
        -------
        See :meth:`colour.SpectralDistribution.interpolate` method warning
        section.

        References
        ----------
        :cite:`CIETC1-382005e`

        Examples
        --------
        Multi-spectral distributions with a uniformly spaced independent
        variable uses *Sprague (1880)* interpolation:

        >>> from colour.utilities import numpy_print_options
        >>> data = {
        ...     500: (0.004900, 0.323000, 0.272000),
        ...     510: (0.009300, 0.503000, 0.158200),
        ...     520: (0.063270, 0.710000, 0.078250),
        ...     530: (0.165500, 0.862000, 0.042160),
        ...     540: (0.290400, 0.954000, 0.020300),
        ...     550: (0.433450, 0.994950, 0.008750),
        ...     560: (0.594500, 0.995000, 0.003900)
        ... }
        >>> msds = MultiSpectralDistributions(data)
        >>> with numpy_print_options(suppress=True):
        ...     print(msds.interpolate(SpectralShape(interval=1)))
        ... # doctest: +ELLIPSIS
        [[ 500.            0.0049   ...    0.323    ...    0.272    ...]
         [ 501.            0.0043252...    0.3400642...    0.2599848...]
         [ 502.            0.0037950...    0.3572165...    0.2479849...]
         [ 503.            0.0033761...    0.3744030...    0.2360688...]
         [ 504.            0.0031397...    0.3916650...    0.2242878...]
         [ 505.            0.0031582...    0.4091067...    0.2126801...]
         [ 506.            0.0035019...    0.4268629...    0.2012748...]
         [ 507.            0.0042365...    0.4450668...    0.1900968...]
         [ 508.            0.0054192...    0.4638181...    0.1791709...]
         [ 509.            0.0070965...    0.4831505...    0.1685260...]
         [ 510.            0.0093   ...    0.503    ...    0.1582   ...]
         [ 511.            0.0120562...    0.5232543...    0.1482365...]
         [ 512.            0.0154137...    0.5439717...    0.1386625...]
         [ 513.            0.0193991...    0.565139 ...    0.1294993...]
         [ 514.            0.0240112...    0.5866255...    0.1207676...]
         [ 515.            0.0292289...    0.6082226...    0.1124864...]
         [ 516.            0.0350192...    0.6296821...    0.1046717...]
         [ 517.            0.0413448...    0.6507558...    0.0973361...]
         [ 518.            0.0481727...    0.6712346...    0.0904871...]
         [ 519.            0.0554816...    0.6909873...    0.0841267...]
         [ 520.            0.06327  ...    0.71     ...    0.07825  ...]
         [ 521.            0.0715642...    0.7283456...    0.0728614...]
         [ 522.            0.0803970...    0.7459679...    0.0680051...]
         [ 523.            0.0897629...    0.7628184...    0.0636823...]
         [ 524.            0.0996227...    0.7789004...    0.0598449...]
         [ 525.            0.1099142...    0.7942533...    0.0564111...]
         [ 526.            0.1205637...    0.8089368...    0.0532822...]
         [ 527.            0.1314973...    0.8230153...    0.0503588...]
         [ 528.            0.1426523...    0.8365417...    0.0475571...]
         [ 529.            0.1539887...    0.8495422...    0.0448253...]
         [ 530.            0.1655   ...    0.862    ...    0.04216  ...]
         [ 531.            0.1772055...    0.8738585...    0.0395936...]
         [ 532.            0.1890877...    0.8850940...    0.0371046...]
         [ 533.            0.2011304...    0.8957073...    0.0346733...]
         [ 534.            0.2133310...    0.9057092...    0.0323006...]
         [ 535.            0.2256968...    0.9151181...    0.0300011...]
         [ 536.            0.2382403...    0.9239560...    0.0277974...]
         [ 537.            0.2509754...    0.9322459...    0.0257131...]
         [ 538.            0.2639130...    0.9400080...    0.0237668...]
         [ 539.            0.2770569...    0.9472574...    0.0219659...]
         [ 540.            0.2904   ...    0.954    ...    0.0203   ...]
         [ 541.            0.3039194...    0.9602409...    0.0187414...]
         [ 542.            0.3175893...    0.9660106...    0.0172748...]
         [ 543.            0.3314022...    0.9713260...    0.0158947...]
         [ 544.            0.3453666...    0.9761850...    0.0146001...]
         [ 545.            0.3595019...    0.9805731...    0.0133933...]
         [ 546.            0.3738324...    0.9844703...    0.0122777...]
         [ 547.            0.3883818...    0.9878583...    0.0112562...]
         [ 548.            0.4031674...    0.9907270...    0.0103302...]
         [ 549.            0.4181943...    0.9930817...    0.0094972...]
         [ 550.            0.43345  ...    0.99495  ...    0.00875  ...]
         [ 551.            0.4489082...    0.9963738...    0.0080748...]
         [ 552.            0.4645599...    0.9973682...    0.0074580...]
         [ 553.            0.4803950...    0.9979568...    0.0068902...]
         [ 554.            0.4963962...    0.9981802...    0.0063660...]
         [ 555.            0.5125410...    0.9980910...    0.0058818...]
         [ 556.            0.5288034...    0.9977488...    0.0054349...]
         [ 557.            0.5451560...    0.9972150...    0.0050216...]
         [ 558.            0.5615719...    0.9965479...    0.0046357...]
         [ 559.            0.5780267...    0.9957974...    0.0042671...]
         [ 560.            0.5945   ...    0.995    ...    0.0039   ...]]

        Multi-spectral distributions with a non-uniformly spaced independent
        variable uses *Cubic Spline* interpolation:

        >>> data[511] = (0.00314, 0.31416, 0.03142)
        >>> msds = MultiSpectralDistributions(data)
        >>> with numpy_print_options(suppress=True):
        ...     print(msds.interpolate(SpectralShape(interval=1)))
        ... # doctest: +ELLIPSIS
        [[ 500.            0.0049   ...    0.323    ...    0.272    ...]
         [ 501.            0.0300110...    0.9455153...    0.5985102...]
         [ 502.            0.0462136...    1.3563103...    0.8066498...]
         [ 503.            0.0547925...    1.5844039...    0.9126502...]
         [ 504.            0.0570325...    1.6588148...    0.9327429...]
         [ 505.            0.0542183...    1.6085619...    0.8831594...]
         [ 506.            0.0476346...    1.4626640...    0.7801312...]
         [ 507.            0.0385662...    1.2501401...    0.6398896...]
         [ 508.            0.0282978...    1.0000089...    0.4786663...]
         [ 509.            0.0181142...    0.7412892...    0.3126925...]
         [ 510.            0.0093   ...    0.503    ...    0.1582   ...]
         [ 511.            0.00314  ...    0.31416  ...    0.03142  ...]
         [ 512.            0.0006228...    0.1970419...   -0.0551709...]
         [ 513.            0.0015528...    0.1469341...   -0.1041165...]
         [ 514.            0.0054381...    0.1523785...   -0.1217152...]
         [ 515.            0.0117869...    0.2019173...   -0.1142659...]
         [ 516.            0.0201073...    0.2840925...   -0.0880670...]
         [ 517.            0.0299077...    0.3874463...   -0.0494174...]
         [ 518.            0.0406961...    0.5005208...   -0.0046156...]
         [ 519.            0.0519808...    0.6118579...    0.0400397...]
         [ 520.            0.06327  ...    0.71     ...    0.07825  ...]
         [ 521.            0.0741690...    0.7859059...    0.1050384...]
         [ 522.            0.0846726...    0.8402033...    0.1207164...]
         [ 523.            0.0948728...    0.8759363...    0.1269173...]
         [ 524.            0.1048614...    0.8961496...    0.1252743...]
         [ 525.            0.1147305...    0.9038874...    0.1174207...]
         [ 526.            0.1245719...    0.9021942...    0.1049899...]
         [ 527.            0.1344776...    0.8941145...    0.0896151...]
         [ 528.            0.1445395...    0.8826926...    0.0729296...]
         [ 529.            0.1548497...    0.8709729...    0.0565668...]
         [ 530.            0.1655   ...    0.862    ...    0.04216  ...]
         [ 531.            0.1765618...    0.858179 ...    0.0309976...]
         [ 532.            0.1880244...    0.8593588...    0.0229897...]
         [ 533.            0.1998566...    0.8647493...    0.0177013...]
         [ 534.            0.2120269...    0.8735601...    0.0146975...]
         [ 535.            0.2245042...    0.8850011...    0.0135435...]
         [ 536.            0.2372572...    0.8982820...    0.0138044...]
         [ 537.            0.2502546...    0.9126126...    0.0150454...]
         [ 538.            0.2634650...    0.9272026...    0.0168315...]
         [ 539.            0.2768572...    0.9412618...    0.0187280...]
         [ 540.            0.2904   ...    0.954    ...    0.0203   ...]
         [ 541.            0.3040682...    0.9647869...    0.0211987...]
         [ 542.            0.3178617...    0.9736329...    0.0214207...]
         [ 543.            0.3317865...    0.9807080...    0.0210486...]
         [ 544.            0.3458489...    0.9861825...    0.0201650...]
         [ 545.            0.3600548...    0.9902267...    0.0188525...]
         [ 546.            0.3744103...    0.9930107...    0.0171939...]
         [ 547.            0.3889215...    0.9947048...    0.0152716...]
         [ 548.            0.4035944...    0.9954792...    0.0131685...]
         [ 549.            0.4184352...    0.9955042...    0.0109670...]
         [ 550.            0.43345  ...    0.99495  ...    0.00875  ...]
         [ 551.            0.4486447...    0.9939867...    0.0065999...]
         [ 552.            0.4640255...    0.9927847...    0.0045994...]
         [ 553.            0.4795984...    0.9915141...    0.0028313...]
         [ 554.            0.4953696...    0.9903452...    0.0013781...]
         [ 555.            0.5113451...    0.9894483...    0.0003224...]
         [ 556.            0.5275310...    0.9889934...   -0.0002530...]
         [ 557.            0.5439334...    0.9891509...   -0.0002656...]
         [ 558.            0.5605583...    0.9900910...    0.0003672...]
         [ 559.            0.5774118...    0.9919840...    0.0017282...]
         [ 560.            0.5945   ...    0.995    ...    0.0039   ...]]
        """

        for signal in self.signals.values():
            signal.interpolate(shape, interpolator, interpolator_args)

        return self

    def align(self,
              shape,
              interpolator=None,
              interpolator_args=None,
              extrapolator=None,
              extrapolator_args=None):
        """
        Aligns the multi-spectral distributions in-place to given spectral
        shape: Interpolates first then extrapolates to fit the given range.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for alignment.
        interpolator : object, optional
            Interpolator class type to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when instantiating the interpolating function.
        extrapolator : object, optional
            Extrapolator class type to use as extrapolating function.
        extrapolator_args : dict_like, optional
            Arguments to use when instantiating the extrapolating function.

        Returns
        -------
        MultiSpectralDistributions
            Aligned multi-spectral distributions.

        Examples
        --------
        >>> from colour.utilities import numpy_print_options
        >>> data = {
        ...     500: (0.004900, 0.323000, 0.272000),
        ...     510: (0.009300, 0.503000, 0.158200),
        ...     520: (0.063270, 0.710000, 0.078250),
        ...     530: (0.165500, 0.862000, 0.042160),
        ...     540: (0.290400, 0.954000, 0.020300),
        ...     550: (0.433450, 0.994950, 0.008750),
        ...     560: (0.594500, 0.995000, 0.003900)
        ... }
        >>> msds = MultiSpectralDistributions(data)
        >>> with numpy_print_options(suppress=True):
        ...     print(msds.align(SpectralShape(505, 565, 1)))
        ... # doctest: +ELLIPSIS
        [[ 505.            0.0031582...    0.4091067...    0.2126801...]
         [ 506.            0.0035019...    0.4268629...    0.2012748...]
         [ 507.            0.0042365...    0.4450668...    0.1900968...]
         [ 508.            0.0054192...    0.4638181...    0.1791709...]
         [ 509.            0.0070965...    0.4831505...    0.1685260...]
         [ 510.            0.0093   ...    0.503    ...    0.1582   ...]
         [ 511.            0.0120562...    0.5232543...    0.1482365...]
         [ 512.            0.0154137...    0.5439717...    0.1386625...]
         [ 513.            0.0193991...    0.565139 ...    0.1294993...]
         [ 514.            0.0240112...    0.5866255...    0.1207676...]
         [ 515.            0.0292289...    0.6082226...    0.1124864...]
         [ 516.            0.0350192...    0.6296821...    0.1046717...]
         [ 517.            0.0413448...    0.6507558...    0.0973361...]
         [ 518.            0.0481727...    0.6712346...    0.0904871...]
         [ 519.            0.0554816...    0.6909873...    0.0841267...]
         [ 520.            0.06327  ...    0.71     ...    0.07825  ...]
         [ 521.            0.0715642...    0.7283456...    0.0728614...]
         [ 522.            0.0803970...    0.7459679...    0.0680051...]
         [ 523.            0.0897629...    0.7628184...    0.0636823...]
         [ 524.            0.0996227...    0.7789004...    0.0598449...]
         [ 525.            0.1099142...    0.7942533...    0.0564111...]
         [ 526.            0.1205637...    0.8089368...    0.0532822...]
         [ 527.            0.1314973...    0.8230153...    0.0503588...]
         [ 528.            0.1426523...    0.8365417...    0.0475571...]
         [ 529.            0.1539887...    0.8495422...    0.0448253...]
         [ 530.            0.1655   ...    0.862    ...    0.04216  ...]
         [ 531.            0.1772055...    0.8738585...    0.0395936...]
         [ 532.            0.1890877...    0.8850940...    0.0371046...]
         [ 533.            0.2011304...    0.8957073...    0.0346733...]
         [ 534.            0.2133310...    0.9057092...    0.0323006...]
         [ 535.            0.2256968...    0.9151181...    0.0300011...]
         [ 536.            0.2382403...    0.9239560...    0.0277974...]
         [ 537.            0.2509754...    0.9322459...    0.0257131...]
         [ 538.            0.2639130...    0.9400080...    0.0237668...]
         [ 539.            0.2770569...    0.9472574...    0.0219659...]
         [ 540.            0.2904   ...    0.954    ...    0.0203   ...]
         [ 541.            0.3039194...    0.9602409...    0.0187414...]
         [ 542.            0.3175893...    0.9660106...    0.0172748...]
         [ 543.            0.3314022...    0.9713260...    0.0158947...]
         [ 544.            0.3453666...    0.9761850...    0.0146001...]
         [ 545.            0.3595019...    0.9805731...    0.0133933...]
         [ 546.            0.3738324...    0.9844703...    0.0122777...]
         [ 547.            0.3883818...    0.9878583...    0.0112562...]
         [ 548.            0.4031674...    0.9907270...    0.0103302...]
         [ 549.            0.4181943...    0.9930817...    0.0094972...]
         [ 550.            0.43345  ...    0.99495  ...    0.00875  ...]
         [ 551.            0.4489082...    0.9963738...    0.0080748...]
         [ 552.            0.4645599...    0.9973682...    0.0074580...]
         [ 553.            0.4803950...    0.9979568...    0.0068902...]
         [ 554.            0.4963962...    0.9981802...    0.0063660...]
         [ 555.            0.5125410...    0.9980910...    0.0058818...]
         [ 556.            0.5288034...    0.9977488...    0.0054349...]
         [ 557.            0.5451560...    0.9972150...    0.0050216...]
         [ 558.            0.5615719...    0.9965479...    0.0046357...]
         [ 559.            0.5780267...    0.9957974...    0.0042671...]
         [ 560.            0.5945   ...    0.995    ...    0.0039   ...]
         [ 561.            0.5945   ...    0.995    ...    0.0039   ...]
         [ 562.            0.5945   ...    0.995    ...    0.0039   ...]
         [ 563.            0.5945   ...    0.995    ...    0.0039   ...]
         [ 564.            0.5945   ...    0.995    ...    0.0039   ...]
         [ 565.            0.5945   ...    0.995    ...    0.0039   ...]]
        """

        for signal in self.signals.values():
            signal.align(shape, interpolator, interpolator_args, extrapolator,
                         extrapolator_args)

        return self

    def trim(self, shape):
        """
        Trims the multi-spectral distributions wavelengths to given shape.

        Parameters
        ----------
        shape : SpectralShape
            Spectral shape used for trimming.

        Returns
        -------
        MultiSpectralDistributions
            Trimmed multi-spectral distributions.

        Examples
        --------
        >>> from colour.utilities import numpy_print_options
        >>> data = {
        ...     500: (0.004900, 0.323000, 0.272000),
        ...     510: (0.009300, 0.503000, 0.158200),
        ...     520: (0.063270, 0.710000, 0.078250),
        ...     530: (0.165500, 0.862000, 0.042160),
        ...     540: (0.290400, 0.954000, 0.020300),
        ...     550: (0.433450, 0.994950, 0.008750),
        ...     560: (0.594500, 0.995000, 0.003900)
        ... }
        >>> msds = MultiSpectralDistributions(data)
        >>> msds = msds.interpolate(SpectralShape(interval=1))
        >>> with numpy_print_options(suppress=True):
        ...     print(msds.trim(SpectralShape(520, 580, 5)))
        ... # doctest: +ELLIPSIS
        [[ 520.            0.06327  ...    0.71     ...    0.07825  ...]
         [ 521.            0.0715642...    0.7283456...    0.0728614...]
         [ 522.            0.0803970...    0.7459679...    0.0680051...]
         [ 523.            0.0897629...    0.7628184...    0.0636823...]
         [ 524.            0.0996227...    0.7789004...    0.0598449...]
         [ 525.            0.1099142...    0.7942533...    0.0564111...]
         [ 526.            0.1205637...    0.8089368...    0.0532822...]
         [ 527.            0.1314973...    0.8230153...    0.0503588...]
         [ 528.            0.1426523...    0.8365417...    0.0475571...]
         [ 529.            0.1539887...    0.8495422...    0.0448253...]
         [ 530.            0.1655   ...    0.862    ...    0.04216  ...]
         [ 531.            0.1772055...    0.8738585...    0.0395936...]
         [ 532.            0.1890877...    0.8850940...    0.0371046...]
         [ 533.            0.2011304...    0.8957073...    0.0346733...]
         [ 534.            0.2133310...    0.9057092...    0.0323006...]
         [ 535.            0.2256968...    0.9151181...    0.0300011...]
         [ 536.            0.2382403...    0.9239560...    0.0277974...]
         [ 537.            0.2509754...    0.9322459...    0.0257131...]
         [ 538.            0.2639130...    0.9400080...    0.0237668...]
         [ 539.            0.2770569...    0.9472574...    0.0219659...]
         [ 540.            0.2904   ...    0.954    ...    0.0203   ...]
         [ 541.            0.3039194...    0.9602409...    0.0187414...]
         [ 542.            0.3175893...    0.9660106...    0.0172748...]
         [ 543.            0.3314022...    0.9713260...    0.0158947...]
         [ 544.            0.3453666...    0.9761850...    0.0146001...]
         [ 545.            0.3595019...    0.9805731...    0.0133933...]
         [ 546.            0.3738324...    0.9844703...    0.0122777...]
         [ 547.            0.3883818...    0.9878583...    0.0112562...]
         [ 548.            0.4031674...    0.9907270...    0.0103302...]
         [ 549.            0.4181943...    0.9930817...    0.0094972...]
         [ 550.            0.43345  ...    0.99495  ...    0.00875  ...]
         [ 551.            0.4489082...    0.9963738...    0.0080748...]
         [ 552.            0.4645599...    0.9973682...    0.0074580...]
         [ 553.            0.4803950...    0.9979568...    0.0068902...]
         [ 554.            0.4963962...    0.9981802...    0.0063660...]
         [ 555.            0.5125410...    0.9980910...    0.0058818...]
         [ 556.            0.5288034...    0.9977488...    0.0054349...]
         [ 557.            0.5451560...    0.9972150...    0.0050216...]
         [ 558.            0.5615719...    0.9965479...    0.0046357...]
         [ 559.            0.5780267...    0.9957974...    0.0042671...]
         [ 560.            0.5945   ...    0.995    ...    0.0039   ...]]
        """

        for signal in self.signals.values():
            signal.trim(shape)

        return self

    def normalise(self, factor=1):
        """
        Normalises the multi-spectral distributions with given normalization
        factor.

        Parameters
        ----------
        factor : numeric, optional
            Normalization factor

        Returns
        -------
        MultiSpectralDistributions
            Normalised multi- spectral distribution.

        Notes
        -----
        -   The implementation uses the maximum value for each
            :class:`colour.SpectralDistribution` class instances.

        Examples
        --------
        >>> from colour.utilities import numpy_print_options
        >>> data = {
        ...     500: (0.004900, 0.323000, 0.272000),
        ...     510: (0.009300, 0.503000, 0.158200),
        ...     520: (0.063270, 0.710000, 0.078250),
        ...     530: (0.165500, 0.862000, 0.042160),
        ...     540: (0.290400, 0.954000, 0.020300),
        ...     550: (0.433450, 0.994950, 0.008750),
        ...     560: (0.594500, 0.995000, 0.003900)
        ... }
        >>> msds = MultiSpectralDistributions(data)
        >>> with numpy_print_options(suppress=True):
        ...     print(msds.normalise())  # doctest: +ELLIPSIS
        [[ 500.            0.0082422...    0.3246231...    1.       ...]
         [ 510.            0.0156434...    0.5055276...    0.5816176...]
         [ 520.            0.1064255...    0.7135678...    0.2876838...]
         [ 530.            0.2783852...    0.8663316...    0.155    ...]
         [ 540.            0.4884777...    0.9587939...    0.0746323...]
         [ 550.            0.7291000...    0.9999497...    0.0321691...]
         [ 560.            1.       ...    1.       ...    0.0143382...]]
        """

        for signal in self.signals.values():
            signal.normalise(factor)

        return self

    def to_sds(self):
        """
        Converts the multi-spectral distributions to a list of spectral
        distributions and update their name and strict name using the labels
        and strict labels.

        Returns
        -------
        list
            List of spectral distributions.

        Examples
        --------
        >>> from colour.utilities import numpy_print_options
        >>> data = {
        ...     500: (0.004900, 0.323000, 0.272000),
        ...     510: (0.009300, 0.503000, 0.158200),
        ...     520: (0.063270, 0.710000, 0.078250),
        ...     530: (0.165500, 0.862000, 0.042160),
        ...     540: (0.290400, 0.954000, 0.020300),
        ...     550: (0.433450, 0.994950, 0.008750),
        ...     560: (0.594500, 0.995000, 0.003900)
        ... }
        >>> msds = MultiSpectralDistributions(data)
        >>> with numpy_print_options(suppress=True):
        ...     for sd in msds.to_sds():
        ...         print(sd)  # doctest: +ELLIPSIS
        [[ 500.         0.0049 ...]
         [ 510.         0.0093 ...]
         [ 520.         0.06327...]
         [ 530.         0.1655 ...]
         [ 540.         0.2904 ...]
         [ 550.         0.43345...]
         [ 560.         0.5945 ...]]
        [[ 500.         0.323  ...]
         [ 510.         0.503  ...]
         [ 520.         0.71   ...]
         [ 530.         0.862  ...]
         [ 540.         0.954  ...]
         [ 550.         0.99495...]
         [ 560.         0.995  ...]]
        [[ 500.         0.272  ...]
         [ 510.         0.1582 ...]
         [ 520.         0.07825...]
         [ 530.         0.04216...]
         [ 540.         0.0203 ...]
         [ 550.         0.00875...]
         [ 560.         0.0039 ...]]
        """

        sds = []
        for i, signal in enumerate(self.signals.values()):
            signal = signal.copy()
            signal.name = '{0} - {1}'.format(self.labels[i], signal.name)
            signal.strict_name = '{0} - {1}'.format(self.strict_labels[i],
                                                    signal.strict_name)

            sds.append(signal)

        return sds

    # ------------------------------------------------------------------------#
    # ---              API Changes and Deprecation Management              ---#
    # ------------------------------------------------------------------------#
    @property
    def title(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        runtime_warning(
            str(
                ObjectRenamed('TriSpectralPowerDistribution.title',
                              'SpectralDistribution.strict_name')))

        return self.strict_name

    @title.setter
    def title(self, value):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        runtime_warning(
            str(
                ObjectRenamed('TriSpectralPowerDistribution.title',
                              'SpectralDistribution.strict_name')))

        self.strict_name = value

    @property
    def data(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(
            str(ObjectRemoved('MultiSpectralDistributions.data')))

    @property
    def items(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(
            str(ObjectRemoved('MultiSpectralDistributions.items')))

    @property
    def mapping(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(
            str(ObjectRemoved('MultiSpectralDistributions.mapping')))

    @property
    def x(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(
            str(ObjectRemoved('MultiSpectralDistributions.x')))

    @property
    def y(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(
            str(ObjectRemoved('MultiSpectralDistributions.y')))

    @property
    def z(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(
            str(ObjectRemoved('MultiSpectralDistributions.z')))

    def __iter__(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(
            str(ObjectRemoved('MultiSpectralDistributions.__iter__')))

    def get(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(
            str(ObjectRemoved('MultiSpectralDistributions.get')))

    def zeros(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        raise AttributeError(
            str(ObjectRemoved('MultiSpectralDistributions.zeros')))

    def trim_wavelengths(self, shape):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        runtime_warning(
            str(
                ObjectRenamed('TriSpectralPowerDistribution.trim_wavelengths',
                              'MultiSpectralDistributions.trim')))

        return self.trim(shape)

    def clone(self):  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        runtime_warning(
            str(
                ObjectRenamed('TriSpectralPowerDistribution.clone',
                              'MultiSpectralDistributions.copy')))

        return self.copy()


def sds_and_multi_sds_to_sds(sds):
    """
    Converts given spectral and multi-spectral distributions to a flat list of
    spectral distributions.

    Parameters
    ----------
    sds : array_like
        Spectral and multi-spectral distributions to convert to a flat list of
        spectral distributions.

    Returns
    -------
    list
        Flat list of spectral distributions.

    Examples
    --------
    >>> data = {
    ...     500: 0.0651,
    ...     520: 0.0705,
    ...     540: 0.0772,
    ...     560: 0.0870,
    ...     580: 0.1128,
    ...     600: 0.1360
    ... }
    >>> sd_1 = SpectralDistribution(data)
    >>> sd_2 = SpectralDistribution(data)
    >>> data = {
    ...     500: (0.004900, 0.323000, 0.272000),
    ...     510: (0.009300, 0.503000, 0.158200),
    ...     520: (0.063270, 0.710000, 0.078250),
    ...     530: (0.165500, 0.862000, 0.042160),
    ...     540: (0.290400, 0.954000, 0.020300),
    ...     550: (0.433450, 0.994950, 0.008750),
    ...     560: (0.594500, 0.995000, 0.003900)
    ... }
    >>> multi_sds_1 = MultiSpectralDistributions(data)
    >>> multi_sds_2 = MultiSpectralDistributions(data)
    >>> len(sds_and_multi_sds_to_sds([sd_1, sd_2, multi_sds_1, multi_sds_2]))
    8
    """

    if not len(sds):
        return

    if isinstance(sds, MultiSpectralDistributions):
        sds = sds.to_sds()
    else:
        sds = list(sds)
        for i, sd in enumerate(sds[:]):
            if isinstance(sd, MultiSpectralDistributions):
                sds.remove(sd)
                sds[i:i] = sd.to_sds()

    return sds


def sds_and_multi_sds_to_multi_sds(sds):
    """
    Converts given spectral and multi-spectral distributions to
    multi-spectral distributions.

    The spectral and multi-spectral distributions will be aligned to the
    intersection of their spectral shapes.

    Parameters
    ----------
    sds : array_like
        Spectral and multi-spectral distributions to convert to
        multi-spectral distributions.

    Returns
    -------
    MultiSpectralDistributions
        Multi-spectral distributions.

    Examples
    --------
    >>> data = {
    ...     500: 0.0651,
    ...     520: 0.0705,
    ...     540: 0.0772,
    ...     560: 0.0870,
    ...     580: 0.1128,
    ...     600: 0.1360
    ... }
    >>> sd_1 = SpectralDistribution(data)
    >>> sd_2 = SpectralDistribution(data)
    >>> data = {
    ...     500: (0.004900, 0.323000, 0.272000),
    ...     510: (0.009300, 0.503000, 0.158200),
    ...     520: (0.063270, 0.710000, 0.078250),
    ...     530: (0.165500, 0.862000, 0.042160),
    ...     540: (0.290400, 0.954000, 0.020300),
    ...     550: (0.433450, 0.994950, 0.008750),
    ...     560: (0.594500, 0.995000, 0.003900)
    ... }
    >>> multi_sds_1 = MultiSpectralDistributions(data)
    >>> multi_sds_2 = MultiSpectralDistributions(data)
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True, linewidth=160):
    ...     sds_and_multi_sds_to_multi_sds(  # doctest: +SKIP
    ...         [sd_1, sd_2, multi_sds_1, multi_sds_2])
    MultiSpectralDistributions([[ 500.        ,    0.0651   ...,\
    0.0651   ...,    0.0049   ...,    0.323    ...,    0.272    ...,\
    0.0049   ...,    0.323    ...,    0.272    ...],
                                [ 510.        ,    0.0676692...,\
    0.0676692...,    0.0093   ...,    0.503    ...,    0.1582   ...,\
    0.0093   ...,    0.503    ...,    0.1582   ...],
                                [ 520.        ,    0.0705   ...,\
    0.0705   ...,    0.06327  ...,    0.71     ...,    0.07825  ...,\
    0.06327  ...,    0.71     ...,    0.07825  ...],
                                [ 530.        ,    0.0737808...,\
    0.0737808...,    0.1655   ...,    0.862    ...,    0.04216  ...,\
    0.1655   ...,    0.862    ...,    0.04216  ...],
                                [ 540.        ,    0.0772   ...,\
    0.0772   ...,    0.2904   ...,    0.954    ...,    0.0203   ...,\
    0.2904   ...,    0.954    ...,    0.0203   ...],
                                [ 550.        ,    0.0806671...,\
    0.0806671...,    0.43345  ...,    0.99495  ...,    0.00875  ...,\
    0.43345  ...,    0.99495  ...,    0.00875  ...],
                                [ 560.        ,    0.087    ...,\
    0.087    ...,    0.5945   ...,    0.995    ...,    0.0039   ...,\
    0.5945   ...,    0.995    ...,    0.0039   ...]],
                               labels=['SpectralDistribution (...)', \
'SpectralDistribution (...)', '0 - SpectralDistribution (...)', \
'1 - SpectralDistribution (...)', '2 - SpectralDistribution (...)', \
'0 - SpectralDistribution (...)', '1 - SpectralDistribution (...)', \
'2 - SpectralDistribution (...)'],
                               interpolator=SpragueInterpolator,
                               interpolator_args={},
                               extrapolator=Extrapolator,
                               extrapolator_args={...})
    """

    if not len(sds):
        return

    if isinstance(sds, MultiSpectralDistributions):
        msds = sds
    elif len(sds) == 1 and isinstance(sds[0], MultiSpectralDistributions):
        msds = sds[0]
    else:
        sds_u = []
        shapes = []
        for sd in sds:
            if isinstance(sd, MultiSpectralDistributions):
                sds_m = sds_and_multi_sds_to_sds(sd)
                sds_u.extend(sds_m)
                shapes.extend([sd_m.shape for sd_m in sds_m])
            else:
                sds_u.append(sd)
                shapes.append(sd.shape)

        shapes = tuple(set(shapes))
        shape = SpectralShape(
            max([shape.start for shape in shapes]),
            min([shape.end for shape in shapes]),
            min([shape.interval for shape in shapes]))

        values = []
        labels = []
        strict_labels = []
        for sd_u in sds_u:
            sd_u.align(shape)
            values.append(sd_u.values)
            labels.append(sd_u.name)
            strict_labels.append(sd_u.strict_name)

        msds = MultiSpectralDistributions(
            tstack(values), shape.range(), labels, strict_labels=strict_labels)

    return msds
