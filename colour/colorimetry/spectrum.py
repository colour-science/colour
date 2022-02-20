"""
Spectrum
========

Defines the classes and objects handling spectral data computations:

-   :class:`colour.SPECTRAL_SHAPE_DEFAULT`
-   :class:`colour.SpectralShape`
-   :class:`colour.SpectralDistribution`
-   :class:`colour.MultiSpectralDistributions`
-   :func:`colour.colorimetry.sds_and_msds_to_sds`
-   :func:`colour.colorimetry.sds_and_msds_to_msds`
-   :func:`colour.colorimetry.reshape_sd`
-   :func:`colour.colorimetry.reshape_msds`

References
----------
-   :cite:`CIETC1-382005e` : CIE TC 1-38. (2005). 9. INTERPOLATION. In CIE
    167:2005 Recommended Practice for Tabulating Spectral Data for Use in
    Colour Computations (pp. 14-19). ISBN:978-3-901906-41-1
-   :cite:`CIETC1-382005g` : CIE TC 1-38. (2005). EXTRAPOLATION. In CIE
    167:2005 Recommended Practice for Tabulating Spectral Data for Use in
    Colour Computations (pp. 19-20). ISBN:978-3-901906-41-1
-   :cite:`CIETC1-482004l` : CIE TC 1-48. (2004). Extrapolation. In CIE
    015:2004 Colorimetry, 3rd Edition (p. 24). ISBN:978-3-901906-33-6
"""

from __future__ import annotations

import numpy as np
from collections.abc import Mapping

from colour.algebra import (
    Extrapolator,
    CubicSplineInterpolator,
    SpragueInterpolator,
)
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.continuous import Signal, MultiSignals
from colour.hints import (
    ArrayLike,
    Any,
    Dict,
    DTypeFloating,
    FloatingOrArrayLike,
    Generator,
    Integer,
    List,
    Literal,
    NDArray,
    Number,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeExtrapolator,
    TypeInterpolator,
    Union,
    cast,
)
from colour.utilities import (
    CACHE_REGISTRY,
    as_float_array,
    as_int,
    attest,
    filter_kwargs,
    first_item,
    is_iterable,
    is_numeric,
    is_pandas_installed,
    is_string,
    is_uniform,
    interval,
    optional,
    runtime_warning,
    tstack,
    validate_method,
)

if is_pandas_installed():
    from pandas import DataFrame, Series
else:  # pragma: no cover
    from unittest import mock

    DataFrame = mock.MagicMock()
    Series = mock.MagicMock()

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SpectralShape",
    "SPECTRAL_SHAPE_DEFAULT",
    "SpectralDistribution",
    "MultiSpectralDistributions",
    "reshape_sd",
    "reshape_msds",
    "sds_and_msds_to_sds",
    "sds_and_msds_to_msds",
]

_CACHE_SHAPE_RANGE: Dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_SHAPE_RANGE"
)


class SpectralShape:
    """
    Define the base object for spectral distribution shape.

    Parameters
    ----------
    start
        Wavelength :math:`\\lambda_{i}` range start in nm.
    end
        Wavelength :math:`\\lambda_{i}` range end in nm.
    interval
        Wavelength :math:`\\lambda_{i}` range interval.

    Attributes
    ----------
    -   :attr:`~colour.SpectralShape.start`
    -   :attr:`~colour.SpectralShape.end`
    -   :attr:`~colour.SpectralShape.interval`
    -   :attr:`~colour.SpectralShape.boundaries`

    Methods
    -------
    -   :meth:`~colour.SpectralShape.__init__`
    -   :meth:`~colour.SpectralShape.__str__`
    -   :meth:`~colour.SpectralShape.__repr__`
    -   :meth:`~colour.SpectralShape.__hash__`
    -   :meth:`~colour.SpectralShape.__iter__`
    -   :meth:`~colour.SpectralShape.__contains__`
    -   :meth:`~colour.SpectralShape.__len__`
    -   :meth:`~colour.SpectralShape.__eq__`
    -   :meth:`~colour.SpectralShape.__ne__`
    -   :meth:`~colour.SpectralShape.range`

    Examples
    --------
    >>> SpectralShape(360, 830, 1)
    SpectralShape(360, 830, 1)
    """

    def __init__(self, start: Number, end: Number, interval: Number):
        self._start: Number = 360
        self._end: Number = 780
        self._interval: Number = 1
        self.start = start
        self.end = end
        self.interval = interval

    @property
    def start(self) -> Number:
        """
        Getter and setter property for the spectral shape start.

        Parameters
        ----------
        value
            Value to set the spectral shape start with.

        Returns
        -------
        Number
            Spectral shape start.
        """

        return self._start

    @start.setter
    def start(self, value: Number):
        """Setter for the **self.start** property."""

        attest(
            is_numeric(value),
            f'"start" property: "{value}" is not a "number"!',
        )

        attest(
            bool(value < self._end),
            f'"start" attribute value must be strictly less than '
            f'"{self._end}"!',
        )

        self._start = value

    @property
    def end(self) -> Number:
        """
        Getter and setter property for the spectral shape end.

        Parameters
        ----------
        value
            Value to set the spectral shape end with.

        Returns
        -------
        Number
            Spectral shape end.
        """

        return self._end

    @end.setter
    def end(self, value: Number):
        """Setter for the **self.end** property."""

        attest(
            is_numeric(value),
            f'"end" property: "{value}" is not a "number"!',
        )

        attest(
            bool(value > self._start),
            f'"end" attribute value must be strictly greater than '
            f'"{self._start}"!',
        )

        self._end = value

    @property
    def interval(self) -> Number:
        """
        Getter and setter property for the spectral shape interval.

        Parameters
        ----------
        value
            Value to set the spectral shape interval with.

        Returns
        -------
        Number
            Spectral shape interval.
        """

        return self._interval

    @interval.setter
    def interval(self, value: Number):
        """Setter for the **self.interval** property."""

        attest(
            is_numeric(value),
            f'"interval" property: "{value}" is not a "number"!',
        )

        self._interval = value

    @property
    def boundaries(self) -> Tuple:
        """
        Getter and setter property for the spectral shape boundaries.

        Parameters
        ----------
        value
            Value to set the spectral shape boundaries with.

        Returns
        -------
        :class:`tuple`
            Spectral shape boundaries.
        """

        return self._start, self._end

    @boundaries.setter
    def boundaries(self, value: ArrayLike):
        """Setter for the **self.boundaries** property."""

        value = np.asarray(value)

        attest(
            value.size == 2,
            f'"boundaries" property: "{value}" must have exactly two '
            f"elements!",
        )

        self.start, self.end = value

    def __str__(self) -> str:
        """
        Return a formatted string representation of the spectral shape.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return f"({self._start}, {self._end}, {self._interval})"

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the spectral shape.

        Returns
        -------
        :class:`str`
            Evaluable string representation.
        """

        return f"SpectralShape({self._start}, {self._end}, {self._interval})"

    def __hash__(self) -> Integer:
        """
        Return the spectral shape hash.

        Returns
        -------
        :class:`numpy.integer`
            Object hash.
        """

        return hash(repr(self))

    def __iter__(self) -> Generator:
        """
        Return a generator for the spectral shape data.

        Yields
        ------
        Generator
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

        yield from self.range()

    def __contains__(self, wavelength: FloatingOrArrayLike) -> bool:
        """
        Return if the spectral shape contains given wavelength
        :math:`\\lambda`.

        Parameters
        ----------
        wavelength
            Wavelength :math:`\\lambda`.

        Returns
        -------
        :class:`bool`
            Whether wavelength :math:`\\lambda` is contained in the spectral
            shape.

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

        decimals = np.finfo(DEFAULT_FLOAT_DTYPE).precision

        return bool(
            np.all(
                np.in1d(
                    np.around(wavelength, decimals),
                    np.around(self.range(), decimals),
                )
            )
        )

    def __len__(self) -> Integer:
        """
        Return the spectral shape wavelength :math:`\\lambda_n` count.

        Returns
        -------
        :class:`numpy.integer`
            Spectral shape wavelength :math:`\\lambda_n` count.

        Examples
        --------
        >>> len(SpectralShape(0, 10, 0.1))
        101
        """

        return len(self.range())

    def __eq__(self, other: Any) -> bool:
        """
        Return whether the spectral shape is equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is equal to the spectral shape.

        Returns
        -------
        :class:`bool`
            Whether given object is equal to the spectral shape.

        Examples
        --------
        >>> SpectralShape(0, 10, 0.1) == SpectralShape(0, 10, 0.1)
        True
        >>> SpectralShape(0, 10, 0.1) == SpectralShape(0, 10, 1)
        False
        """

        if isinstance(other, SpectralShape):
            return np.array_equal(self.range(), other.range())
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        """
        Return whether the spectral shape is not equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is not equal to the spectral shape.

        Returns
        -------
        :class:`bool`
            Whether given object is not equal to the spectral shape.

        Examples
        --------
        >>> SpectralShape(0, 10, 0.1) != SpectralShape(0, 10, 0.1)
        False
        >>> SpectralShape(0, 10, 0.1) != SpectralShape(0, 10, 1)
        True
        """

        return not (self == other)

    def range(self, dtype: Optional[Type[DTypeFloating]] = None) -> NDArray:
        """
        Return an iterable range for the spectral shape.

        Parameters
        ----------
        dtype
            Data type used to generate the range.

        Returns
        -------
        :class:`numpy.ndarray`
            Iterable range for the spectral distribution shape

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

        dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

        hash_key = tuple(hash(arg) for arg in (self, dtype))
        if hash_key in _CACHE_SHAPE_RANGE:
            return _CACHE_SHAPE_RANGE[hash_key].copy()

        start, end, interval = (
            dtype(self._start),
            dtype(self._end),
            dtype(self._interval),
        )

        samples = as_int(round((interval + end - start) / interval))
        range_, interval_effective = np.linspace(
            start, end, samples, retstep=True, dtype=dtype
        )

        _CACHE_SHAPE_RANGE[hash_key] = range_

        if interval_effective != self._interval:
            self._interval = interval_effective
            runtime_warning(
                f'"{(self._start, self._end, self._interval)}" shape could '
                f'not be honoured, using "{self}"!'
            )

        return range_


SPECTRAL_SHAPE_DEFAULT: SpectralShape = SpectralShape(360, 780, 1)
"""Default spectral shape according to *ASTM E308-15* practise shape."""


class SpectralDistribution(Signal):
    """
    Define the spectral distribution: the base object for spectral
    computations.

    The spectral distribution will be initialised according to *CIE 15:2004*
    recommendation: the method developed by *Sprague (1880)* will be used for
    interpolating functions having a uniformly spaced independent variable and
    the *Cubic Spline* method for non-uniformly spaced independent variable.
    Extrapolation is performed according to *CIE 167:2005* recommendation.

    .. important::

        Specific documentation about getting, setting, indexing and slicing the
        spectral power distribution values is available in the
        :ref:`spectral-representation-and-continuous-signal` section.

    Parameters
    ----------
    data
        Data to be stored in the spectral distribution.
    domain
        Values to initialise the
        :attr:`colour.SpectralDistribution.wavelength` property with.
        If both ``data`` and ``domain`` arguments are defined, the latter will
        be used to initialise the
        :attr:`colour.SpectralDistribution.wavelength` property.

    Other Parameters
    ----------------
    extrapolator
        Extrapolator class type to use as extrapolating function.
    extrapolator_kwargs
        Arguments to use when instantiating the extrapolating function.
    interpolator
        Interpolator class type to use as interpolating function.
    interpolator_kwargs
        Arguments to use when instantiating the interpolating function.
    name
        Spectral distribution name.
    strict_name
        Spectral distribution name for figures, default to
        :attr:`colour.SpectralDistribution.name` property value.

    Attributes
    ----------
    -   :attr:`~colour.SpectralDistribution.strict_name`
    -   :attr:`~colour.SpectralDistribution.wavelengths`
    -   :attr:`~colour.SpectralDistribution.values`
    -   :attr:`~colour.SpectralDistribution.shape`

    Methods
    -------
    -   :meth:`~colour.SpectralDistribution.__init__`
    -   :meth:`~colour.SpectralDistribution.interpolate`
    -   :meth:`~colour.SpectralDistribution.extrapolate`
    -   :meth:`~colour.SpectralDistribution.align`
    -   :meth:`~colour.SpectralDistribution.trim`
    -   :meth:`~colour.SpectralDistribution.normalise`

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
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})

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
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})

    Instantiation with a *Pandas* :class:`pandas.Series`:

    >>> from colour.utilities import is_pandas_installed
    >>> if is_pandas_installed():
    ...     from pandas import Series
    ...     print(SpectralDistribution(Series(data)))  # doctest: +SKIP
    [[  5.0000000...e+02   6.5100000...e-02]
     [  5.2000000...e+02   7.0500000...e-02]
     [  5.4000000...e+02   7.7200000...e-02]
     [  5.6000000...e+02   8.7000000...e-02]
     [  5.8000000...e+02   1.1280000...e-01]
     [  6.0000000...e+02   1.3600000...e-01]
     [  5.1000000...e+02   3.1416000...e-01]]
    """

    def __init__(
        self,
        data: Optional[
            Union[ArrayLike, dict, Series, Signal, SpectralDistribution]
        ] = None,
        domain: Optional[Union[ArrayLike, SpectralShape]] = None,
        **kwargs: Any,
    ):
        domain = (
            domain.range() if isinstance(domain, SpectralShape) else domain
        )
        domain_unpacked, range_unpacked = self.signal_unpack_data(data, domain)

        # Initialising with *CIE 15:2004* and *CIE 167:2005* recommendations
        # defaults.
        kwargs["interpolator"] = kwargs.get(
            "interpolator",
            SpragueInterpolator
            if is_uniform(domain_unpacked)
            else CubicSplineInterpolator,
        )
        kwargs["interpolator_kwargs"] = kwargs.get("interpolator_kwargs", {})

        kwargs["extrapolator"] = kwargs.get("extrapolator", Extrapolator)
        kwargs["extrapolator_kwargs"] = kwargs.get(
            "extrapolator_kwargs",
            {"method": "Constant", "left": None, "right": None},
        )

        super().__init__(range_unpacked, domain_unpacked, **kwargs)

        self._strict_name: str = self.name
        self.strict_name = kwargs.get("strict_name", self._strict_name)

    @property
    def strict_name(self) -> str:
        """
        Getter and setter property for the spectral distribution strict name.

        Parameters
        ----------
        value
            Value to set the spectral distribution strict name with.

        Returns
        -------
        :class:`str`
            Spectral distribution strict name.
        """

        return self._strict_name

    @strict_name.setter
    def strict_name(self, value: str):
        """Setter for the **self.strict_name** property."""

        attest(
            is_string(value),
            f'"strict_name" property: "{value}" type is not "str"!',
        )

        self._strict_name = value

    @property
    def wavelengths(self) -> NDArray:
        """
        Getter and setter property for the spectral distribution wavelengths
        :math:`\\lambda_n`.

        Parameters
        ----------
        value
            Value to set the spectral distribution wavelengths
            :math:`\\lambda_n` with.

        Returns
        -------
        :class:`numpy.ndarray`
            Spectral distribution wavelengths :math:`\\lambda_n`.
        """

        return self.domain

    @wavelengths.setter
    def wavelengths(self, value: ArrayLike):
        """Setter for the **self.wavelengths** property."""

        self.domain = as_float_array(value, self.dtype)

    @property
    def values(self) -> NDArray:
        """
        Getter and setter property for the spectral distribution values.

        Parameters
        ----------
        value
            Value to set the spectral distribution wavelengths values with.

        Returns
        -------
        :class:`numpy.ndarray`
            Spectral distribution values.
        """

        return self.range

    @values.setter
    def values(self, value: ArrayLike):
        """Setter for the **self.values** property."""

        self.range = as_float_array(value, self.dtype)

    @property
    def shape(self) -> SpectralShape:
        """
        Getter property for the spectral distribution shape.

        Returns
        -------
        :class:`colour.SpectralShape`
            Spectral distribution shape.

        Notes
        -----
        -   A spectral distribution with a non-uniformly spaced independent
            variable have multiple intervals, in that case
            :attr:`colour.SpectralDistribution.shape` property returns
            the *minimum* interval size.

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
            runtime_warning(
                f'"{self.name}" spectral distribution is not uniform, using '
                f"minimum interval!"
            )

        return SpectralShape(
            min(self.wavelengths),
            max(self.wavelengths),
            min(wavelengths_interval),
        )

    def interpolate(
        self,
        shape: SpectralShape,
        interpolator: Optional[Type[TypeInterpolator]] = None,
        interpolator_kwargs: Optional[Dict] = None,
    ) -> SpectralDistribution:
        """
        Interpolate the spectral distribution in-place according to
        *CIE 167:2005* recommendation (if the interpolator has not been changed
        at instantiation time) or given interpolation arguments.

        The logic for choosing the interpolator class when ``interpolator`` is
        not given is as follows:

        .. code-block:: python

            if self.interpolator not in (SpragueInterpolator,
                                         CubicSplineInterpolator):
                interpolator = self.interpolator
            elif self.is_uniform():
                interpolator = SpragueInterpolator
            else:
                interpolator = CubicSplineInterpolator

        The logic for choosing the interpolator keyword arguments when
        ``interpolator_kwargs`` is not given is as follows:

        .. code-block:: python

            if self.interpolator not in (SpragueInterpolator,
                                         CubicSplineInterpolator):
                interpolator_kwargs = self.interpolator_kwargs
            else:
                interpolator_kwargs = {}

        Parameters
        ----------
        shape
            Spectral shape used for interpolation.
        interpolator
            Interpolator class type to use as interpolating function.
        interpolator_kwargs
            Arguments to use when instantiating the interpolating function.

        Returns
        -------
        :class:`colour.SpectralDistribution`
            Interpolated spectral distribution.

        Notes
        -----
        -   Interpolation will be performed over boundaries range, if you need
            to extend the range of the spectral distribution use the
            :meth:`colour.SpectralDistribution.extrapolate` or
            :meth:`colour.SpectralDistribution.align` methods.

        Warnings
        --------
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
        ...     print(sd.interpolate(SpectralShape(500, 600, 1)))
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

        Spectral distribution with a non-uniformly spaced independent variable
        uses *Cubic Spline* interpolation:

        >>> sd = SpectralDistribution(data)
        >>> sd[510] = np.pi / 10
        >>> with numpy_print_options(suppress=True):
        ...     print(sd.interpolate(SpectralShape(500, 600, 1)))
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

        shape_start, shape_end, shape_interval = as_float_array(
            [
                self.shape.start,
                self.shape.end,
                self.shape.interval,
            ]
        )

        shape = SpectralShape(
            *[
                x[0] if x[0] is not None else x[1]
                for x in zip(
                    (shape.start, shape.end, shape.interval),
                    (shape_start, shape_end, shape_interval),
                )
            ]
        )

        # Defining proper interpolation bounds.
        # TODO: Provide support for fractional interval like 0.1, etc...
        if (
            np.around(shape_start) != shape_start
            or np.around(shape_end) != shape_end
        ):
            runtime_warning(
                "Fractional bound encountered, rounding will occur!"
            )

        shape.start = max([shape.start, np.ceil(shape_start)])
        shape.end = min([shape.end, np.floor(shape_end)])

        if interpolator is None:
            # User has specifically chosen the interpolator thus it is used
            # instead of those from *CIE 167:2005* recommendation.
            if self.interpolator not in (
                SpragueInterpolator,
                CubicSplineInterpolator,
            ):
                interpolator = self.interpolator
            elif self.is_uniform():
                interpolator = SpragueInterpolator
            else:
                interpolator = CubicSplineInterpolator

        if interpolator_kwargs is None:
            # User has specifically chosen the interpolator thus its keyword
            # arguments are used.
            if self.interpolator not in (
                SpragueInterpolator,
                CubicSplineInterpolator,
            ):
                interpolator_kwargs = self.interpolator_kwargs
            else:
                interpolator_kwargs = {}

        wavelengths, values = self.wavelengths, self.values

        self.domain = shape.range()
        self.range = as_float_array(
            interpolator(wavelengths, values, **interpolator_kwargs)(
                self.domain
            )
        )

        return self

    def extrapolate(
        self,
        shape: SpectralShape,
        extrapolator: Optional[Type[TypeExtrapolator]] = None,
        extrapolator_kwargs: Optional[Dict] = None,
    ) -> SpectralDistribution:
        """
        Extrapolate the spectral distribution in-place according to
        *CIE 15:2004* and *CIE 167:2005* recommendations or given extrapolation
        arguments.

        Parameters
        ----------
        shape
            Spectral shape used for extrapolation.
        extrapolator
            Extrapolator class type to use as extrapolating function.
        extrapolator_kwargs
            Arguments to use when instantiating the extrapolating function.

        Returns
        -------
        :class:`colour.SpectralDistribution`
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
        >>> sd.extrapolate(SpectralShape(400, 700, 20)).shape
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

        shape_start, shape_end, shape_interval = as_float_array(
            [
                self.shape.start,
                self.shape.end,
                self.shape.interval,
            ]
        )

        wavelengths = np.hstack(
            [
                np.arange(shape.start, shape_start, shape_interval),
                np.arange(shape_end, shape.end, shape_interval)
                + shape_interval,
            ]
        )

        extrapolator = optional(extrapolator, Extrapolator)
        extrapolator_kwargs = optional(
            extrapolator_kwargs,
            {"method": "Constant", "left": None, "right": None},
        )

        self_extrapolator = self.extrapolator
        self_extrapolator_kwargs = self.extrapolator_kwargs

        self.extrapolator = extrapolator
        self.extrapolator_kwargs = extrapolator_kwargs

        # The following self-assignment is written as intended and triggers the
        # extrapolation.
        self[wavelengths] = self[wavelengths]

        self.extrapolator = self_extrapolator
        self.extrapolator_kwargs = self_extrapolator_kwargs

        return self

    def align(
        self,
        shape: SpectralShape,
        interpolator: Optional[Type[TypeInterpolator]] = None,
        interpolator_kwargs: Optional[Dict] = None,
        extrapolator: Optional[Type[TypeExtrapolator]] = None,
        extrapolator_kwargs: Optional[Dict] = None,
    ) -> SpectralDistribution:
        """
        Align the spectral distribution in-place to given spectral shape:
        Interpolates first then extrapolates to fit the given range.

        Interpolation is performed according to *CIE 167:2005* recommendation
        (if the interpolator has not been changed at instantiation time) or
        given interpolation arguments.

        The logic for choosing the interpolator class when ``interpolator`` is
        not given is as follows:

        .. code-block:: python

            if self.interpolator not in (SpragueInterpolator,
                                         CubicSplineInterpolator):
                interpolator = self.interpolator
            elif self.is_uniform():
                interpolator = SpragueInterpolator
            else:
                interpolator = CubicSplineInterpolator

        The logic for choosing the interpolator keyword arguments when
        ``interpolator_kwargs`` is not given is as follows:

        .. code-block:: python

            if self.interpolator not in (SpragueInterpolator,
                                         CubicSplineInterpolator):
                interpolator_kwargs = self.interpolator_kwargs
            else:
                interpolator_kwargs = {}

        Parameters
        ----------
        shape
            Spectral shape used for alignment.
        interpolator
            Interpolator class type to use as interpolating function.
        interpolator_kwargs
            Arguments to use when instantiating the interpolating function.
        extrapolator
            Extrapolator class type to use as extrapolating function.
        extrapolator_kwargs
            Arguments to use when instantiating the extrapolating function.

        Returns
        -------
        :class:`colour.SpectralDistribution`
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

        self.interpolate(shape, interpolator, interpolator_kwargs)
        self.extrapolate(shape, extrapolator, extrapolator_kwargs)

        return self

    def trim(self, shape: SpectralShape) -> SpectralDistribution:
        """
        Trim the spectral distribution wavelengths to given spectral shape.

        Parameters
        ----------
        shape
            Spectral shape used for trimming.

        Returns
        -------
        :class:`colour.SpectralDistribution`
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
        >>> sd = sd.interpolate(SpectralShape(500, 600, 1))
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

        start = max([shape.start, self.shape.start])
        end = min([shape.end, self.shape.end])

        indexes = np.where(
            np.logical_and(self.domain >= start, self.domain <= end)
        )

        wavelengths = self.wavelengths[indexes]
        values = self.values[indexes]

        self.wavelengths = wavelengths
        self.values = values

        return self

    def normalise(self, factor: Number = 1) -> SpectralDistribution:
        """
        Normalise the spectral distribution using given normalization factor.

        Parameters
        ----------
        factor
            Normalization factor.

        Returns
        -------
        :class:`colour.SpectralDistribution`
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

        self *= 1 / max(self.values) * factor  # type: ignore[misc]

        return self


class MultiSpectralDistributions(MultiSignals):
    """
    Define the multi-spectral distributions: the base object for multi
    spectral computations. It is used to model colour matching functions,
    display primaries, camera sensitivities, etc...

    The multi-spectral distributions will be initialised according to
    *CIE 15:2004* recommendation: the method developed by *Sprague (1880)* will
    be used for interpolating functions having a uniformly spaced independent
    variable and the *Cubic Spline* method for non-uniformly spaced independent
    variable. Extrapolation is performed according to *CIE 167:2005*
    recommendation.

    .. important::

        Specific documentation about getting, setting, indexing and slicing the
        multi-spectral power distributions values is available in the
        :ref:`spectral-representation-and-continuous-signal` section.

    Parameters
    ----------
    data
        Data to be stored in the multi-spectral distributions.
    domain
        Values to initialise the multiple :class:`colour.SpectralDistribution`
        class instances :attr:`colour.continuous.Signal.wavelengths` attribute
        with. If both ``data`` and ``domain`` arguments are defined, the latter
        will be used to initialise the
        :attr:`colour.continuous.Signal.wavelengths` property.
    labels
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.

    Other Parameters
    ----------------
    extrapolator
        Extrapolator class type to use as extrapolating function for the
        :class:`colour.SpectralDistribution` class instances.
    extrapolator_kwargs
        Arguments to use when instantiating the extrapolating function of the
        :class:`colour.SpectralDistribution` class instances.
    interpolator
        Interpolator class type to use as interpolating function for the
        :class:`colour.SpectralDistribution` class instances.
    interpolator_kwargs
        Arguments to use when instantiating the interpolating function of the
        :class:`colour.SpectralDistribution` class instances.
    name
       Multi-spectral distributions name.
    strict_labels
        Multi-spectral distributions labels for figures, default to
        :attr:`colour.MultiSpectralDistributions.labels` property value.

    Attributes
    ----------
    -   :attr:`~colour.MultiSpectralDistributions.strict_name`
    -   :attr:`~colour.MultiSpectralDistributions.strict_labels`
    -   :attr:`~colour.MultiSpectralDistributions.wavelengths`
    -   :attr:`~colour.MultiSpectralDistributions.values`
    -   :attr:`~colour.MultiSpectralDistributions.shape`

    Methods
    -------
    -   :meth:`~colour.MultiSpectralDistributions.__init__`
    -   :meth:`~colour.MultiSpectralDistributions.interpolate`
    -   :meth:`~colour.MultiSpectralDistributions.extrapolate`
    -   :meth:`~colour.MultiSpectralDistributions.align`
    -   :meth:`~colour.MultiSpectralDistributions.trim`
    -   :meth:`~colour.MultiSpectralDistributions.normalise`
    -   :meth:`~colour.MultiSpectralDistributions.to_sds`

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
                 ... interpolator_kwargs={},
                 ... extrapolator=Extrapolator,
                 ... extrapolator_kwargs={...})

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
                 ... interpolator_kwargs={},
                 ... extrapolator=Extrapolator,
                 ... extrapolator_kwargs={...})

    Instantiation with a *Pandas* `DataFrame`:

    >>> from colour.utilities import is_pandas_installed
    >>> if is_pandas_installed():
    ...     from pandas import DataFrame
    ...     x_bar = [data[key][0] for key in sorted(data.keys())]
    ...     y_bar = [data[key][1] for key in sorted(data.keys())]
    ...     z_bar = [data[key][2] for key in sorted(data.keys())]
    ...     print(MultiSignals(  # doctest: +SKIP
    ...         DataFrame(
    ...             dict(zip(labels, [x_bar, y_bar, z_bar])), data.keys())))
    [[  5.0000000...e+02   4.9000000...e-03   3.2300000...e-01   \
2.7200000...e-01]
     [  5.1000000...e+02   9.3000000...e-03   5.0300000...e-01   \
1.5820000...e-01]
     [  5.2000000...e+02   3.1400000...e-03   3.1416000...e-01   \
3.1420000...e-02]
     [  5.3000000...e+02   6.3270000...e-02   7.1000000...e-01   \
7.8250000...e-02]
     [  5.4000000...e+02   1.6550000...e-01   8.6200000...e-01   \
4.2160000...e-02]
     [  5.5000000...e+02   2.9040000...e-01   9.5400000...e-01   \
2.0300000...e-02]
     [  5.6000000...e+02   4.3345000...e-01   9.9495000...e-01   \
8.7500000...e-03]
     [  5.1100000...e+02   5.9450000...e-01   9.9500000...e-01   \
3.9000000...e-03]]
    """

    def __init__(
        self,
        data: Optional[
            Union[
                ArrayLike,
                DataFrame,
                dict,
                MultiSignals,
                MultiSpectralDistributions,
                Sequence,
                Series,
                Signal,
                SpectralDistribution,
            ]
        ] = None,
        domain: Optional[Union[ArrayLike, SpectralShape]] = None,
        labels: Optional[Sequence] = None,
        **kwargs: Any,
    ):
        domain = (
            domain.range() if isinstance(domain, SpectralShape) else domain
        )
        signals = self.multi_signals_unpack_data(data, domain, labels)

        domain = signals[list(signals.keys())[0]].domain if signals else None
        uniform = is_uniform(domain) if domain is not None else True

        # Initialising with *CIE 15:2004* and *CIE 167:2005* recommendations
        # defaults.
        kwargs["interpolator"] = kwargs.get(
            "interpolator",
            SpragueInterpolator if uniform else CubicSplineInterpolator,
        )
        kwargs["interpolator_kwargs"] = kwargs.get("interpolator_kwargs", {})

        kwargs["extrapolator"] = kwargs.get("extrapolator", Extrapolator)
        kwargs["extrapolator_kwargs"] = kwargs.get(
            "extrapolator_kwargs",
            {"method": "Constant", "left": None, "right": None},
        )

        super().__init__(
            signals, domain, signal_type=SpectralDistribution, **kwargs
        )

        self._strict_name: str = self.name
        self.strict_name = kwargs.get("strict_name", self._strict_name)
        self._strict_labels: List = list(self.signals.keys())
        self.strict_labels = kwargs.get("strict_labels", self._strict_labels)

    @property
    def strict_name(self) -> str:
        """
        Getter and setter property for the multi-spectral distributions strict
        name.

        Parameters
        ----------
        value
            Value to set the multi-spectral distributions strict name with.

        Returns
        -------
        :class:`str`
            Multi-spectral distributions strict name.
        """

        return self._strict_name

    @strict_name.setter
    def strict_name(self, value: str):
        """Setter for the **self.strict_name** property."""

        attest(
            is_string(value),
            f'"strict_name" property: "{value}" type is not "str"!',
        )

        self._strict_name = value

    @property
    def strict_labels(self) -> List[str]:
        """
        Getter and setter property for the multi-spectral distributions strict
        labels.

        Parameters
        ----------
        value
            Value to set the multi-spectral distributions strict labels with.

        Returns
        -------
        :class:`list`
            Multi-spectral distributions strict labels.
        """

        return self._strict_labels

    @strict_labels.setter
    def strict_labels(self, value: Sequence):
        """Setter for the **self.strict_labels** property."""

        attest(
            is_iterable(value),
            f'"strict_labels" property: "{value}" is not an "iterable" like object!',
        )

        attest(
            len(set(value)) == len(value),
            '"strict_labels" property: values must be unique!',
        )

        attest(
            len(value) == len(self.labels),
            f'"strict_labels" property: length must be "{len(self.labels)}"!',
        )

        self._strict_labels = [str(label) for label in value]
        for i, signal in enumerate(self.signals.values()):
            cast(
                SpectralDistribution, signal
            ).strict_name = self._strict_labels[i]

    @property
    def wavelengths(self) -> NDArray:
        """
        Getter and setter property for the multi-spectral distributions
        wavelengths :math:`\\lambda_n`.

        Parameters
        ----------
        value
            Value to set the multi-spectral distributions wavelengths
            :math:`\\lambda_n` with.

        Returns
        -------
        :class:`numpy.ndarray`
            Multi-spectral distributions wavelengths :math:`\\lambda_n`.
        """

        return self.domain

    @wavelengths.setter
    def wavelengths(self, value: ArrayLike):
        """Setter for the **self.wavelengths** property."""

        self.domain = as_float_array(value, self.dtype)

    @property
    def values(self) -> NDArray:
        """
        Getter and setter property for the multi-spectral distributions values.

        Parameters
        ----------
        value
            Value to set the multi-spectral distributions wavelengths values
            with.

        Returns
        -------
        :class:`numpy.ndarray`
            Multi-spectral distributions values.
        """

        return self.range

    @values.setter
    def values(self, value: ArrayLike):
        """Setter for the **self.values** property."""

        self.range = as_float_array(value, self.dtype)

    @property
    def shape(self) -> SpectralShape:
        """
        Getter property for the multi-spectral distributions shape.

        Returns
        -------
        :class:`colour.SpectralShape`
            Multi-spectral distributions shape.

        Notes
        -----
        -   Multi-spectral distributions with a non-uniformly spaced
            independent variable have multiple intervals, in that case
            :attr:`colour.MultiSpectralDistributions.shape` property returns
            the *minimum* interval size.

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

        return first_item(self._signals.values()).shape

    def interpolate(
        self,
        shape: SpectralShape,
        interpolator: Optional[Type[TypeInterpolator]] = None,
        interpolator_kwargs: Optional[Dict] = None,
    ) -> MultiSpectralDistributions:
        """
        Interpolate the multi-spectral distributions in-place according to
        *CIE 167:2005* recommendation (if the interpolator has not been changed
        at instantiation time) or given interpolation arguments.

        The logic for choosing the interpolator class when ``interpolator`` is
        not given is as follows:

        .. code-block:: python

            if self.interpolator not in (SpragueInterpolator,
                                         CubicSplineInterpolator):
                interpolator = self.interpolator
            elif self.is_uniform():
                interpolator = SpragueInterpolator
            else:
                interpolator = CubicSplineInterpolator

        The logic for choosing the interpolator keyword arguments when
        ``interpolator_kwargs`` is not given is as follows:

        .. code-block:: python

            if self.interpolator not in (SpragueInterpolator,
                                         CubicSplineInterpolator):
                interpolator_kwargs = self.interpolator_kwargs
            else:
                interpolator_kwargs = {}

        Parameters
        ----------
        shape
            Spectral shape used for interpolation.
        interpolator
            Interpolator class type to use as interpolating function.
        interpolator_kwargs
            Arguments to use when instantiating the interpolating function.

        Returns
        -------
        :class:`colour.MultiSpectralDistributions`
            Interpolated multi-spectral distributions.

        Notes
        -----
        -   See :meth:`colour.SpectralDistribution.interpolate` method notes
            section.

        Warnings
        --------
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
        ...     print(msds.interpolate(SpectralShape(500, 560, 1)))
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
        ...     print(msds.interpolate(SpectralShape(500, 560, 1)))
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
            cast(SpectralDistribution, signal).interpolate(
                shape, interpolator, interpolator_kwargs
            )

        return self

    def extrapolate(
        self,
        shape: SpectralShape,
        extrapolator: Optional[Type[TypeExtrapolator]] = None,
        extrapolator_kwargs: Optional[Dict] = None,
    ) -> MultiSpectralDistributions:
        """
        Extrapolate the multi-spectral distributions in-place according to
        *CIE 15:2004* and *CIE 167:2005* recommendations or given extrapolation
        arguments.

        Parameters
        ----------
        shape
            Spectral shape used for extrapolation.
        extrapolator
            Extrapolator class type to use as extrapolating function.
        extrapolator_kwargs
            Arguments to use when instantiating the extrapolating function.

        Returns
        -------
        :class:`colour.MultiSpectralDistributions`
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
        >>> msds.extrapolate(SpectralShape(400, 700, 10)).shape
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
            cast(SpectralDistribution, signal).extrapolate(
                shape, extrapolator, extrapolator_kwargs
            )

        return self

    def align(
        self,
        shape: SpectralShape,
        interpolator: Optional[Type[TypeInterpolator]] = None,
        interpolator_kwargs: Optional[Dict] = None,
        extrapolator: Optional[Type[TypeExtrapolator]] = None,
        extrapolator_kwargs: Optional[Dict] = None,
    ) -> MultiSpectralDistributions:
        """
        Align the multi-spectral distributions in-place to given spectral
        shape: Interpolates first then extrapolates to fit the given range.

        Interpolation is performed according to *CIE 167:2005* recommendation
        (if the interpolator has not been changed at instantiation time) or
        given interpolation arguments.

        The logic for choosing the interpolator class when ``interpolator`` is
        not given is as follows:

        .. code-block:: python

            if self.interpolator not in (SpragueInterpolator,
                                         CubicSplineInterpolator):
                interpolator = self.interpolator
            elif self.is_uniform():
                interpolator = SpragueInterpolator
            else:
                interpolator = CubicSplineInterpolator

        The logic for choosing the interpolator keyword arguments when
        ``interpolator_kwargs`` is not given is as follows:

        .. code-block:: python

            if self.interpolator not in (SpragueInterpolator,
                                         CubicSplineInterpolator):
                interpolator_kwargs = self.interpolator_kwargs
            else:
                interpolator_kwargs = {}

        Parameters
        ----------
        shape
            Spectral shape used for alignment.
        interpolator
            Interpolator class type to use as interpolating function.
        interpolator_kwargs
            Arguments to use when instantiating the interpolating function.
        extrapolator
            Extrapolator class type to use as extrapolating function.
        extrapolator_kwargs
            Arguments to use when instantiating the extrapolating function.

        Returns
        -------
        :class:`colour.MultiSpectralDistributions`
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
            cast(SpectralDistribution, signal).align(
                shape,
                interpolator,
                interpolator_kwargs,
                extrapolator,
                extrapolator_kwargs,
            )

        return self

    def trim(self, shape: SpectralShape) -> MultiSpectralDistributions:
        """
        Trim the multi-spectral distributions wavelengths to given shape.

        Parameters
        ----------
        shape
            Spectral shape used for trimming.

        Returns
        -------
        :class:`colour.MultiSpectralDistributions`
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
        >>> msds = msds.interpolate(SpectralShape(500, 560, 1))
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
            cast(SpectralDistribution, signal).trim(shape)

        return self

    def normalise(self, factor: Number = 1) -> MultiSpectralDistributions:
        """
        Normalise the multi-spectral distributions with given normalization
        factor.

        Parameters
        ----------
        factor
            Normalization factor.

        Returns
        -------
        :class:`colour.MultiSpectralDistributions`
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
            cast(SpectralDistribution, signal).normalise(factor)

        return self

    def to_sds(self) -> List[SpectralDistribution]:
        """
        Convert the multi-spectral distributions to a list of spectral
        distributions.

        Returns
        -------
        :class:`list`
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

        return [
            cast(SpectralDistribution, signal.copy())
            for signal in self.signals.values()
        ]


_CACHE_RESHAPED_SDS_AND_MSDS: Dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_RESHAPED_SDS_AND_MSDS"
)


def reshape_sd(
    sd: SpectralDistribution,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    method: Union[
        Literal["Align", "Extrapolate", "Interpolate", "Trim"], str
    ] = "Align",
    **kwargs: Any,
) -> SpectralDistribution:
    """
    Reshape given spectral distribution with given spectral shape.

    The reshaped object is cached, thus another call to the definition with the
    same arguments will yield the cached object immediately.

    Parameters
    ----------
    sd
        Spectral distribution to reshape.
    shape
        Spectral shape to reshape the spectral distribution with.
    method
        Reshape method.

    Other Parameters
    ----------------
    kwargs
        {:meth:`colour.SpectralDistribution.align`,
        :meth:`colour.SpectralDistribution.extrapolate`,
        :meth:`colour.SpectralDistribution.interpolate`,
        :meth:`colour.SpectralDistribution.trim`},
        See the documentation of the previously listed methods.

    Returns
    -------
    :class:`colour.SpectralDistribution`

    Warnings
    --------
    Contrary to *Numpy*, reshaping a spectral distribution alters its data!
    """

    method = validate_method(
        method, valid_methods=["Align", "Extrapolate", "Interpolate", "Trim"]
    )

    # Handling dict-like keyword arguments.
    kwargs_items = list(kwargs.items())
    for i, (keyword, value) in enumerate(kwargs_items):
        if isinstance(value, Mapping):
            kwargs_items[i] = (keyword, tuple(value.items()))

    hash_key = tuple(
        hash(arg) for arg in (sd, shape, method, tuple(kwargs_items))
    )
    if hash_key in _CACHE_RESHAPED_SDS_AND_MSDS:
        return _CACHE_RESHAPED_SDS_AND_MSDS[hash_key].copy()

    function = getattr(sd, method)

    reshaped_sd = getattr(sd.copy(), method)(
        shape, **filter_kwargs(function, **kwargs)
    )

    _CACHE_RESHAPED_SDS_AND_MSDS[hash_key] = reshaped_sd

    return reshaped_sd


def reshape_msds(
    msds: MultiSpectralDistributions,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    method: Union[
        Literal["Align", "Extrapolate", "Interpolate", "Trim"], str
    ] = "Align",
    **kwargs: Any,
) -> MultiSpectralDistributions:
    """
    Reshape given multi-spectral distributions with given spectral shape.

    The reshaped object is cached, thus another call to the definition with the
    same arguments will yield the cached object immediately.

    Parameters
    ----------
    msds
        Spectral distribution to reshape.
    shape
        Spectral shape to reshape the multi-spectral distributions with.
    method
        Reshape method.

    Other Parameters
    ----------------
    kwargs
        {:meth:`colour.MultiSpectralDistributions.align`,
        :meth:`colour.MultiSpectralDistributions.extrapolate`,
        :meth:`colour.MultiSpectralDistributions.interpolate`,
        :meth:`colour.MultiSpectralDistributions.trim`},
        See the documentation of the previously listed methods.

    Returns
    -------
    :class:`colour.MultiSpectralDistributions`

    Warnings
    --------
    Contrary to *Numpy*, reshaping a multi-spectral distributions alters its
    data!
    """

    return reshape_sd(
        msds, shape, method, **kwargs  # type: ignore[arg-type]
    )  # type: ignore[return-value]


def sds_and_msds_to_sds(
    sds: Union[
        Sequence[Union[SpectralDistribution, MultiSpectralDistributions]],
        MultiSpectralDistributions,
    ]
) -> List[SpectralDistribution]:
    """
    Convert given spectral and multi-spectral distributions to a list of
    spectral distributions.

    Parameters
    ----------
    sds
        Spectral and multi-spectral distributions to convert to a list of
        spectral distributions.

    Returns
    -------
    :class:`list`
        List of spectral distributions.

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
    >>> len(sds_and_msds_to_sds([sd_1, sd_2, multi_sds_1, multi_sds_2]))
    8
    """

    if isinstance(sds, MultiSpectralDistributions):
        sds_converted = sds.to_sds()
    else:
        sds_converted = []
        for sd in sds:
            sds_converted += (
                sd.to_sds()
                if isinstance(sd, MultiSpectralDistributions)
                else [sd]
            )

    return sds_converted


def sds_and_msds_to_msds(
    sds: Union[
        Sequence[Union[SpectralDistribution, MultiSpectralDistributions]],
        MultiSpectralDistributions,
    ]
) -> MultiSpectralDistributions:
    """
    Convert given spectral and multi-spectral distributions to
    multi-spectral distributions.

    The spectral and multi-spectral distributions will be aligned to the
    intersection of their spectral shapes.

    Parameters
    ----------
    sds
        Spectral and multi-spectral distributions to convert to
        multi-spectral distributions.

    Returns
    -------
    :class:`colour.MultiSpectralDistributions`
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
    ...     sds_and_msds_to_msds(  # doctest: +SKIP
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
                               interpolator_kwargs={},
                               extrapolator=Extrapolator,
                               extrapolator_kwargs={...})
    """

    if isinstance(sds, MultiSpectralDistributions):
        msds_converted = sds
    else:
        sds_converted = sds_and_msds_to_sds(sds)

        shapes = tuple({sd.shape for sd in sds_converted})
        shape = SpectralShape(
            max(shape.start for shape in shapes),
            min(shape.end for shape in shapes),
            min(shape.interval for shape in shapes),
        )

        values = []
        labels = []
        strict_labels = []
        for sd in sds_converted:
            if sd.shape != shape:
                sd = sd.align(shape)

            values.append(sd.values)
            labels.append(
                sd.name if sd.name not in labels else f"{sd.name} ({id(sd)})"
            )
            strict_labels.append(
                sd.strict_name
                if sd.strict_name not in strict_labels
                else f"{sd.strict_name} ({id(sd)})"
            )

        msds_converted = MultiSpectralDistributions(
            tstack(values), shape.range(), labels, strict_labels=strict_labels
        )

    return msds_converted
