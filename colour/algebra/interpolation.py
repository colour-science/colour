"""
Interpolation
=============

Defines the classes and definitions for interpolating variables.

-   :class:`colour.KernelInterpolator`: 1-D function generic interpolation with
    arbitrary kernel.
-   :class:`colour.NearestNeighbourInterpolator`: 1-D function
    nearest-neighbour interpolation.
-   :class:`colour.LinearInterpolator`: 1-D function linear interpolation.
-   :class:`colour.SpragueInterpolator`: 1-D function fifth-order polynomial
    interpolation using *Sprague (1880)* method.
-   :class:`colour.CubicSplineInterpolator`: 1-D function cubic spline
    interpolation.
-   :class:`colour.PchipInterpolator`: 1-D function piecewise cube Hermite
    interpolation.
-   :class:`colour.NullInterpolator`: 1-D function null interpolation.
-   :func:`colour.lagrange_coefficients`: Computation of
    *Lagrange Coefficients*.
-   :func:`colour.algebra.table_interpolation_trilinear`: Trilinear
    interpolation with table.
-   :func:`colour.algebra.table_interpolation_tetrahedral`: Tetrahedral
    interpolation with table.
-   :attr:`colour.TABLE_INTERPOLATION_METHODS`: Supported table interpolation
    methods.
-   :func:`colour.table_interpolation`: Interpolation with table using given
    method.

References
----------
-   :cite:`Bourkeb` : Bourke, P. (n.d.). Trilinear Interpolation. Retrieved
    January 13, 2018, from http://paulbourke.net/miscellaneous/interpolation/
-   :cite:`Burger2009b` : Burger, W., & Burge, M. J. (2009). Principles of
    Digital Image Processing. Springer London. doi:10.1007/978-1-84800-195-4
-   :cite:`CIETC1-382005f` : CIE TC 1-38. (2005). 9.2.4 Method of
    interpolation for uniformly spaced independent variable. In CIE 167:2005
    Recommended Practice for Tabulating Spectral Data for Use in Colour
    Computations (pp. 1-27). ISBN:978-3-901906-41-1
-   :cite:`CIETC1-382005h` : CIE TC 1-38. (2005). Table V. Values of the
    c-coefficients of Equ.s 6 and 7. In CIE 167:2005 Recommended Practice for
    Tabulating Spectral Data for Use in Colour Computations (p. 19).
    ISBN:978-3-901906-41-1
-   :cite:`Fairman1985b` : Fairman, H. S. (1985). The calculation of weight
    factors for tristimulus integration. Color Research & Application, 10(4),
    199-203. doi:10.1002/col.5080100407
-   :cite:`Kirk2006` : Kirk, R. (2006). Truelight Software Library 2.0.
    Retrieved July 8, 2017, from
    https://www.filmlight.ltd.uk/pdf/whitepapers/FL-TL-TN-0057-SoftwareLib.pdf
-   :cite:`Westland2012h` : Westland, S., Ripamonti, C., & Cheung, V. (2012).
    Interpolation Methods. In Computational Colour Science Using MATLAB (2nd
    ed., pp. 29-37). ISBN:978-0-470-66569-5
-   :cite:`Wikipedia2003a` : Wikipedia. (2003). Lagrange polynomial -
    Definition. Retrieved January 20, 2016, from
    https://en.wikipedia.org/wiki/Lagrange_polynomial#Definition
-   :cite:`Wikipedia2005b` : Wikipedia. (2005). Lanczos resampling. Retrieved
    October 14, 2017, from https://en.wikipedia.org/wiki/Lanczos_resampling
"""

from __future__ import annotations

import itertools
import numpy as np
import scipy.interpolate
from collections.abc import Mapping
from functools import reduce

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.hints import (
    Any,
    ArrayLike,
    Callable,
    Dict,
    DTypeNumber,
    Floating,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Integer,
    Literal,
    NDArray,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    as_array,
    as_float_array,
    as_float,
    as_int_array,
    attest,
    closest_indexes,
    interval,
    is_numeric,
    optional,
    runtime_warning,
    tsplit,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "kernel_nearest_neighbour",
    "kernel_linear",
    "kernel_sinc",
    "kernel_lanczos",
    "kernel_cardinal_spline",
    "KernelInterpolator",
    "NearestNeighbourInterpolator",
    "LinearInterpolator",
    "SpragueInterpolator",
    "CubicSplineInterpolator",
    "PchipInterpolator",
    "NullInterpolator",
    "lagrange_coefficients",
    "vertices_and_relative_coordinates",
    "table_interpolation_trilinear",
    "table_interpolation_tetrahedral",
    "TABLE_INTERPOLATION_METHODS",
    "table_interpolation",
]


def kernel_nearest_neighbour(x: ArrayLike) -> NDArray:
    """
    Return the *nearest-neighbour* kernel evaluated at given samples.

    Parameters
    ----------
    x
        Samples at which to evaluate the *nearest-neighbour* kernel.

    Returns
    -------
    :class:`numpy.ndarray`
        The *nearest-neighbour* kernel evaluated at given samples.

    References
    ----------
    :cite:`Burger2009b`

    Examples
    --------
    >>> kernel_nearest_neighbour(np.linspace(0, 1, 10))
    array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    """

    return np.where(np.abs(x) < 0.5, 1, 0)


def kernel_linear(x: ArrayLike) -> NDArray:
    """
    Return the *linear* kernel evaluated at given samples.

    Parameters
    ----------
    x
        Samples at which to evaluate the *linear* kernel.

    Returns
    -------
    :class:`numpy.ndarray`
        The *linear* kernel evaluated at given samples.

    References
    ----------
    :cite:`Burger2009b`

    Examples
    --------
    >>> kernel_linear(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([ 1.        ,  0.8888888...,  0.7777777...,  \
0.6666666...,  0.5555555...,
            0.4444444...,  0.3333333...,  0.2222222...,  \
0.1111111...,  0.        ])
    """

    return np.where(np.abs(x) < 1, 1 - np.abs(x), 0)


def kernel_sinc(x: ArrayLike, a: Floating = 3) -> NDArray:
    """
    Return the *sinc* kernel evaluated at given samples.

    Parameters
    ----------
    x
        Samples at which to evaluate the *sinc* kernel.
    a
        Size of the *sinc* kernel.

    Returns
    -------
    :class:`numpy.ndarray`
        The *sinc* kernel evaluated at given samples.

    References
    ----------
    :cite:`Burger2009b`

    Examples
    --------
    >>> kernel_sinc(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([  1.0000000...e+00,   9.7981553...e-01,   9.2072542...e-01,
             8.2699334...e-01,   7.0531659...e-01,   5.6425327...e-01,
             4.1349667...e-01,   2.6306440...e-01,   1.2247694...e-01,
             3.8981718...e-17])
    """

    x = as_float_array(x)

    attest(bool(a >= 1), '"a" must be equal or superior to 1!')

    return np.where(np.abs(x) < a, np.sinc(x), 0)


def kernel_lanczos(x: ArrayLike, a: Floating = 3) -> NDArray:
    """
    Return the *lanczos* kernel evaluated at given samples.

    Parameters
    ----------
    x
        Samples at which to evaluate the *lanczos* kernel.
    a
        Size of the *lanczos* kernel.

    Returns
    -------
    :class:`numpy.ndarray`
        The *lanczos* kernel evaluated at given samples.

    References
    ----------
    :cite:`Wikipedia2005b`

    Examples
    --------
    >>> kernel_lanczos(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([  1.0000000...e+00,   9.7760615...e-01,   9.1243770...e-01,
             8.1030092...e-01,   6.8012706...e-01,   5.3295773...e-01,
             3.8071690...e-01,   2.3492839...e-01,   1.0554054...e-01,
             3.2237621...e-17])
    """

    x = as_float_array(x)

    attest(bool(a >= 1), '"a" must be equal or superior to 1!')

    return np.where(np.abs(x) < a, np.sinc(x) * np.sinc(x / a), 0)


def kernel_cardinal_spline(
    x: ArrayLike, a: Floating = 0.5, b: Floating = 0.0
) -> NDArray:
    """
    Return the *cardinal spline* kernel evaluated at given samples.

    Notable *cardinal spline* :math:`a` and :math:`b` parameterizations:

    -   *Catmull-Rom*: :math:`(a=0.5, b=0)`
    -   *Cubic B-Spline*: :math:`(a=0, b=1)`
    -   *Mitchell-Netravalli*: :math:`(a=\\cfrac{1}{3}, b=\\cfrac{1}{3})`

    Parameters
    ----------
    x
        Samples at which to evaluate the *cardinal spline* kernel.
    a
        :math:`a` control parameter.
    b
        :math:`b` control parameter.

    Returns
    -------
    :class:`numpy.ndarray`
        The *cardinal spline* kernel evaluated at given samples.

    References
    ----------
    :cite:`Burger2009b`

    Examples
    --------
    >>> kernel_cardinal_spline(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([ 1.        ,  0.9711934...,  0.8930041...,  \
0.7777777...,  0.6378600...,
            0.4855967...,  0.3333333...,  0.1934156...,  \
0.0781893...,  0.        ])
    """

    x = as_float_array(x)

    x_abs = np.abs(x)
    y = np.where(
        x_abs < 1,
        (-6 * a - 9 * b + 12) * x_abs**3
        + (6 * a + 12 * b - 18) * x_abs**2
        - 2 * b
        + 6,
        (-6 * a - b) * x_abs**3
        + (30 * a + 6 * b) * x_abs**2
        + (-48 * a - 12 * b) * x_abs
        + 24 * a
        + 8 * b,
    )
    y[x_abs >= 2] = 0

    return 1 / 6 * y


class KernelInterpolator:
    """
    Kernel based interpolation of a 1-D function.

    The reconstruction of a continuous signal can be described as a linear
    convolution operation. Interpolation can be expressed as a convolution of
    the given discrete function :math:`g(x)` with some continuous interpolation
    kernel :math:`k(w)`::

        :math:`\\hat{g}(w_0) = [k * g](w_0) = \
\\sum_{x=-\\infty}^{\\infty}k(w_0 - x)\\cdot g(x)`

    Parameters
    ----------
    x
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y
        Dependent and already known :math:`y` variable values to
        interpolate.
    window
        Width of the window in samples on each side.
    kernel
        Kernel to use for interpolation.
    kernel_kwargs
         Arguments to use when calling the kernel.
    padding_kwargs
         Arguments to use when padding :math:`y` variable values with the
         :func:`np.pad` definition.
    dtype
        Data type used for internal conversions.

    Attributes
    ----------
    -   :attr:`~colour.KernelInterpolator.x`
    -   :attr:`~colour.KernelInterpolator.y`
    -   :attr:`~colour.KernelInterpolator.window`
    -   :attr:`~colour.KernelInterpolator.kernel`
    -   :attr:`~colour.KernelInterpolator.kernel_kwargs`
    -   :attr:`~colour.KernelInterpolator.padding_kwargs`

    Methods
    -------
    -   :meth:`~colour.KernelInterpolator.__init__`
    -   :meth:`~colour.KernelInterpolator.__call__`

    References
    ----------
    :cite:`Burger2009b`, :cite:`Wikipedia2005b`

    Examples
    --------
    Interpolating a single numeric variable:

    >>> y = np.array([5.9200, 9.3700, 10.8135, 4.5100,
    ...               69.5900, 27.8007, 86.0500])
    >>> x = np.arange(len(y))
    >>> f = KernelInterpolator(x, y)
    >>> f(0.5)  # doctest: +ELLIPSIS
    6.9411400...

    Interpolating an `ArrayLike` variable:

    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([ 6.1806208...,  8.0823848...])

    Using a different *lanczos* kernel:

    >>> f = KernelInterpolator(x, y, kernel=kernel_sinc)
    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([ 6.5147317...,  8.3965466...])

    Using a different window size:

    >>> f = KernelInterpolator(
    ...     x,
    ...     y,
    ...     window=16,
    ...     kernel=kernel_lanczos,
    ...     kernel_kwargs={'a': 16})
    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([ 5.3961792...,  5.6521093...])
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        window: Floating = 3,
        kernel: Callable = kernel_lanczos,
        kernel_kwargs: Optional[Dict] = None,
        padding_kwargs: Optional[Dict] = None,
        dtype: Optional[Type[DTypeNumber]] = None,
    ):
        dtype = cast(Type[DTypeNumber], optional(dtype, DEFAULT_FLOAT_DTYPE))

        self._x_p: NDArray = np.array([])
        self._y_p: NDArray = np.array([])

        self._x: NDArray = np.array([])
        self._y: NDArray = np.array([])
        self._window: Floating = 3
        self._padding_kwargs: Dict = {
            "pad_width": (window, window),
            "mode": "reflect",
        }
        self._kernel: Callable = kernel_lanczos
        self._kernel_kwargs: Dict = {}
        self._dtype: Type[DTypeNumber] = dtype

        # TODO: Remove pragma when https://github.com/python/mypy/issues/3004
        # is resolved.
        self.x = x  # type: ignore[assignment]
        self.y = y  # type: ignore[assignment]
        self.window = window
        self.padding_kwargs = optional(padding_kwargs, self._padding_kwargs)
        self.kernel = kernel
        self.kernel_kwargs = optional(kernel_kwargs, self._kernel_kwargs)

        self._validate_dimensions()

    @property
    def x(self) -> NDArray:
        """
        Getter and setter property for the independent :math:`x` variable.

        Parameters
        ----------
        value
            Value to set the independent :math:`x` variable with.

        Returns
        -------
        :class:`numpy.ndarray`
            Independent :math:`x` variable.
        """

        return self._x

    @x.setter
    def x(self, value: ArrayLike):
        """Setter for the **self.x** property."""

        value = np.atleast_1d(value).astype(self._dtype)

        attest(
            value.ndim == 1,
            '"x" independent variable must have exactly one dimension!',
        )

        value_interval = interval(value)

        if value_interval.size != 1:
            runtime_warning(
                '"x" independent variable is not uniform, '
                "unpredictable results may occur!"
            )

        self._x = as_array(value, self._dtype)

        self._x_p = np.pad(
            self._x,
            as_int_array([self._window, self._window]),
            "linear_ramp",
            end_values=(
                np.min(self._x) - self._window * value_interval[0],
                np.max(self._x) + self._window * value_interval[0],
            ),
        )

    @property
    def y(self) -> NDArray:
        """
        Getter and setter property for the dependent and already known
        :math:`y` variable.

        Parameters
        ----------
        value
            Value to set the dependent and already known :math:`y` variable
            with.

        Returns
        -------
        :class:`numpy.ndarray`
            Dependent and already known :math:`y` variable.
        """

        return self._y

    @y.setter
    def y(self, value: ArrayLike):
        """Setter for the **self.y** property."""

        value = np.atleast_1d(value).astype(self._dtype)

        attest(
            value.ndim == 1,
            '"y" dependent variable must have exactly one dimension!',
        )

        self._y = as_array(value, self._dtype)

        if self._window is not None:
            self._y_p = np.pad(self._y, **self._padding_kwargs)

    @property
    def window(self) -> Floating:
        """
        Getter and setter property for the window.

        Parameters
        ----------
        value
            Value to set the window with.

        Returns
        -------
        :class:`numpy.floating`
            Window.
        """

        return self._window

    @window.setter
    def window(self, value: Floating):
        """Setter for the **self.window** property."""

        attest(
            bool(value >= 1), '"window" must be equal to or greater than 1!'
        )

        self._window = value

        # Triggering "self._x_p" update.
        if self._x is not None:
            self.x = self._x

        # Triggering "self._y_p" update.
        if self._y is not None:
            self.y = self._y

    @property
    def kernel(self) -> Callable:
        """
        Getter and setter property for the kernel callable.

        Parameters
        ----------
        value
            Value to set the kernel callable.

        Returns
        -------
        Callable
            Kernel callable.
        """

        return self._kernel

    @kernel.setter
    def kernel(self, value: Callable):
        """Setter for the **self.kernel** property."""

        attest(
            hasattr(value, "__call__"),
            f'"kernel" property: "{value}" is not callable!',
        )

        self._kernel = value

    @property
    def kernel_kwargs(self) -> Dict:
        """
        Getter and setter property for the kernel call time arguments.

        Parameters
        ----------
        value
            Value to call the interpolation kernel with.

        Returns
        -------
        :class:`dict`
            Kernel call time arguments.
        """

        return self._kernel_kwargs

    @kernel_kwargs.setter
    def kernel_kwargs(self, value: Dict):
        """Setter for the **self.kernel_kwargs** property."""

        attest(
            isinstance(value, dict),
            f'"kernel_kwargs" property: "{value}" type is not "dict"!',
        )

        self._kernel_kwargs = value

    @property
    def padding_kwargs(self) -> Dict:
        """
        Getter and setter property for the kernel call time arguments.

        Parameters
        ----------
        value
            Value to call the interpolation kernel with.

        Returns
        -------
        :class:`dict`
            Kernel call time arguments.
        """

        return self._padding_kwargs

    @padding_kwargs.setter
    def padding_kwargs(self, value: Dict):
        """Setter for the **self.padding_kwargs** property."""

        attest(
            isinstance(value, Mapping),
            f'"padding_kwargs" property: "{value}" type is not a "Mapping" instance!',
        )

        self._padding_kwargs = value

        # Triggering "self._y_p" update.
        if self._y is not None:
            self.y = self._y

    def __call__(self, x: FloatingOrArrayLike) -> FloatingOrNDArray:
        """
        Evaluate the interpolator at given point(s).

        Parameters
        ----------
        x
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.floating` or :class:`numpy.ndarray`
            Interpolated value(s).
        """

        x = np.atleast_1d(x).astype(self._dtype)

        xi = as_float(self._evaluate(x))

        return xi

    def _evaluate(self, x: NDArray) -> NDArray:
        """
        Perform the interpolator evaluation at given points.

        Parameters
        ----------
        x
            Points to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated points values.
        """

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        x_interval = interval(self._x)[0]
        x_f = np.floor(x / x_interval)

        windows = x_f[:, np.newaxis] + np.arange(
            -self._window + 1, self._window + 1
        )
        clip_l = min(self._x_p) / x_interval
        clip_h = max(self._x_p) / x_interval
        windows = np.clip(windows, clip_l, clip_h) - clip_l
        windows = as_int_array(np.around(windows))

        return np.sum(
            self._y_p[windows]
            * self._kernel(
                x[:, np.newaxis] / x_interval
                - windows
                - min(self._x_p) / x_interval,
                **self._kernel_kwargs,
            ),
            axis=-1,
        )

    def _validate_dimensions(self):
        """Validate that the variables dimensions are the same."""

        if len(self._x) != len(self._y):
            raise ValueError(
                '"x" independent and "y" dependent variables have different '
                f'dimensions: "{len(self._x)}", "{len(self._y)}"'
            )

    def _validate_interpolation_range(self, x: NDArray):
        """Validate given point to be in interpolation range."""

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if below_interpolation_range.any():
            raise ValueError(f'"{x}" is below interpolation range.')

        if above_interpolation_range.any():
            raise ValueError(f'"{x}" is above interpolation range.')


class NearestNeighbourInterpolator(KernelInterpolator):
    """
    A nearest-neighbour interpolator.

    Other Parameters
    ----------------
    dtype
        Data type used for internal conversions.
    padding_kwargs
         Arguments to use when padding :math:`y` variable values with the
         :func:`np.pad` definition.
    window
        Width of the window in samples on each side.
    x
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y
        Dependent and already known :math:`y` variable values to
        interpolate.

    Methods
    -------
    -   :meth:`~colour.NearestNeighbourInterpolator.__init__`
    """

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["kernel"] = kernel_nearest_neighbour
        if "kernel_kwargs" in kwargs:
            del kwargs["kernel_kwargs"]

        super().__init__(*args, **kwargs)


class LinearInterpolator:
    """
    Interpolate linearly a 1-D function.

    Parameters
    ----------
    x
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y
        Dependent and already known :math:`y` variable values to
        interpolate.
    dtype
        Data type used for internal conversions.

    Attributes
    ----------
    -   :attr:`~colour.LinearInterpolator.x`
    -   :attr:`~colour.LinearInterpolator.y`

    Methods
    -------
    -   :meth:`~colour.LinearInterpolator.__init__`
    -   :meth:`~colour.LinearInterpolator.__call__`

    Notes
    -----
    -   This class is a wrapper around *numpy.interp* definition.

    Examples
    --------
    Interpolating a single numeric variable:

    >>> y = np.array([5.9200, 9.3700, 10.8135, 4.5100,
    ...               69.5900, 27.8007, 86.0500])
    >>> x = np.arange(len(y))
    >>> f = LinearInterpolator(x, y)
    >>> f(0.5)  # doctest: +ELLIPSIS
    7.64...

    Interpolating an `ArrayLike` variable:

    >>> f([0.25, 0.75])
    array([ 6.7825,  8.5075])
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        dtype: Optional[Type[DTypeNumber]] = None,
    ):
        dtype = cast(Type[DTypeNumber], optional(dtype, DEFAULT_FLOAT_DTYPE))

        self._x: NDArray = np.array([])
        self._y: NDArray = np.array([])
        self._dtype: Type[DTypeNumber] = dtype

        # TODO: Remove pragma when https://github.com/python/mypy/issues/3004
        # is resolved.
        self.x = x  # type: ignore[assignment]
        self.y = y  # type: ignore[assignment]

        self._validate_dimensions()

    @property
    def x(self) -> NDArray:
        """
        Getter and setter property for the independent :math:`x` variable.

        Parameters
        ----------
        value
            Value to set the independent :math:`x` variable with.

        Returns
        -------
        :class:`numpy.ndarray`
            Independent :math:`x` variable.
        """

        return self._x

    @x.setter
    def x(self, value: ArrayLike):
        """Setter for the **self.x** property."""

        value = np.atleast_1d(value).astype(self._dtype)

        attest(
            value.ndim == 1,
            '"x" independent variable must have exactly one dimension!',
        )

        self._x = value

    @property
    def y(self) -> NDArray:
        """
        Getter and setter property for the dependent and already known
        :math:`y` variable.

        Parameters
        ----------
        value
            Value to set the dependent and already known :math:`y` variable
            with.

        Returns
        -------
        :class:`numpy.ndarray`
            Dependent and already known :math:`y` variable.
        """

        return self._y

    @y.setter
    def y(self, value: ArrayLike):
        """Setter for the **self.y** property."""

        value = np.atleast_1d(value).astype(self._dtype)

        attest(
            value.ndim == 1,
            '"y" dependent variable must have exactly one dimension!',
        )

        self._y = value

    def __call__(self, x: FloatingOrArrayLike) -> FloatingOrNDArray:
        """
        Evaluate the interpolating polynomial at given point(s).


        Parameters
        ----------
        x
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.floating` or :class:`numpy.ndarray`
            Interpolated value(s).
        """

        x = np.atleast_1d(x).astype(self._dtype)

        xi = as_float(self._evaluate(x))

        return xi

    def _evaluate(self, x: NDArray) -> NDArray:
        """
        Perform the interpolating polynomial evaluation at given points.

        Parameters
        ----------
        x
            Points to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated points values.
        """

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        return np.interp(x, self._x, self._y)

    def _validate_dimensions(self):
        """Validate that the variables dimensions are the same."""

        if len(self._x) != len(self._y):
            raise ValueError(
                '"x" independent and "y" dependent variables have different '
                f'dimensions: "{len(self._x)}", "{len(self._y)}"'
            )

    def _validate_interpolation_range(self, x: NDArray):
        """Validate given point to be in interpolation range."""

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if below_interpolation_range.any():
            raise ValueError(f'"{x}" is below interpolation range.')

        if above_interpolation_range.any():
            raise ValueError(f'"{x}" is above interpolation range.')


class SpragueInterpolator:
    """
    Construct a fifth-order polynomial that passes through :math:`y` dependent
    variable.

    *Sprague (1880)* method is recommended by the *CIE* for interpolating
    functions having a uniformly spaced independent variable.

    Parameters
    ----------
    x
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y
        Dependent and already known :math:`y` variable values to
        interpolate.
    dtype
        Data type used for internal conversions.

    Attributes
    ----------
    -   :attr:`~colour.SpragueInterpolator.x`
    -   :attr:`~colour.SpragueInterpolator.y`

    Methods
    -------
    -   :meth:`~colour.SpragueInterpolator.__init__`
    -   :meth:`~colour.SpragueInterpolator.__call__`

    Notes
    -----
    -   The minimum number :math:`k` of data points required along the
        interpolation axis is :math:`k=6`.

    References
    ----------
    :cite:`CIETC1-382005f`, :cite:`Westland2012h`

    Examples
    --------
    Interpolating a single numeric variable:

    >>> y = np.array([5.9200, 9.3700, 10.8135, 4.5100,
    ...               69.5900, 27.8007, 86.0500])
    >>> x = np.arange(len(y))
    >>> f = SpragueInterpolator(x, y)
    >>> f(0.5)  # doctest: +ELLIPSIS
    7.2185025...

    Interpolating an `ArrayLike` variable:

    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([ 6.7295161...,  7.8140625...])
    """

    SPRAGUE_C_COEFFICIENTS = np.array(
        [
            [884, -1960, 3033, -2648, 1080, -180],
            [508, -540, 488, -367, 144, -24],
            [-24, 144, -367, 488, -540, 508],
            [-180, 1080, -2648, 3033, -1960, 884],
        ]
    )
    """
    Defines the coefficients used to generate extra points for boundaries
    interpolation.

    SPRAGUE_C_COEFFICIENTS, (4, 6)

    References
    ----------
    :cite:`CIETC1-382005h`
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        dtype: Optional[Type[DTypeNumber]] = None,
    ):
        dtype = cast(Type[DTypeNumber], optional(dtype, DEFAULT_FLOAT_DTYPE))

        self._xp: NDArray = np.array([])
        self._yp: NDArray = np.array([])

        self._x: NDArray = np.array([])
        self._y: NDArray = np.array([])
        self._dtype: Type[DTypeNumber] = dtype

        # TODO: Remove pragma when https://github.com/python/mypy/issues/3004
        # is resolved.
        self.x = x  # type: ignore[assignment]
        self.y = y  # type: ignore[assignment]

        self._validate_dimensions()

    @property
    def x(self) -> NDArray:
        """
        Getter and setter property for the independent :math:`x` variable.

        Parameters
        ----------
        value
            Value to set the independent :math:`x` variable with.

        Returns
        -------
        :class:`numpy.ndarray`
            Independent :math:`x` variable.
        """

        return self._x

    @x.setter
    def x(self, value: ArrayLike):
        """Setter for the **self.x** property."""

        value = as_array(np.atleast_1d(value), self._dtype)

        attest(
            value.ndim == 1,
            '"x" independent variable must have exactly one dimension!',
        )

        self._x = value

        value_interval = interval(self._x)[0]

        xp1 = self._x[0] - value_interval * 2
        xp2 = self._x[0] - value_interval
        xp3 = self._x[-1] + value_interval
        xp4 = self._x[-1] + value_interval * 2

        self._xp = np.concatenate(
            [
                as_array([xp1, xp2], self._dtype),
                value,
                as_array([xp3, xp4], self._dtype),
            ]
        )

    @property
    def y(self) -> NDArray:
        """
        Getter and setter property for the dependent and already known
        :math:`y` variable.

        Parameters
        ----------
        value
            Value to set the dependent and already known :math:`y` variable
            with.

        Returns
        -------
        :class:`numpy.ndarray`
            Dependent and already known :math:`y` variable.
        """

        return self._y

    @y.setter
    def y(self, value: ArrayLike):
        """Setter for the **self.y** property."""

        value = as_array(np.atleast_1d(value), self._dtype)

        attest(
            value.ndim == 1,
            '"y" dependent variable must have exactly one dimension!',
        )

        attest(
            len(value) >= 6,
            '"y" dependent variable values count must be equal to or '
            "greater than 6!",
        )

        self._y = value

        yp1 = np.ravel(
            (
                np.dot(
                    self.SPRAGUE_C_COEFFICIENTS[0],
                    np.array(value[0:6]).reshape([6, 1]),
                )
            )
            / 209
        )[0]
        yp2 = np.ravel(
            (
                np.dot(
                    self.SPRAGUE_C_COEFFICIENTS[1],
                    np.array(value[0:6]).reshape([6, 1]),
                )
            )
            / 209
        )[0]
        yp3 = np.ravel(
            (
                np.dot(
                    self.SPRAGUE_C_COEFFICIENTS[2],
                    np.array(value[-6:]).reshape([6, 1]),
                )
            )
            / 209
        )[0]
        yp4 = np.ravel(
            (
                np.dot(
                    self.SPRAGUE_C_COEFFICIENTS[3],
                    np.array(value[-6:]).reshape([6, 1]),
                )
            )
            / 209
        )[0]

        self._yp = np.concatenate(
            [
                as_array([yp1, yp2], self._dtype),
                value,
                as_array([yp3, yp4], self._dtype),
            ]
        )

    def __call__(self, x: FloatingOrArrayLike) -> FloatingOrNDArray:
        """
        Evaluate the interpolating polynomial at given point(s).

        Parameters
        ----------
        x
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.floating` or :class:`numpy.ndarray`
            Interpolated value(s).
        """

        x = np.atleast_1d(x).astype(self._dtype)

        xi = as_float(self._evaluate(x))

        return xi

    def _evaluate(self, x: NDArray) -> NDArray:
        """
        Perform the interpolating polynomial evaluation at given point.

        Parameters
        ----------
        x
            Point to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated point values.
        """

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        i = np.searchsorted(self._xp, x) - 1
        X = (x - self._xp[i]) / (self._xp[i + 1] - self._xp[i])

        r = self._yp

        a0p = r[i]
        a1p = (
            2 * r[i - 2] - 16 * r[i - 1] + 16 * r[i + 1] - 2 * r[i + 2]
        ) / 24
        a2p = (
            -r[i - 2] + 16 * r[i - 1] - 30 * r[i] + 16 * r[i + 1] - r[i + 2]
        ) / 24
        a3p = (
            -9 * r[i - 2]
            + 39 * r[i - 1]
            - 70 * r[i]
            + 66 * r[i + 1]
            - 33 * r[i + 2]
            + 7 * r[i + 3]
        ) / 24
        a4p = (
            13 * r[i - 2]
            - 64 * r[i - 1]
            + 126 * r[i]
            - 124 * r[i + 1]
            + 61 * r[i + 2]
            - 12 * r[i + 3]
        ) / 24
        a5p = (
            -5 * r[i - 2]
            + 25 * r[i - 1]
            - 50 * r[i]
            + 50 * r[i + 1]
            - 25 * r[i + 2]
            + 5 * r[i + 3]
        ) / 24

        y = (
            a0p
            + a1p * X
            + a2p * X**2
            + a3p * X**3
            + a4p * X**4
            + a5p * X**5
        )

        return y

    def _validate_dimensions(self):
        """Validate that the variables dimensions are the same."""

        if len(self._x) != len(self._y):
            raise ValueError(
                '"x" independent and "y" dependent variables have different '
                f'dimensions: "{len(self._x)}", "{len(self._y)}"'
            )

    def _validate_interpolation_range(self, x: NDArray):
        """Validate given point to be in interpolation range."""

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if below_interpolation_range.any():
            raise ValueError(f'"{x}" is below interpolation range.')

        if above_interpolation_range.any():
            raise ValueError(f'"{x}" is above interpolation range.')


class CubicSplineInterpolator(scipy.interpolate.interp1d):
    """
    Interpolate a 1-D function using cubic spline interpolation.

    Methods
    -------
    -   :meth:`~colour.CubicSplineInterpolator.__init__`

    Notes
    -----
    -   This class is a wrapper around *scipy.interpolate.interp1d* class.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(kind="cubic", *args, **kwargs)


class PchipInterpolator(scipy.interpolate.PchipInterpolator):
    """
    Interpolate a 1-D function using Piecewise Cubic Hermite Interpolating
    Polynomial interpolation.

    Attributes
    ----------
    -   :attr:`~colour.PchipInterpolator.y`

    Methods
    -------
    -   :meth:`~colour.PchipInterpolator.__init__`

    Notes
    -----
    -   This class is a wrapper around *scipy.interpolate.PchipInterpolator*
        class.
    """

    def __init__(self, x: ArrayLike, y: ArrayLike, *args: Any, **kwargs: Any):
        super().__init__(x, y, *args, **kwargs)

        self._y: NDArray = as_float_array(y)

    @property
    def y(self) -> NDArray:
        """
        Getter property for the dependent and already known :math:`y`
        variable.

        Returns
        -------
        :class:`numpy.ndarray`
            Dependent and already known :math:`y` variable.
        """

        return self._y


class NullInterpolator:
    """
    Perform 1-D function null interpolation, i.e. a call within given
    tolerances will return existing :math:`y` variable values and ``default``
    if outside tolerances.

    Parameters
    ----------
    x
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y
        Dependent and already known :math:`y` variable values to
        interpolate.
    absolute_tolerance
        Absolute tolerance.
    relative_tolerance
        Relative tolerance.
    default
        Default value for interpolation outside tolerances.
    dtype
        Data type used for internal conversions.

    Attributes
    ----------
    -   :attr:`~colour.NullInterpolator.x`
    -   :attr:`~colour.NullInterpolator.y`
    -   :attr:`~colour.NullInterpolator.relative_tolerance`
    -   :attr:`~colour.NullInterpolator.absolute_tolerance`
    -   :attr:`~colour.NullInterpolator.default`

    Methods
    -------
    -   :meth:`~colour.NullInterpolator.__init__`
    -   :meth:`~colour.NullInterpolator.__call__`

    Examples
    --------
    >>> y = np.array([5.9200, 9.3700, 10.8135, 4.5100,
    ...               69.5900, 27.8007, 86.0500])
    >>> x = np.arange(len(y))
    >>> f = NullInterpolator(x, y)
    >>> f(0.5)
    nan
    >>> f(1.0)  # doctest: +ELLIPSIS
    9.3699999...
    >>> f = NullInterpolator(x, y, absolute_tolerance=0.01)
    >>> f(1.01)  # doctest: +ELLIPSIS
    9.3699999...
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        absolute_tolerance: Floating = 10e-7,
        relative_tolerance: Floating = 10e-7,
        default: Floating = np.nan,
        dtype: Optional[Type[DTypeNumber]] = None,
    ):
        dtype = cast(Type[DTypeNumber], optional(dtype, DEFAULT_FLOAT_DTYPE))

        self._x: NDArray = np.array([])
        self._y: NDArray = np.array([])
        self._absolute_tolerance: float = 10e-7
        self._relative_tolerance: float = 10e-7
        self._default: Floating = np.nan
        self._dtype: Type[DTypeNumber] = dtype

        # TODO: Remove pragma when https://github.com/python/mypy/issues/3004
        # is resolved.
        self.x = x  # type: ignore[assignment]
        self.y = y  # type: ignore[assignment]
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.default = default

        self._validate_dimensions()

    @property
    def x(self) -> NDArray:
        """
        Getter and setter property for the independent :math:`x` variable.

        Parameters
        ----------
        value
            Value to set the independent :math:`x` variable with.

        Returns
        -------
        :class:`numpy.ndarray`
            Independent :math:`x` variable.
        """

        return self._x

    @x.setter
    def x(self, value: ArrayLike):
        """Setter for the **self.x** property."""

        value = np.atleast_1d(value).astype(self._dtype)

        attest(
            value.ndim == 1,
            '"x" independent variable must have exactly one dimension!',
        )

        self._x = value

    @property
    def y(self) -> NDArray:
        """
        Getter and setter property for the dependent and already known
        :math:`y` variable.

        Parameters
        ----------
        value
            Value to set the dependent and already known :math:`y` variable
            with.

        Returns
        -------
        :class:`numpy.ndarray`
            Dependent and already known :math:`y` variable.
        """

        return self._y

    @y.setter
    def y(self, value: ArrayLike):
        """Setter for the **self.y** property."""

        value = np.atleast_1d(value).astype(self._dtype)

        attest(
            value.ndim == 1,
            '"y" dependent variable must have exactly one dimension!',
        )

        self._y = value

    @property
    def relative_tolerance(self) -> Floating:
        """
        Getter and setter property for the relative tolerance.

        Parameters
        ----------
        value
            Value to set the relative tolerance with.

        Returns
        -------
        :class:`numpy.floating`
            Relative tolerance.
        """

        return self._relative_tolerance

    @relative_tolerance.setter
    def relative_tolerance(self, value: Floating):
        """Setter for the **self.relative_tolerance** property."""

        attest(
            is_numeric(value),
            '"relative_tolerance" variable must be a "numeric"!',
        )

        self._relative_tolerance = float(value)

    @property
    def absolute_tolerance(self) -> Floating:
        """
        Getter and setter property for the absolute tolerance.

        Parameters
        ----------
        value
            Value to set the absolute tolerance with.

        Returns
        -------
        :class:`numpy.floating`
            Absolute tolerance.
        """

        return self._absolute_tolerance

    @absolute_tolerance.setter
    def absolute_tolerance(self, value: Floating):
        """Setter for the **self.absolute_tolerance** property."""

        attest(
            is_numeric(value),
            '"absolute_tolerance" variable must be a "numeric"!',
        )

        self._absolute_tolerance = float(value)

    @property
    def default(self) -> Floating:
        """
        Getter and setter property for the default value for call outside
        tolerances.

        Parameters
        ----------
        value
            Value to set the default value with.

        Returns
        -------
        :class:`numpy.floating`
            Default value.
        """

        return self._default

    @default.setter
    def default(self, value: Floating):
        """Setter for the **self.default** property."""

        attest(is_numeric(value), '"default" variable must be a "numeric"!')

        self._default = value

    def __call__(self, x: FloatingOrArrayLike) -> FloatingOrNDArray:
        """
        Evaluate the interpolator at given point(s).


        Parameters
        ----------
        x
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.floating` or :class:`numpy.ndarray`
            Interpolated value(s).
        """

        x = np.atleast_1d(x).astype(self._dtype)

        xi = as_float(self._evaluate(x))

        return xi

    def _evaluate(self, x: NDArray) -> NDArray:
        """
        Perform the interpolator evaluation at given points.

        Parameters
        ----------
        x
            Points to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated points values.
        """

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        indexes = closest_indexes(self._x, x)
        values = self._y[indexes]
        values[
            ~np.isclose(
                self._x[indexes],
                x,
                rtol=self._absolute_tolerance,
                atol=self._relative_tolerance,
            )
        ] = self._default

        return values

    def _validate_dimensions(self):
        """Validate that the variables dimensions are the same."""

        if len(self._x) != len(self._y):
            raise ValueError(
                '"x" independent and "y" dependent variables have different '
                f'dimensions: "{len(self._x)}", "{len(self._y)}"'
            )

    def _validate_interpolation_range(self, x: NDArray):
        """Validate given point to be in interpolation range."""

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if below_interpolation_range.any():
            raise ValueError(f'"{x}" is below interpolation range.')

        if above_interpolation_range.any():
            raise ValueError(f'"{x}" is above interpolation range.')


def lagrange_coefficients(r: Floating, n: Integer = 4) -> NDArray:
    """
    Compute the *Lagrange Coefficients* at given point :math:`r` for degree
    :math:`n`.

    Parameters
    ----------
    r
        Point to get the *Lagrange Coefficients* at.
    n
        Degree of the *Lagrange Coefficients* being calculated.

    Returns
    -------
    :class:`numpy.ndarray`

    References
    ----------
    :cite:`Fairman1985b`, :cite:`Wikipedia2003a`

    Examples
    --------
    >>> lagrange_coefficients(0.1)
    array([ 0.8265,  0.2755, -0.1305,  0.0285])
    """

    r_i = np.arange(n)
    L_n = []
    for j in range(len(r_i)):
        basis = [
            (r - r_i[i]) / (r_i[j] - r_i[i]) for i in range(len(r_i)) if i != j
        ]
        L_n.append(reduce(lambda x, y: x * y, basis))  # noqa

    return np.array(L_n)


def vertices_and_relative_coordinates(
    V_xyz: ArrayLike, table: ArrayLike
) -> Tuple[NDArray, NDArray]:
    """
    Compute the vertices coordinates and indexes relative :math:`V_{xyzr}`
    coordinates from given :math:`V_{xyzr}` values and interpolation table.

    Parameters
    ----------
    V_xyz
        :math:`V_{xyz}` values to transform to indexes relative
        :math:`V_{xyzr}` values.
    table
        4-Dimensional (NxNxNx3) interpolation table.

    Returns
    -------
    :class:`tuple`
        Vertices coordinates and indexes relative :math:`V_{xyzr}` coordinates.

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     os.path.dirname(__file__),'..', 'io', 'luts', 'tests', 'resources',
    ...     'iridas_cube', 'Colour_Correct.cube')
    >>> LUT = colour.read_LUT(path)
    >>> table = LUT.table
    >>> prng = np.random.RandomState(4)
    >>> V_xyz = colour.algebra.random_triplet_generator(3, random_state=prng)
    >>> print(V_xyz)  # doctest: +ELLIPSIS
    [[ 0.9670298...  0.7148159...  0.9762744...]
     [ 0.5472322...  0.6977288...  0.0062302...]
     [ 0.9726843...  0.2160895...  0.2529823...]]
    >>> vertices, V_xyzr = vertices_and_relative_coordinates(V_xyz, table)
    >>> print(vertices)
    [[[ 0.833311  0.833311  0.833311]
      [ 0.349416  0.657749  0.041083]
      [ 0.797894 -0.035412 -0.035412]]
    <BLANKLINE>
     [[ 0.833311  0.833311  1.249963]
      [ 0.340435  0.743769  0.340435]
      [ 0.752767 -0.028479  0.362144]]
    <BLANKLINE>
     [[ 0.707102  1.110435  0.707102]
      [ 0.344991  1.050213 -0.007621]
      [ 0.633333  0.316667  0.      ]]
    <BLANKLINE>
     [[ 0.519714  0.744729  0.744729]
      [ 0.314204  1.120871  0.314204]
      [ 0.732278  0.315626  0.315626]]
    <BLANKLINE>
     [[ 1.06561   0.648957  0.648957]
      [ 0.589195  0.589195  0.139164]
      [ 1.196841 -0.053117 -0.053117]]
    <BLANKLINE>
     [[ 1.        0.666667  1.      ]
      [ 0.594601  0.594601  0.369586]
      [ 1.162588 -0.050372  0.353948]]
    <BLANKLINE>
     [[ 0.894606  0.894606  0.66959 ]
      [ 0.663432  0.930188  0.12992 ]
      [ 1.038439  0.310899 -0.05287 ]]
    <BLANKLINE>
     [[ 1.249966  1.249966  1.249966]
      [ 0.682749  0.991082  0.374416]
      [ 1.131225  0.29792   0.29792 ]]]
    >>> print(V_xyzr)  # doctest: +ELLIPSIS
    [[ 0.9010895...  0.1444479...  0.9288233...]
     [ 0.6416967...  0.0931864...  0.0186907...]
     [ 0.9180530...  0.6482684...  0.7589470...]]
    """

    V_xyz = np.clip(V_xyz, 0, 1)
    table = as_float_array(table)

    V_xyz = np.reshape(V_xyz, (-1, 3))

    # Indexes computations where ``i_m`` is the maximum index value on a given
    # table axis, ``i_f`` and ``i_c`` respectively the floor and ceiling
    # indexes encompassing a given V_xyz value.
    i_m = np.array(table.shape[0:-1]) - 1
    i_f = as_int_array(np.floor(V_xyz * i_m))
    i_f = np.clip(i_f, 0, i_m)
    i_c = np.clip(i_f + 1, 0, i_m)

    # Relative to indexes ``V_xyz`` values.
    V_xyzr = i_m * V_xyz - i_f

    i_f_c = i_f, i_c

    # Vertices computations by indexing ``table`` with the ``i_f`` and ``i_c``
    # indexes. 8 encompassing vertices are computed for a given V_xyz value
    # forming a cube around it:
    vertices = np.array(
        [
            table[
                i_f_c[i[0]][..., 0], i_f_c[i[1]][..., 1], i_f_c[i[2]][..., 2]
            ]
            for i in itertools.product(*zip([0, 0, 0], [1, 1, 1]))
        ]
    )

    return vertices, V_xyzr


def table_interpolation_trilinear(
    V_xyz: ArrayLike, table: ArrayLike
) -> NDArray:
    """
    Perform the trilinear interpolation of given :math:`V_{xyz}` values using
    given interpolation table.

    Parameters
    ----------
    V_xyz
        :math:`V_{xyz}` values to interpolate.
    table
        4-Dimensional (NxNxNx3) interpolation table.

    Returns
    -------
    :class:`numpy.ndarray`
        Interpolated :math:`V_{xyz}` values.

    References
    ----------
    :cite:`Bourkeb`

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     os.path.dirname(__file__),'..', 'io', 'luts', 'tests', 'resources',
    ...     'iridas_cube', 'Colour_Correct.cube')
    >>> LUT = colour.read_LUT(path)
    >>> table = LUT.table
    >>> prng = np.random.RandomState(4)
    >>> V_xyz = colour.algebra.random_triplet_generator(3, random_state=prng)
    >>> print(V_xyz)  # doctest: +ELLIPSIS
    [[ 0.9670298...  0.7148159...  0.9762744...]
     [ 0.5472322...  0.6977288...  0.0062302...]
     [ 0.9726843...  0.2160895...  0.2529823...]]
    >>> table_interpolation_trilinear(V_xyz, table)  # doctest: +ELLIPSIS
    array([[ 1.0120664...,  0.7539146...,  1.0228540...],
           [ 0.5075794...,  0.6479459...,  0.1066404...],
           [ 1.0976519...,  0.1785998...,  0.2299897...]])
    """

    V_xyz = as_float_array(V_xyz)

    vertices, V_xyzr = vertices_and_relative_coordinates(V_xyz, table)

    vertices = np.moveaxis(vertices, 0, 1)
    x, y, z = (f[:, np.newaxis] for f in tsplit(V_xyzr))

    weights = np.moveaxis(
        np.transpose(
            [
                (1 - x) * (1 - y) * (1 - z),
                (1 - x) * (1 - y) * z,
                (1 - x) * y * (1 - z),
                (1 - x) * y * z,
                x * (1 - y) * (1 - z),
                x * (1 - y) * z,
                x * y * (1 - z),
                x * y * z,
            ]
        ),
        0,
        -1,
    )

    xyz_o = np.reshape(np.sum(vertices * weights, 1), V_xyz.shape)

    return xyz_o


def table_interpolation_tetrahedral(
    V_xyz: ArrayLike, table: ArrayLike
) -> NDArray:
    """
    Perform the tetrahedral interpolation of given :math:`V_{xyz}` values using
    given interpolation table.

    Parameters
    ----------
    V_xyz
        :math:`V_{xyz}` values to interpolate.
    table
        4-Dimensional (NxNxNx3) interpolation table.

    Returns
    -------
    :class:`numpy.ndarray`
        Interpolated :math:`V_{xyz}` values.

    References
    ----------
    :cite:`Kirk2006`

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     os.path.dirname(__file__),'..', 'io', 'luts', 'tests', 'resources',
    ...     'iridas_cube', 'Colour_Correct.cube')
    >>> LUT = colour.read_LUT(path)
    >>> table = LUT.table
    >>> prng = np.random.RandomState(4)
    >>> V_xyz = colour.algebra.random_triplet_generator(3, random_state=prng)
    >>> print(V_xyz)  # doctest: +ELLIPSIS
    [[ 0.9670298...  0.7148159...  0.9762744...]
     [ 0.5472322...  0.6977288...  0.0062302...]
     [ 0.9726843...  0.2160895...  0.2529823...]]
    >>> table_interpolation_tetrahedral(V_xyz, table)  # doctest: +ELLIPSIS
    array([[ 1.0196197...,  0.7674062...,  1.0311751...],
           [ 0.5105603...,  0.6466722...,  0.1077296...],
           [ 1.1178206...,  0.1762039...,  0.2209534...]])
    """

    V_xyz = as_float_array(V_xyz)

    vertices, V_xyzr = vertices_and_relative_coordinates(V_xyz, table)

    vertices = np.moveaxis(vertices, 0, -1)
    V000, V001, V010, V011, V100, V101, V110, V111 = tsplit(vertices)
    x, y, z = (r[:, np.newaxis] for r in tsplit(V_xyzr))

    xyz_o = np.select(
        [
            np.logical_and(x > y, y > z),
            np.logical_and(x > y, x > z),
            np.logical_and(x > y, np.logical_and(y <= z, x <= z)),
            np.logical_and(x <= y, z > y),
            np.logical_and(x <= y, z > x),
            np.logical_and(x <= y, np.logical_and(z <= y, z <= x)),
        ],
        [
            (1 - x) * V000 + (x - y) * V100 + (y - z) * V110 + z * V111,
            (1 - x) * V000 + (x - z) * V100 + (z - y) * V101 + y * V111,
            (1 - z) * V000 + (z - x) * V001 + (x - y) * V101 + y * V111,
            (1 - z) * V000 + (z - y) * V001 + (y - x) * V011 + x * V111,
            (1 - y) * V000 + (y - z) * V010 + (z - x) * V011 + x * V111,
            (1 - y) * V000 + (y - x) * V010 + (x - z) * V110 + z * V111,
        ],
    )

    xyz_o = np.reshape(xyz_o, V_xyz.shape)

    return xyz_o


TABLE_INTERPOLATION_METHODS = CaseInsensitiveMapping(
    {
        "Trilinear": table_interpolation_trilinear,
        "Tetrahedral": table_interpolation_tetrahedral,
    }
)
TABLE_INTERPOLATION_METHODS.__doc__ = """
Supported table interpolation methods.

References
----------
:cite:`Bourkeb`, :cite:`Kirk2006`
"""


def table_interpolation(
    V_xyz: ArrayLike,
    table: ArrayLike,
    method: Union[Literal["Trilinear", "Tetrahedral"], str] = "Trilinear",
) -> NDArray:
    """
    Perform interpolation of given :math:`V_{xyz}` values using given
    interpolation table.

    Parameters
    ----------
    V_xyz
        :math:`V_{xyz}` values to interpolate.
    table
        4-Dimensional (NxNxNx3) interpolation table.
    method
        Interpolation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Interpolated :math:`V_{xyz}` values.

    References
    ----------
    :cite:`Bourkeb`, :cite:`Kirk2006`

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     os.path.dirname(__file__),'..', 'io', 'luts', 'tests', 'resources',
    ...     'iridas_cube', 'Colour_Correct.cube')
    >>> LUT = colour.read_LUT(path)
    >>> table = LUT.table
    >>> prng = np.random.RandomState(4)
    >>> V_xyz = colour.algebra.random_triplet_generator(3, random_state=prng)
    >>> print(V_xyz)  # doctest: +ELLIPSIS
    [[ 0.9670298...  0.7148159...  0.9762744...]
     [ 0.5472322...  0.6977288...  0.0062302...]
     [ 0.9726843...  0.2160895...  0.2529823...]]
    >>> table_interpolation(V_xyz, table)  # doctest: +ELLIPSIS
    array([[ 1.0120664...,  0.7539146...,  1.0228540...],
           [ 0.5075794...,  0.6479459...,  0.1066404...],
           [ 1.0976519...,  0.1785998...,  0.2299897...]])
    >>> table_interpolation(V_xyz, table, method='Tetrahedral')
    ... # doctest: +ELLIPSIS
    array([[ 1.0196197...,  0.7674062...,  1.0311751...],
           [ 0.5105603...,  0.6466722...,  0.1077296...],
           [ 1.1178206...,  0.1762039...,  0.2209534...]])
    """

    method = validate_method(method, TABLE_INTERPOLATION_METHODS)

    return TABLE_INTERPOLATION_METHODS[method](V_xyz, table)
