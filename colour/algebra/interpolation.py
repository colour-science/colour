"""
Interpolation
=============

Provide classes and functions for interpolating variables in colour science
computations.

This module implements various interpolation methods for one-dimensional
functions and multi-dimensional table-based interpolation. These methods
support spectral data processing, colour transformations, and general
numerical interpolation tasks in colour science applications.

-   :class:`colour.KernelInterpolator`: 1-D function generic interpolation
    with arbitrary kernel.
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
-   :func:`colour.lagrange_coefficients`: Compute *Lagrange Coefficients*.
-   :func:`colour.algebra.table_interpolation_trilinear`: Perform trilinear
    interpolation with table.
-   :func:`colour.algebra.table_interpolation_tetrahedral`: Perform
    tetrahedral interpolation with table.
-   :attr:`colour.TABLE_INTERPOLATION_METHODS`: Supported table interpolation
    methods.
-   :func:`colour.table_interpolation`: Perform interpolation with table using
    specified method.

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

import sys
import typing
from functools import reduce
from unittest.mock import MagicMock

import numpy as np

from colour.utilities.requirements import is_scipy_installed
from colour.utilities.verbose import usage_warning

if not is_scipy_installed():  # pragma: no cover
    try:
        is_scipy_installed(raise_exception=True)
    except ImportError as error:
        usage_warning(str(error))

    mock = MagicMock()
    mock.__name__ = ""

    for module in (
        "scipy",
        "scipy.interpolate",
    ):
        sys.modules[module] = mock

import scipy.interpolate

from colour.algebra import sdiv, sdiv_mode
from colour.constants import (
    DTYPE_FLOAT_DEFAULT,
    DTYPE_INT_DEFAULT,
    TOLERANCE_ABSOLUTE_DEFAULT,
    TOLERANCE_RELATIVE_DEFAULT,
)

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        Callable,
        DTypeReal,
        Literal,
        Type,
    )

from colour.hints import NDArrayFloat, NDArrayReal, cast
from colour.utilities import (
    CanonicalMapping,
    array_namespace,
    as_array,
    as_float,
    as_float_array,
    as_float_scalar,
    as_int_array,
    as_ndarray,
    attest,
    closest_indexes,
    first_item,
    interval,
    is_numeric,
    is_numpy_namespace,
    optional,
    runtime_warning,
    validate_method,
    xp_as_float_array,
    xp_as_int_array,
    xp_astype,
    xp_atleast_1d,
    xp_isclose,
    xp_pad,
    xp_reshape,
    xp_select,
    xp_sinc,
    xp_squeeze,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
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
    "table_interpolation_trilinear",
    "table_interpolation_tetrahedral",
    "TABLE_INTERPOLATION_METHODS",
    "table_interpolation",
    "linear_interpolation_index_and_factor",
]


def kernel_nearest_neighbour(x: ArrayLike) -> NDArrayFloat:
    """
    Return the *nearest-neighbour* kernel evaluated at specified samples.

    The *nearest-neighbour* kernel is a discontinuous kernel function that
    equals 1 for samples within the range [-0.5, 0.5) and 0 elsewhere. This
    kernel represents the simplest interpolation method where each output
    value is determined by the closest input sample.

    Parameters
    ----------
    x
        Samples at which to evaluate the *nearest-neighbour* kernel.

    Returns
    -------
    :class:`numpy.ndarray`
        The *nearest-neighbour* kernel evaluated at specified samples.

    References
    ----------
    :cite:`Burger2009b`

    Examples
    --------
    >>> kernel_nearest_neighbour(np.linspace(0, 1, 10))
    array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    """

    x = as_float_array(x)

    xp = array_namespace(x)

    return xp.where(xp.abs(x) < 0.5, 1, 0)


def kernel_linear(x: ArrayLike) -> NDArrayFloat:
    """
    Evaluate the *linear* kernel at specified samples.

    The *linear* kernel is a triangular function that returns 1 when
    :math:`|x| < 0.5` and 0 otherwise, providing a simple binary response
    based on the absolute value of the input.

    Parameters
    ----------
    x
        Samples at which to evaluate the *linear* kernel.

    Returns
    -------
    :class:`numpy.ndarray`
        The *linear* kernel evaluated at specified samples, with values of 1
        for :math:`|x| < 0.5` and 0 otherwise.

    References
    ----------
    :cite:`Burger2009b`

    Examples
    --------
    >>> kernel_linear(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([1.        , 0.8888888..., 0.7777777..., 0.6666666..., 0.5555555...,
           0.4444444..., 0.3333333..., 0.2222222..., 0.1111111..., 0.        ])
    """

    x = as_float_array(x)

    xp = array_namespace(x)

    return xp.where(xp.abs(x) < 1, 1 - xp.abs(x), 0)


def kernel_sinc(x: ArrayLike, a: float = 3) -> NDArrayFloat:
    """
    Evaluate the *sinc* kernel at specified sample positions.

    Compute the *sinc* kernel function, commonly used in signal processing
    and interpolation applications, for the specified sample positions.

    Parameters
    ----------
    x
        Sample positions at which to evaluate the *sinc* kernel.
    a
        Size parameter of the *sinc* kernel, controlling the function's
        support width.

    Returns
    -------
    :class:`numpy.ndarray`
        *Sinc* kernel values evaluated at the specified sample positions.

    Raises
    ------
    AssertionError
        If ``a`` is less than 1.

    References
    ----------
    :cite:`Burger2009b`

    Examples
    --------
    >>> kernel_sinc(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([1.00000000e+00, 9.7981553...e-01, 9.2072542...e-01, 8.2699334...e-01,
           7.0531659...e-01, 5.6425327...e-01, 4.1349667...e-01, 2.6306440...e-01,
           1.2247694...e-01, 3.8981718...e-17])
    """

    x = as_float_array(x)

    xp = array_namespace(x)

    attest(bool(a >= 1), '"a" must be equal or superior to 1!')

    return xp.where(xp.abs(x) < a, xp_sinc(x, xp=xp), 0)


def kernel_lanczos(x: ArrayLike, a: float = 3) -> NDArrayFloat:
    """
    Return the *Lanczos* kernel evaluated at specified samples.

    The *Lanczos* kernel is a sinc-based windowing function commonly used
    in signal processing and image resampling applications. It is defined
    as :math:`L(x) = \\text{sinc}(x) \\cdot \\text{sinc}(x/a)` for
    :math:`|x| < a`, and zero otherwise.

    Parameters
    ----------
    x
        Samples at which to evaluate the *Lanczos* kernel.
    a
        Size of the *Lanczos* kernel, defining the support region
        :math:`[-a, a]`.

    Returns
    -------
    :class:`numpy.ndarray`
        The *Lanczos* kernel evaluated at specified samples.

    References
    ----------
    :cite:`Wikipedia2005b`

    Examples
    --------
    >>> kernel_lanczos(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([1.00000000e+00, 9.7760615...e-01, 9.1243770...e-01, 8.1030092...e-01,
           6.8012706...e-01, 5.3295773...e-01, 3.8071690...e-01, 2.3492839...e-01,
           1.0554054...e-01, 3.2237621...e-17])
    """

    x = as_float_array(x)

    xp = array_namespace(x)

    attest(bool(a >= 1), '"a" must be equal or superior to 1!')

    return xp.where(xp.abs(x) < a, xp_sinc(x, xp=xp) * xp_sinc(x / a, xp=xp), 0)


def kernel_cardinal_spline(
    x: ArrayLike, a: float = 0.5, b: float = 0.0
) -> NDArrayFloat:
    """
    Return the *cardinal spline* kernel evaluated at specified samples.

    Notable *cardinal spline* :math:`a` and :math:`b` parameterizations:

    -   *Catmull-Rom*: :math:`(a=0.5, b=0)`
    -   *Cubic B-Spline*: :math:`(a=0, b=1)`
    -   *Mitchell-Netravalli*:
        :math:`(a=\\cfrac{1}{3}, b=\\cfrac{1}{3})`

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
        The *cardinal spline* kernel evaluated at specified samples.

    References
    ----------
    :cite:`Burger2009b`

    Examples
    --------
    >>> kernel_cardinal_spline(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([1.        , 0.9711934..., 0.8930041..., 0.7777777..., 0.6378600...,
           0.4855967..., 0.3333333..., 0.1934156..., 0.0781893..., 0.        ])
    """

    x = as_float_array(x)

    xp = array_namespace(x)

    x_abs = xp.abs(x)
    y = xp.where(
        x_abs < 1,
        (-6 * a - 9 * b + 12) * x_abs**3 + (6 * a + 12 * b - 18) * x_abs**2 - 2 * b + 6,
        (-6 * a - b) * x_abs**3
        + (30 * a + 6 * b) * x_abs**2
        + (-48 * a - 12 * b) * x_abs
        + 24 * a
        + 8 * b,
    )
    y = xp.where(x_abs >= 2, 0, y)

    return 1 / 6 * y


class KernelInterpolator:
    """
    Perform kernel-based interpolation of a 1-D function.

    Reconstruct a continuous signal from discrete samples using linear
    convolution. Express interpolation as the convolution of the specified
    discrete function :math:`g(x)` with a continuous interpolation kernel
    :math:`k(w)`:

        :math:`\\hat{g}(w_0) = [k * g](w_0) = \
\\sum_{x=-\\infty}^{\\infty}k(w_0 - x)\\cdot g(x)`

    Parameters
    ----------
    x
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y
        Dependent and already known :math:`y` variable values to interpolate.
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

    >>> y = np.array(
    ...     [5.9200, 9.3700, 10.8135, 4.5100, 69.5900, 27.8007, 86.0500]
    ... )
    >>> x = np.arange(len(y))
    >>> f = KernelInterpolator(x, y)
    >>> f(0.5)  # doctest: +ELLIPSIS
    np.float64(6.9411400...)

    Interpolating an `ArrayLike` variable:

    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([6.1806208..., 8.0823848...])

    Using a different *lanczos* kernel:

    >>> f = KernelInterpolator(x, y, kernel=kernel_sinc)
    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([6.5147317..., 8.3965466...])

    Using a different window size:

    >>> f = KernelInterpolator(
    ...     x, y, window=16, kernel=kernel_lanczos, kernel_kwargs={"a": 16}
    ... )
    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([5.3961792..., 5.6521093...])
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        window: float = 3,
        kernel: Callable = kernel_lanczos,
        kernel_kwargs: dict | None = None,
        padding_kwargs: dict | None = None,
        dtype: Type[DTypeReal] | None = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

        self._x_p: NDArrayFloat = np.asarray([])
        self._y_p: NDArrayFloat = np.asarray([])

        self._x: NDArrayFloat = np.asarray([])
        self._y: NDArrayFloat = np.asarray([])
        self._window: float = 3
        self._padding_kwargs: dict = {
            "pad_width": (window, window),
            "mode": "reflect",
        }
        self._kernel: Callable = kernel_lanczos
        self._kernel_kwargs: dict = {}
        self._dtype: Type[DTypeReal] = dtype

        self.x = x
        self.y = y
        self.window = window
        self.padding_kwargs = optional(padding_kwargs, self._padding_kwargs)
        self.kernel = kernel
        self.kernel_kwargs = optional(kernel_kwargs, self._kernel_kwargs)

        self._validate_dimensions()

    @property
    def x(self) -> NDArrayFloat:
        """
        Getter and setter for the independent :math:`x` variable.

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
    def x(self, value: ArrayLike) -> None:
        """Setter for the **self.x** property."""

        xp = array_namespace(value)

        value = xp_astype(xp_atleast_1d(value, xp=xp), self._dtype, xp=xp)

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

        self._x_p = xp_pad(
            self._x,
            as_int_array([self._window, self._window]),
            "linear_ramp",
            end_values=(
                float(xp.min(self._x) - self._window * value_interval[0]),
                float(xp.max(self._x) + self._window * value_interval[0]),
            ),
            xp=xp,
        )

    @property
    def y(self) -> NDArrayFloat:
        """
        Getter and setter for the dependent and already known :math:`y` variable.

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
    def y(self, value: ArrayLike) -> None:
        """Setter for the **self.y** property."""

        xp = array_namespace(value)

        value = xp_astype(xp_atleast_1d(value, xp=xp), self._dtype, xp=xp)

        attest(
            value.ndim in (1, 2),
            '"y" dependent variable must have one or two dimensions!',
        )

        self._y = as_array(value, self._dtype)

        if self._window is not None:
            pad_kwargs = dict(self._padding_kwargs)
            if self._y.ndim == 2:
                # Pad only along the leading axis for rank-2 ``y``; the
                # trailing axis is preserved.
                pad_width = pad_kwargs["pad_width"]
                pad_kwargs["pad_width"] = (pad_width, (0, 0))
            self._y_p = xp_pad(self._y, xp=xp, **pad_kwargs)

    @property
    def window(self) -> float:
        """
        Getter and setter for the filtering window size for the moving average.

        The window determines the number of samples used in the moving
        average calculation. A larger window produces smoother results
        with greater lag, while a smaller window yields more responsive
        but potentially noisier output.

        Parameters
        ----------
        value
            Value to set the window with.

        Returns
        -------
        :class:`float`
            Window size for the moving average filter.
        """

        return self._window

    @window.setter
    def window(self, value: float) -> None:
        """Setter for the **self.window** property."""

        attest(bool(value >= 1), '"window" must be equal to or greater than 1!')

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
        Getter and setter for the kernel callable for the interpolator.

        Parameters
        ----------
        value
            Value to set the callable object to use as the interpolation kernel
            with. Must be a callable that accepts numeric arguments.

        Returns
        -------
        Callable
             Callable object to use as the interpolation kernel.

        Raises
        ------
        AssertionError
            If the provided value is not callable.
        """

        return self._kernel

    @kernel.setter
    def kernel(self, value: Callable) -> None:
        """Setter for the **self.kernel** property."""

        attest(
            callable(value),
            f'"kernel" property: "{value}" is not callable!',
        )

        self._kernel = value

    @property
    def kernel_kwargs(self) -> dict:
        """
        Getter and setter for the kernel keyword arguments for the convolution
        operation.

        Parameters
        ----------
        value
            Value to set the keyword arguments to pass to the kernel function
            with.

        Returns
        -------
        :class:`dict`
            Keyword arguments to pass to the kernel function.

        Raises
        ------
        AssertionError
            If the provided value is not a :class:'dict` class instance.
        """

        return self._kernel_kwargs

    @kernel_kwargs.setter
    def kernel_kwargs(self, value: dict) -> None:
        """Setter for the **self.kernel_kwargs** property."""

        attest(
            isinstance(value, dict),
            f'"kernel_kwargs" property: "{value}" type is not "dict"!',
        )

        self._kernel_kwargs = value

    @property
    def padding_kwargs(self) -> dict:
        """
        Getter and setter for the padding keyword arguments for edge handling.

        Parameters
        ----------
        value
            Value to set the keyword arguments to pass to the padding function
            when handling edges during interpolation.

        Returns
        -------
        :class:`dict`
            Keyword arguments to pass to the padding function when handling
            edges during interpolation.

        Raises
        ------
        AssertionError
            If the provided value is not a :class:`dict` class instance.
        """

        return self._padding_kwargs

    @padding_kwargs.setter
    def padding_kwargs(self, value: dict) -> None:
        """Setter for the **self.padding_kwargs** property."""

        attest(
            isinstance(value, dict),
            f'"padding_kwargs" property: "{value}" type is not a "dict" instance!',
        )

        self._padding_kwargs = value

        # Triggering "self._y_p" update.
        if self._y is not None:
            self.y = self._y

    def __call__(self, x: ArrayLike) -> NDArrayFloat:
        """
        Evaluate the interpolator at specified point(s).

        Parameters
        ----------
        x
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated value(s).
        """

        x = as_float_array(x)

        xi = self._evaluate(x)

        # Collapse the ``xp_atleast_1d`` leading axis for 0-d ``x``.
        if x.ndim == 0:
            xi = first_item(xi)

        return as_float(xi)

    def _evaluate(self, x: NDArrayFloat) -> NDArrayFloat:
        """
        Evaluate the interpolating polynomial at the specified point.

        Parameters
        ----------
        x
            Point at which to evaluate the interpolant.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated values at the specified point.
        """

        xp = array_namespace(x, self._x, self._y)

        # Promoted to at least 1-D so the gather pipeline broadcasts uniformly.
        x = xp_atleast_1d(xp_as_float_array(x, xp=xp), xp=xp)

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        x_interval = float(interval(self._x)[0])
        x_f = xp.floor(x / x_interval)

        windows = x_f[..., None] + xp_as_float_array(
            np.arange(-self._window + 1, self._window + 1), xp=xp, like=x
        )
        clip_l = float(min(self._x_p)) / x_interval
        clip_h = float(max(self._x_p)) / x_interval
        windows = xp.clip(windows, clip_l, clip_h) - clip_l
        windows = as_int_array(xp.round(windows))

        weights = self._kernel(
            x[..., None] / x_interval - windows - float(min(self._x_p)) / x_interval,
            **self._kernel_kwargs,
        )
        if self._y.ndim == 2:
            # Append a trailing axis to the kernel weights so the product
            # with ``_y_p[windows]`` of shape ``(N_query, 2 * window, N_*)``
            # broadcasts over ``y``'s trailing axis.
            weights = weights[..., None]
        return xp.sum(
            xp_as_float_array(self._y_p, xp=xp, like=x)[windows] * weights,
            axis=1,
        )

    def _validate_dimensions(self) -> None:
        """
        Validate that the dimensions of the variables are equal.

        Raises
        ------
        ValueError
            If the x and y variable dimensions do not match.
        """

        if len(self._x) != len(self._y):
            error = (
                '"x" independent and "y" dependent variables have different '
                f'dimensions: "{len(self._x)}", "{len(self._y)}"'
            )

            raise ValueError(error)

    def _validate_interpolation_range(self, x: NDArrayFloat) -> None:
        """
        Validate that the specified interpolation point is within the valid
        interpolation range.

        The interpolation point must be within the bounds defined by the first
        and last x-coordinates of the interpolator's data.

        Parameters
        ----------
        x
            Point to validate for interpolation range compliance.

        Raises
        ------
        ValueError
            If the point is outside the valid interpolation range.
        """

        xp = array_namespace(x)

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if bool(xp.any(below_interpolation_range)):
            error = f'"{x}" is below interpolation range.'

            raise ValueError(error)

        if bool(xp.any(above_interpolation_range)):
            error = f'"{x}" is above interpolation range.'

            raise ValueError(error)


class NearestNeighbourInterpolator(KernelInterpolator):
    """
    Perform nearest-neighbour interpolation on discrete data.

    Implement a kernel-based interpolator that selects the closest known
    data point for each query position. This interpolator provides fast,
    discontinuous interpolation suitable for categorical data or when
    preserving exact measured values is required.

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["kernel"] = kernel_nearest_neighbour
        kwargs.pop("kernel_kwargs", None)

        super().__init__(*args, **kwargs)


class LinearInterpolator:
    """
    Perform linear interpolation of a 1-D function.

    This class provides a wrapper around NumPy's linear interpolation
    functionality for interpolating between specified data points.

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

    >>> y = np.array([5.9200, 9.3700, 10.8135, 4.5100, 69.5900, 27.8007, 86.0500])
    >>> x = np.arange(len(y))
    >>> f = LinearInterpolator(x, y)
    >>> f(0.5)  # doctest: +ELLIPSIS
    np.float64(7.64...)

    Interpolating an `ArrayLike` variable:

    >>> f([0.25, 0.75])
    array([6.7825, 8.5075])
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        dtype: Type[DTypeReal] | None = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

        self._x: NDArrayFloat = np.array([])
        self._y: NDArrayFloat = np.array([])
        self._dtype: Type[DTypeReal] = dtype

        self.x = x
        self.y = y

        self._validate_dimensions()

    @property
    def x(self) -> NDArrayFloat:
        """
        Getter and setter for the independent :math:`x` variable.

        Parameters
        ----------
        value
            Value to set the independent :math:`x` variable with.

        Returns
        -------
        :class:`numpy.ndarray`
            Independent :math:`x` variable.

        Raises
        ------
        AssertionError
            If the provided value has not exactly one dimension.
        """

        return self._x

    @x.setter
    def x(self, value: ArrayLike) -> None:
        """Setter for the **self.x** property."""

        xp = array_namespace(value)

        value = xp_astype(xp_atleast_1d(value, xp=xp), self._dtype, xp=xp)

        attest(
            value.ndim == 1,
            '"x" independent variable must have exactly one dimension!',
        )

        self._x = value

    @property
    def y(self) -> NDArrayFloat:
        """
        Getter and setter for the dependent and already known
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

        Raises
        ------
        AssertionError
            If the provided value has not exactly one dimension.
        """

        return self._y

    @y.setter
    def y(self, value: ArrayLike) -> None:
        """Setter for the **self.y** property."""

        xp = array_namespace(value)

        value = xp_astype(xp_atleast_1d(value, xp=xp), self._dtype, xp=xp)

        attest(
            value.ndim in (1, 2),
            '"y" dependent variable must have one or two dimensions!',
        )

        self._y = value

    def __call__(self, x: ArrayLike) -> NDArrayFloat:
        """
        Evaluate the interpolating polynomial at specified point(s).

        Parameters
        ----------
        x
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated value(s).
        """

        x = as_float_array(x)

        xi = self._evaluate(x)

        return as_float(xi)

    def _evaluate(self, x: NDArrayFloat) -> NDArrayFloat:
        """
        Perform the interpolating polynomial evaluation at specified points.

        Parameters
        ----------
        x
            Points to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated points values.
        """

        xp = array_namespace(x, self._x, self._y)

        x = xp_as_float_array(x, xp=xp)

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        # Bracket each query point with ``searchsorted``, compute the
        # normalised interpolation parameter ``t``, and lerp the bracketing
        # ``y`` rows. Indexing is done CPU-side via numpy so backends without
        # integer-tensor advanced indexing support (e.g. MPS) work correctly,
        # while the gathered values remain in the backend computational graph.
        self_x = xp_as_float_array(self._x, xp=xp, like=x)
        i_np = np.clip(as_ndarray(xp.searchsorted(self_x, x) - 1), 0, len(self._x) - 2)
        self_x_np = as_ndarray(self_x)
        self_y = xp_as_float_array(self._y, xp=xp)

        x_low = xp_as_float_array(self_x_np[i_np], xp=xp, like=x)
        x_high = xp_as_float_array(self_x_np[i_np + 1], xp=xp, like=x)
        y_low = self_y[i_np]
        y_high = self_y[i_np + 1]

        with sdiv_mode():
            t = sdiv(x - x_low, x_high - x_low)

        if self._y.ndim == 2:
            t = t[..., None]

        return y_low + (y_high - y_low) * t

    def _validate_dimensions(self) -> None:
        """Validate that the variables dimensions are the same."""

        if len(self._x) != len(self._y):
            error = (
                '"x" independent and "y" dependent variables have different '
                f'dimensions: "{len(self._x)}", "{len(self._y)}"'
            )

            raise ValueError(error)

    def _validate_interpolation_range(self, x: NDArrayFloat) -> None:
        """Validate specified point to be in interpolation range."""

        xp = array_namespace(x)

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if bool(xp.any(below_interpolation_range)):
            error = f'"{x}" is below interpolation range.'

            raise ValueError(error)

        if bool(xp.any(above_interpolation_range)):
            error = f'"{x}" is above interpolation range.'

            raise ValueError(error)


class SpragueInterpolator:
    """
    Perform fifth-order polynomial interpolation using the *Sprague (1880)*
    method for uniformly spaced data.

    Implement the *Sprague (1880)* interpolation method recommended by the
    *CIE* for interpolating functions with uniformly spaced independent
    variables. This interpolator constructs a fifth-order polynomial that
    passes through specified dependent variable values, providing smooth
    interpolation suitable for spectral data and other colour science
    applications.

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

    >>> y = np.array([5.9200, 9.3700, 10.8135, 4.5100, 69.5900, 27.8007, 86.0500])
    >>> x = np.arange(len(y))
    >>> f = SpragueInterpolator(x, y)
    >>> f(0.5)  # doctest: +ELLIPSIS
    np.float64(7.2185025...)

    Interpolating an `ArrayLike` variable:

    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([6.7295161..., 7.8140625...])
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
        dtype: Type[DTypeReal] | None = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

        self._x_p: NDArrayFloat = np.array([])
        self._y_p: NDArrayFloat = np.array([])

        self._x: NDArrayFloat = np.array([])
        self._y: NDArrayFloat = np.array([])
        self._dtype: Type[DTypeReal] = dtype

        self.x = x
        self.y = y

        self._validate_dimensions()

    @property
    def x(self) -> NDArrayFloat:
        """
        Getter and setter for the independent :math:`x` variable.

        Parameters
        ----------
        value
            Value to set the independent :math:`x` variable with.

        Returns
        -------
        :class:`numpy.ndarray`
            Independent :math:`x` variable.

        Raises
        ------
        AssertionError
            If the provided value has not exactly one dimension.
        """

        return self._x

    @x.setter
    def x(self, value: ArrayLike) -> None:
        """Setter for the **self.x** property."""

        xp = array_namespace(value)

        value = xp_astype(xp_atleast_1d(value, xp=xp), self._dtype, xp=xp)

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

        # NOTE: The boundary values are stacked rather than converted so that
        # they stay on ``value``'s device; the dtype is restored as the
        # interval arithmetic promotes it.
        self._x_p = xp.concat(
            [
                xp_astype(xp.stack([xp1, xp2]), self._dtype, xp=xp),
                value,
                xp_astype(xp.stack([xp3, xp4]), self._dtype, xp=xp),
            ]
        )

    @property
    def y(self) -> NDArrayFloat:
        """
        Getter and setter for the dependent and already known
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

        Raises
        ------
        AssertionError
            If the provided value has not exactly one dimension and its value
            count is less than 6.
        """

        return self._y

    @y.setter
    def y(self, value: ArrayLike) -> None:
        """Setter for the **self.y** property."""

        xp = array_namespace(value)

        value = xp_astype(xp_atleast_1d(value, xp=xp), self._dtype, xp=xp)

        attest(
            value.ndim in (1, 2),
            '"y" dependent variable must have one or two dimensions!',
        )

        attest(
            len(value) >= 6,
            '"y" dependent variable values count must be equal to or greater than 6!',
        )

        self._y = value

        # ``SPRAGUE_C_COEFFICIENTS`` is ``(4, 6)``; for rank-2 ``y`` the
        # coefficient tensor expands to ``(4, 6, 1)`` so the product with
        # the stacked boundary windows broadcasts over ``y``'s trailing
        # axis. The sum collapses the window axis.
        C = xp_as_float_array(self.SPRAGUE_C_COEFFICIENTS, xp=xp, like=value)
        if value.ndim == 2:
            C = C[..., None]
        yp = (
            xp.sum(
                C * xp.stack([value[0:6], value[0:6], value[-6:], value[-6:]]),
                axis=1,
            )
            / 209
        )

        self._y_p = xp.concat([yp[:2], value, yp[2:]], axis=0)

    def __call__(self, x: ArrayLike) -> NDArrayFloat:
        """
        Evaluate the interpolating polynomial at specified point(s).

        Parameters
        ----------
        x
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated value(s).
        """

        x = as_float_array(x)

        xi = self._evaluate(x)

        # Collapse the ``xp_atleast_1d`` leading axis for 0-d ``x``.
        if x.ndim == 0:
            xi = first_item(xi)

        return as_float(xi)

    def _evaluate(self, x: NDArrayFloat) -> NDArrayFloat:
        """
        Perform the interpolating polynomial evaluation at specified point.

        Parameters
        ----------
        x
            Point to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated point values.
        """

        xp = array_namespace(x, self._x, self._y)

        # Promoted to at least 1-D so the stack / matmul pipeline broadcasts uniformly.
        x = xp_atleast_1d(xp_as_float_array(x, xp=xp), xp=xp)

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        i = xp.searchsorted(xp_as_float_array(self._x_p, xp=xp, like=x), x) - 1
        # Convert index to numpy for CPU-side gather so that backends without
        # integer-tensor advanced indexing support (e.g. MPS) work correctly.
        i_np = as_ndarray(i)
        # ``t`` is the normalised interpolation parameter, the local
        # fractional distance ``(x - x_i) / (x_{i+1} - x_i)`` within the
        # bracketing interval.
        with sdiv_mode():
            _x_p_np = as_ndarray(xp_as_float_array(self._x_p, xp=xp))
            t = sdiv(
                x - xp_as_float_array(_x_p_np[i_np], xp=xp, like=x),
                xp_as_float_array(_x_p_np[i_np + 1] - _x_p_np[i_np], xp=xp, like=x),
            )

        r = xp_as_float_array(self._y_p, xp=xp)
        r_s = xp.stack([r[i_np + k] for k in (-2, -1, 0, 1, 2, 3)])
        w_s = xp_as_float_array(
            (
                (2, -16, 0, 16, -2, 0),
                (-1, 16, -30, 16, -1, 0),
                (-9, 39, -70, 66, -33, 7),
                (13, -64, 126, -124, 61, -12),
                (-5, 25, -50, 50, -25, 5),
            ),
            xp=xp,
            like=x,
        )
        # Flatten the trailing axes of ``r_s`` so a single ``matmul`` over
        # the window axis handles rank-1 and rank-2 ``y`` uniformly.
        r_s_shape = r_s.shape
        r_s_flat = xp_reshape(r_s, (r_s_shape[0], -1), xp=xp)
        a = xp_reshape(
            xp.matmul(w_s, r_s_flat) / 24, (w_s.shape[0], *r_s_shape[1:]), xp=xp
        )

        powers = xp_as_float_array([[1], [2], [3], [4], [5]], xp=xp, like=x)
        # Polynomial basis ``t**k`` for k in 1..5; a trailing axis is
        # appended for rank-2 ``y`` so it broadcasts over ``y``'s
        # trailing axis.
        basis = t**powers
        if self._y.ndim == 2:
            basis = basis[..., None]
        return r[i_np] + (a * basis).sum(axis=0)

    def _validate_dimensions(self) -> None:
        """Validate that the variables dimensions are the same."""

        if len(self._x) != len(self._y):
            error = (
                '"x" independent and "y" dependent variables have different '
                f'dimensions: "{len(self._x)}", "{len(self._y)}"'
            )

            raise ValueError(error)

    def _validate_interpolation_range(self, x: NDArrayFloat) -> None:
        """Validate specified point to be in interpolation range."""

        xp = array_namespace(x)

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if bool(xp.any(below_interpolation_range)):
            error = f'"{x}" is below interpolation range.'

            raise ValueError(error)

        if bool(xp.any(above_interpolation_range)):
            error = f'"{x}" is above interpolation range.'

            raise ValueError(error)


class CubicSplineInterpolator(scipy.interpolate.interp1d):
    """
    Perform cubic spline interpolation on one-dimensional data.

    Provide smooth interpolation through specified data points using
    piecewise cubic polynomials. The resulting interpolant maintains
    continuity in the function and its first two derivatives at data
    points, making it suitable for spectral data and colour science
    applications requiring smooth transitions between measured values.

    Methods
    -------
    -   :meth:`~colour.CubicSplineInterpolator.__init__`

    Notes
    -----
    -   This class is a wrapper around *scipy.interpolate.interp1d* class.
    """

    def __init__(self, x: ArrayLike, y: ArrayLike, *args: Any, **kwargs: Any) -> None:
        kwargs["kind"] = "cubic"
        # Interpolate along the leading axis so rank-2 ``y`` evaluates
        # batched along its trailing axis; rank-1 ``y`` is unaffected.
        kwargs.setdefault("axis", 0)
        super().__init__(as_ndarray(x), as_ndarray(y), *args, **kwargs)

    def __call__(self, x: Any) -> Any:
        """Evaluate, converting non-*NumPy* arrays to *NumPy* for *SciPy*."""

        xp = array_namespace(x)
        y = super().__call__(as_ndarray(x))
        if not is_numpy_namespace(xp):
            runtime_warning(
                '"CubicSplineInterpolator" is falling back to "SciPy" for '
                "non-NumPy arrays; the result is converted back to the input "
                "namespace at a performance cost."
            )
            y = xp_as_float_array(y, xp=xp, like=x)
        return y


class PchipInterpolator(scipy.interpolate.PchipInterpolator):
    """
    Interpolate a 1-D function using Piecewise Cubic Hermite Interpolating
    Polynomial (PCHIP) interpolation.

    PCHIP interpolation constructs a smooth curve through specified data
    points while preserving monotonicity between consecutive points. This
    method ensures that the interpolated values do not exhibit spurious
    oscillations, making it particularly suitable for colour science
    applications where physical constraints must be respected.

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

    def __init__(self, x: ArrayLike, y: ArrayLike, *args: Any, **kwargs: Any) -> None:
        x = as_ndarray(x)
        y = as_ndarray(y)
        super().__init__(x, y, *args, **kwargs)

        self._y: NDArrayFloat = y

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate, converting non-*NumPy* arrays to *NumPy* for *SciPy*."""

        xp = array_namespace(x)
        y = super().__call__(as_ndarray(x), *args, **kwargs)
        if not is_numpy_namespace(xp):
            runtime_warning(
                '"PchipInterpolator" is falling back to "SciPy" for non-NumPy '
                "arrays; the result is converted back to the input namespace at "
                "a performance cost."
            )
            y = xp_as_float_array(y, xp=xp, like=x)
        return y

    @property
    def y(self) -> NDArrayFloat:
        """
        Getter and setter for the dependent and already known
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
    def y(self, value: ArrayLike) -> None:
        """Setter for the **self.y** property."""

        self._y = as_float_array(value)


class NullInterpolator:
    """
    Implement 1-D function null interpolation.

    This interpolator returns existing :math:`y` values when called with
    :math:`x` values within specified tolerances, and returns a default
    value when outside tolerances. Unlike traditional interpolators that
    estimate intermediate values, this null interpolator only returns exact
    matches within tolerance bounds.

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
    >>> y = np.array([5.9200, 9.3700, 10.8135, 4.5100, 69.5900, 27.8007, 86.0500])
    >>> x = np.arange(len(y))
    >>> f = NullInterpolator(x, y)
    >>> f(0.5)
    np.float64(nan)
    >>> f(1.0)
    np.float64(9.37)
    >>> f = NullInterpolator(x, y, absolute_tolerance=0.01)
    >>> f(1.01)
    np.float64(9.37)
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        absolute_tolerance: float = TOLERANCE_ABSOLUTE_DEFAULT,
        relative_tolerance: float = TOLERANCE_RELATIVE_DEFAULT,
        default: float = float("nan"),
        dtype: Type[DTypeReal] | None = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

        self._x: NDArrayFloat = np.array([])
        self._y: NDArrayFloat = np.array([])
        self._absolute_tolerance: float = TOLERANCE_ABSOLUTE_DEFAULT
        self._relative_tolerance: float = TOLERANCE_RELATIVE_DEFAULT
        self._default: float = float("nan")
        self._dtype: Type[DTypeReal] = dtype

        self.x = x
        self.y = y
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.default = default

        self._validate_dimensions()

    @property
    def x(self) -> NDArrayFloat:
        """
        Getter and setter for the independent :math:`x` variable.

        Parameters
        ----------
        value
            Value to set the independent :math:`x` variable with.

        Returns
        -------
        :class:`numpy.ndarray`
            Independent :math:`x` variable.

        Raises
        ------
        AssertionError
            If the provided value has not exactly one dimension.
        """

        return self._x

    @x.setter
    def x(self, value: ArrayLike) -> None:
        """Setter for the **self.x** property."""

        xp = array_namespace(value)

        value = xp_astype(xp_atleast_1d(value, xp=xp), self._dtype, xp=xp)

        attest(
            value.ndim == 1,
            '"x" independent variable must have exactly one dimension!',
        )

        self._x = value

    @property
    def y(self) -> NDArrayFloat:
        """
        Getter and setter for the dependent and already known
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

        Raises
        ------
        AssertionError
            If the provided value has not exactly one dimension.
        """

        return self._y

    @y.setter
    def y(self, value: ArrayLike) -> None:
        """Setter for the **self.y** property."""

        xp = array_namespace(value)

        value = xp_astype(xp_atleast_1d(value, xp=xp), self._dtype, xp=xp)

        attest(
            value.ndim in (1, 2),
            '"y" dependent variable must have one or two dimensions!',
        )

        self._y = value

    @property
    def relative_tolerance(self) -> float:
        """
        Getter and setter property for the relative tolerance for numerical
        comparisons.

        Parameters
        ----------
        value
            Value to set the relative tolerance for numerical comparisons with.

        Returns
        -------
        :class:`float`
            Relative tolerance for numerical comparisons.

        Raises
        ------
        AssertionError
            If the value is not numeric.
        """

        return self._relative_tolerance

    @relative_tolerance.setter
    def relative_tolerance(self, value: float) -> None:
        """Setter for the **self.relative_tolerance** property."""

        attest(
            is_numeric(value),
            '"relative_tolerance" variable must be a "numeric"!',
        )

        self._relative_tolerance = as_float_scalar(value)

    @property
    def absolute_tolerance(self) -> float:
        """
        Getter and setter property for the absolute tolerance for numerical
        comparisons.

        Parameters
        ----------
        value
            Value to set the absolute tolerance for numerical comparisons with.

        Returns
        -------
        :class:`float`
            Absolute tolerance for numerical comparisons.

        Raises
        ------
        AssertionError
            If the value is not numeric.
        """

        return self._absolute_tolerance

    @absolute_tolerance.setter
    def absolute_tolerance(self, value: float) -> None:
        """Setter for the **self.absolute_tolerance** property."""

        attest(
            is_numeric(value),
            '"absolute_tolerance" variable must be a "numeric"!',
        )

        self._absolute_tolerance = as_float_scalar(value)

    @property
    def default(self) -> float:
        """
        Getter and setter property for the default value for call outside
        tolerances.

        Parameters
        ----------
        value
            Value to set the default value with for call outside tolerances.

        Returns
        -------
        :class:`float`
            Default value for call outside tolerances.

        Raises
        ------
        AssertionError
            If the value is not numeric.
        """

        return self._default

    @default.setter
    def default(self, value: float) -> None:
        """Setter for the **self.default** property."""

        attest(is_numeric(value), '"default" variable must be a "numeric"!')

        self._default = value

    def __call__(self, x: ArrayLike) -> NDArrayFloat:
        """
        Evaluate the interpolator at specified point(s).

        Parameters
        ----------
        x
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated value(s).
        """

        x = as_float_array(x)

        xi = self._evaluate(x)

        # Collapse the synthetic leading axis for 0-d ``x``; guarded against
        # the rank-1 ``y`` path that already returns a 0-d scalar.
        if x.ndim == 0 and xi.ndim > 0:
            xi = first_item(xi)

        return as_float(xi)

    def _evaluate(self, x: NDArrayFloat) -> NDArrayFloat:
        """
        Perform the interpolator evaluation at specified points.

        Parameters
        ----------
        x
            Points to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated points values.
        """

        xp = array_namespace(x, self._x, self._y)

        x = xp_as_float_array(x, xp=xp)

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        sx = xp_as_float_array(self._x, xp=xp, like=x)
        sy = xp_as_float_array(self._y, xp=xp, like=x)
        indexes = closest_indexes(sx, x)
        # ``close`` is always ``(N_query,)``; a trailing axis is appended
        # for rank-2 ``y`` so it broadcasts over ``y``'s trailing axis.
        values = sy[indexes]
        close = xp_isclose(
            sx[indexes],
            x,
            rtol=self._relative_tolerance,
            atol=self._absolute_tolerance,
            xp=xp,
        )
        if self._y.ndim == 2:
            close = close[..., None]
        values = xp.where(~close, self._default, values)

        return xp_squeeze(values, xp=xp) if self._y.ndim == 1 else values

    def _validate_dimensions(self) -> None:
        """Validate that the variables dimensions are the same."""

        if len(self._x) != len(self._y):
            error = (
                '"x" independent and "y" dependent variables have different '
                f'dimensions: "{len(self._x)}", "{len(self._y)}"'
            )

            raise ValueError(error)

    def _validate_interpolation_range(self, x: NDArrayFloat) -> None:
        """Validate specified point to be in interpolation range."""

        xp = array_namespace(x)

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if bool(xp.any(below_interpolation_range)):
            error = f'"{x}" is below interpolation range.'

            raise ValueError(error)

        if bool(xp.any(above_interpolation_range)):
            error = f'"{x}" is above interpolation range.'

            raise ValueError(error)


def lagrange_coefficients(r: float, n: int = 4) -> NDArrayFloat:
    """
    Compute *Lagrange coefficients* at specified point :math:`r` for
    polynomial interpolation of degree :math:`n`.

    Parameters
    ----------
    r
        Point at which to compute the *Lagrange coefficients*.
    n
        Degree of the polynomial interpolation. The number of coefficients
        returned will be :math:`n + 1`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of *Lagrange coefficients* computed at point :math:`r`.

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
        basis = [(r - r_i[i]) / (r_i[j] - r_i[i]) for i in range(len(r_i)) if i != j]
        L_n.append(reduce(lambda x, y: x * y, basis))

    return np.asarray(L_n)


def table_interpolation_trilinear(V_xyz: ArrayLike, table: ArrayLike) -> NDArrayFloat:
    """
    Perform trilinear interpolation of the specified :math:`V_{xyz}` values using
    the specified interpolation table.

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
    ...     os.path.dirname(__file__),
    ...     "..",
    ...     "io",
    ...     "luts",
    ...     "tests",
    ...     "resources",
    ...     "iridas_cube",
    ...     "Colour_Correct.cube",
    ... )
    >>> LUT = colour.read_LUT(path)
    >>> table = LUT.table
    >>> prng = np.random.RandomState(4)
    >>> V_xyz = colour.algebra.random_triplet_generator(3, random_state=prng)
    >>> print(V_xyz)  # doctest: +ELLIPSIS
    [[0.9670298... 0.7148159... 0.9762744...]
     [0.5472322... 0.6977288... 0.0062302...]
     [0.9726843... 0.2160895... 0.2529823...]]
    >>> table_interpolation_trilinear(V_xyz, table)  # doctest: +ELLIPSIS
    array([[1.0120664..., 0.7539146..., 1.0228540...],
           [0.5075794..., 0.6479459..., 0.1066404...],
           [1.0976519..., 0.1785998..., 0.2299897...]])
    """

    V_xyz = cast("NDArrayFloat", V_xyz)

    xp = array_namespace(V_xyz)

    original_shape = V_xyz.shape

    V_xyz = cast("NDArrayFloat", xp_reshape(xp.clip(V_xyz, 0, 1), (-1, 3), xp=xp))

    # Index computation
    table = xp_as_float_array(table, xp=xp, like=V_xyz)
    i_m = xp_as_int_array(table.shape[:-1], xp=xp, like=V_xyz) - 1
    V_xyz_s = V_xyz * i_m

    i_f = xp_astype(V_xyz_s, DTYPE_INT_DEFAULT, xp=xp)
    i_f = xp.clip(i_f, 0, i_m)
    i_c = xp.minimum(i_f + 1, i_m)

    # Relative coordinates (fractional part)
    frac = V_xyz_s - i_f

    # Extract indices for direct lookup
    fx, fy, fz = i_f[:, 0], i_f[:, 1], i_f[:, 2]
    cx, cy, cz = i_c[:, 0], i_c[:, 1], i_c[:, 2]

    # Extract fractional coordinates
    dx, dy, dz = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]
    dx1, dy1, dz1 = 1.0 - dx, 1.0 - dy, 1.0 - dz

    # Direct vertex lookups (8 corners of cube)
    v000 = table[fx, fy, fz]
    v001 = table[fx, fy, cz]
    v010 = table[fx, cy, fz]
    v011 = table[fx, cy, cz]
    v100 = table[cx, fy, fz]
    v101 = table[cx, fy, cz]
    v110 = table[cx, cy, fz]
    v111 = table[cx, cy, cz]

    # Trilinear interpolation (vectorized)
    result = (
        v000 * (dx1 * dy1 * dz1)
        + v001 * (dx1 * dy1 * dz)
        + v010 * (dx1 * dy * dz1)
        + v011 * (dx1 * dy * dz)
        + v100 * (dx * dy1 * dz1)
        + v101 * (dx * dy1 * dz)
        + v110 * (dx * dy * dz1)
        + v111 * (dx * dy * dz)
    )

    return xp_reshape(result, original_shape, xp=xp)


def table_interpolation_tetrahedral(V_xyz: ArrayLike, table: ArrayLike) -> NDArrayFloat:
    """
    Perform tetrahedral interpolation of the specified :math:`V_{xyz}` values
    using the specified 4-dimensional interpolation table.

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
    ...     os.path.dirname(__file__),
    ...     "..",
    ...     "io",
    ...     "luts",
    ...     "tests",
    ...     "resources",
    ...     "iridas_cube",
    ...     "Colour_Correct.cube",
    ... )
    >>> LUT = colour.read_LUT(path)
    >>> table = LUT.table
    >>> prng = np.random.RandomState(4)
    >>> V_xyz = colour.algebra.random_triplet_generator(3, random_state=prng)
    >>> print(V_xyz)  # doctest: +ELLIPSIS
    [[0.9670298... 0.7148159... 0.9762744...]
     [0.5472322... 0.6977288... 0.0062302...]
     [0.9726843... 0.2160895... 0.2529823...]]
    >>> table_interpolation_tetrahedral(V_xyz, table)  # doctest: +ELLIPSIS
    array([[1.0196197..., 0.7674062..., 1.0311751...],
           [0.5105603..., 0.6466722..., 0.1077296...],
           [1.1178206..., 0.1762039..., 0.2209534...]])
    """

    V_xyz = cast("NDArrayFloat", V_xyz)

    xp = array_namespace(V_xyz)

    original_shape = V_xyz.shape

    V_xyz = cast("NDArrayFloat", xp_reshape(xp.clip(V_xyz, 0, 1), (-1, 3), xp=xp))

    # Index computation
    table = xp_as_float_array(table, xp=xp, like=V_xyz)
    i_m = xp_as_int_array(table.shape[:-1], xp=xp, like=V_xyz) - 1
    V_xyz_s = V_xyz * i_m

    i_f = xp_astype(V_xyz_s, DTYPE_INT_DEFAULT, xp=xp)
    i_f = xp.clip(i_f, 0, i_m)
    i_c = xp.minimum(i_f + 1, i_m)

    # Relative coordinates
    r = V_xyz_s - i_f
    x, y, z = r[:, 0], r[:, 1], r[:, 2]

    # Extract indices for direct lookup
    fx, fy, fz = i_f[:, 0], i_f[:, 1], i_f[:, 2]
    cx, cy, cz = i_c[:, 0], i_c[:, 1], i_c[:, 2]

    # Look up 8 corner vertices
    V000 = table[fx, fy, fz]
    V001 = table[fx, fy, cz]
    V010 = table[fx, cy, fz]
    V011 = table[fx, cy, cz]
    V100 = table[cx, fy, fz]
    V101 = table[cx, fy, cz]
    V110 = table[cx, cy, fz]
    V111 = table[cx, cy, cz]

    # Expand dimensions for broadcasting
    x = x[:, None]
    y = y[:, None]
    z = z[:, None]

    # Tetrahedral interpolation - select tetrahedron based on position
    xyz_o = xp_select(
        [
            xp.logical_and(x > y, y > z),
            xp.logical_and(x > z, z >= y),
            xp.logical_and(z >= x, x > y),
            xp.logical_and(y >= x, x > z),
            xp.logical_and(y >= z, z >= x),
            xp.logical_and(z > y, y >= x),
        ],
        [
            (1 - x) * V000 + (x - y) * V100 + (y - z) * V110 + z * V111,
            (1 - x) * V000 + (x - z) * V100 + (z - y) * V101 + y * V111,
            (1 - z) * V000 + (z - x) * V001 + (x - y) * V101 + y * V111,
            (1 - y) * V000 + (y - x) * V010 + (x - z) * V110 + z * V111,
            (1 - y) * V000 + (y - z) * V010 + (z - x) * V011 + x * V111,
            (1 - z) * V000 + (z - y) * V001 + (y - x) * V011 + x * V111,
        ],
        xp=xp,
    )

    return xp_reshape(xyz_o, original_shape, xp=xp)


TABLE_INTERPOLATION_METHODS = CanonicalMapping(
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
    method: Literal["Trilinear", "Tetrahedral"] | str = "Trilinear",
) -> NDArrayFloat:
    """
    Perform interpolation of the specified :math:`V_{xyz}` values using a
    4-dimensional interpolation table.

    Interpolate the input :math:`V_{xyz}` values through either trilinear
    or tetrahedral interpolation methods using the specified lookup table.

    Parameters
    ----------
    V_xyz
        :math:`V_{xyz}` values to interpolate, where each row represents
        a three-dimensional coordinate within the interpolation table's
        domain.
    table
        4-dimensional (NxNxNx3) interpolation table defining the mapping
        from input coordinates to output values.
    method
        Interpolation method to use. Either "Trilinear" for trilinear
        interpolation or "Tetrahedral" for tetrahedral interpolation.

    Returns
    -------
    :class:`numpy.ndarray`
        Interpolated :math:`V_{xyz}` values with the same shape as the
        input array.

    References
    ----------
    :cite:`Bourkeb`, :cite:`Kirk2006`

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     os.path.dirname(__file__),
    ...     "..",
    ...     "io",
    ...     "luts",
    ...     "tests",
    ...     "resources",
    ...     "iridas_cube",
    ...     "Colour_Correct.cube",
    ... )
    >>> LUT = colour.read_LUT(path)
    >>> table = LUT.table
    >>> prng = np.random.RandomState(4)
    >>> V_xyz = colour.algebra.random_triplet_generator(3, random_state=prng)
    >>> print(V_xyz)  # doctest: +ELLIPSIS
    [[0.9670298... 0.7148159... 0.9762744...]
     [0.5472322... 0.6977288... 0.0062302...]
     [0.9726843... 0.2160895... 0.2529823...]]
    >>> table_interpolation(V_xyz, table)  # doctest: +ELLIPSIS
    array([[1.0120664..., 0.7539146..., 1.0228540...],
           [0.5075794..., 0.6479459..., 0.1066404...],
           [1.0976519..., 0.1785998..., 0.2299897...]])
    >>> table_interpolation(V_xyz, table, method="Tetrahedral")
    ... # doctest: +ELLIPSIS
    array([[1.0196197..., 0.7674062..., 1.0311751...],
           [0.5105603..., 0.6466722..., 0.1077296...],
           [1.1178206..., 0.1762039..., 0.2209534...]])
    """

    method = validate_method(method, tuple(TABLE_INTERPOLATION_METHODS))

    return TABLE_INTERPOLATION_METHODS[method](V_xyz, table)


def linear_interpolation_index_and_factor(
    value: ArrayLike, break_points: ArrayLike
) -> tuple[NDArrayReal, NDArrayFloat]:
    """
    Compute the bin index and fractional position for piecewise linear
    interpolation of *value* within sorted *break_points*.

    For each element in *value*, the returned ``index`` identifies the
    interval ``[break_points[index], break_points[index + 1])`` that
    contains the value, and ``factor`` gives the normalised position
    within that interval (0 at the left edge, 1 at the right).

    Values outside the range of *break_points* are clamped.

    Parameters
    ----------
    value
        Query value(s), scalar or array.
    break_points
        Sorted array of break points defining the piecewise linear
        intervals.

    Returns
    -------
    :class:`tuple`
        Tuple of ``(index, factor)`` arrays with the same shape as
        *value*.

    Examples
    --------
    >>> break_points = np.array([0.0, 1.0, 2.0, 3.0])
    >>> linear_interpolation_index_and_factor(1.5, break_points)
    ... # doctest: +ELLIPSIS
    (array(1...), array(0.5))
    >>> linear_interpolation_index_and_factor(  # doctest: +ELLIPSIS
    ...     np.array([0.0, 0.5, 2.5, 3.0]), break_points
    ... )
    (array([0, 0, 2, 3]...), array([0. , 0.5, 0.5, 0. ]))
    """

    xp = array_namespace(value, break_points)

    value = xp_as_float_array(value, xp=xp, like=break_points)
    break_points = xp_as_float_array(break_points, xp=xp, like=value)

    clamped = xp.clip(value, break_points[0], break_points[-1])

    # Upper bound search starting from break_points[1].
    next_idx = (
        xp_reshape(
            xp.searchsorted(
                break_points[1:],
                xp_reshape(clamped, (-1,), xp=xp),
                side="right",
            ),
            clamped.shape,
            xp=xp,
        )
        + 1
    )

    at_end = next_idx >= len(break_points)
    index = xp.where(at_end, len(break_points) - 1, next_idx - 1)

    safe_next = xp.clip(next_idx, max=len(break_points) - 1)
    denominator = break_points[safe_next] - break_points[index]
    factor = xp.where(
        at_end | (denominator == 0),
        0.0,
        (clamped - break_points[index]) / xp.where(denominator == 0, 1.0, denominator),
    )

    return xp_astype(index, DTYPE_INT_DEFAULT, xp=xp), factor
