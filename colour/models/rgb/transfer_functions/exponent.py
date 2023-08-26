"""
Basic and Monitor-Curve Exponent Transfer Functions
===================================================

Defines the exponent transfer functions:

-   :func:`colour.models.exponent_function_basic`
-   :func:`colour.models.exponent_function_monitor_curve`

References
----------
-   :cite: `TheAcademyofMotionPictureArtsandSciences2020` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2020). Specification
    S-2014-006 - Common LUT Format (CLF) - A Common File Format for Look-Up
    Tables. Retrieved June 24, 2020, from http://j.mp/S-2014-006
"""

from __future__ import annotations


from colour.algebra import sdiv, sdiv_mode
from colour.hints import ArrayLike, Literal, NDArrayFloat
from colour.utilities import (
    as_float,
    as_float_array,
    validate_method,
    zeros,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "exponent_function_basic",
    "exponent_function_monitor_curve",
]


def exponent_function_basic(
    x: ArrayLike,
    exponent: ArrayLike = 1,
    style: Literal[
        "basicFwd",
        "basicRev",
        "basicMirrorFwd",
        "basicMirrorRev",
        "basicPassThruFwd",
        "basicPassThruRev",
    ]
    | str = "basicFwd",
) -> NDArrayFloat:
    """
    Define the *basic* exponent transfer function.

    Parameters
    ----------
    x
        Data to undergo the basic exponent conversion.
    exponent
        Exponent value used for the conversion.
    style
        Defines the behaviour for the transfer function to operate:

        -   *basicFwd*: *Basic Forward* exponential behaviour where the
            definition applies a basic power law using the exponent. Values
            less than zero are clamped.
        -   *basicRev*: *Basic Reverse* exponential behaviour where the
            definition applies a basic power law using the exponent. Values
            less than zero are clamped.
        -   *basicMirrorFwd*: *Basic Mirror Forward* exponential behaviour
            where the definition applies a basic power law using the exponent
            for values greater than or equal to zero and mirrors the function
            for values less than zero (i.e. rotationally symmetric
            around the origin).
        -   *basicMirrorRev*: *Basic Mirror Reverse* exponential behaviour
            where the definition applies a basic power law using the exponent
            for values greater than or equal to zero and mirrors the function
            for values less than zero (i.e. rotationally symmetric around the
            origin).
        -   *basicPassThruFwd*: *Basic Pass Forward* exponential behaviour
            where the definition applies a basic power law using the exponent
            for values greater than or equal to zero and passes values less
            than zero unchanged.
        -   *basicPassThruRev*: *Basic Pass Reverse* exponential behaviour
            where the definition applies a basic power law using the exponent
            for values greater than or equal to zero and passes values less
            than zero unchanged.

    Returns
    -------
    :class:`numpy.ndarray`
        Exponentially converted data.

    Examples
    --------
    >>> exponent_function_basic(0.18, 2.2)  # doctest: +ELLIPSIS
    0.0229932...
    >>> exponent_function_basic(-0.18, 2.2)
    0.0
    >>> exponent_function_basic(0.18, 2.2, "basicRev")  # doctest: +ELLIPSIS
    0.4586564...
    >>> exponent_function_basic(-0.18, 2.2, "basicRev")
    0.0
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, "basicMirrorFwd"
    ... )
    0.0229932...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, "basicMirrorFwd"
    ... )
    -0.0229932...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, "basicMirrorRev"
    ... )
    0.4586564...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, "basicMirrorRev"
    ... )
    -0.4586564...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, "basicPassThruFwd"
    ... )
    0.0229932...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, "basicPassThruFwd"
    ... )
    -0.1799999...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, "basicPassThruRev"
    ... )
    0.4586564...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, "basicPassThruRev"
    ... )
    -0.1799999...
    """

    x = as_float_array(x)
    exponent = as_float_array(exponent)
    style = validate_method(
        style,
        (
            "basicFwd",
            "basicRev",
            "basicMirrorFwd",
            "basicMirrorRev",
            "basicPassThruFwd",
            "basicPassThruRev",
        ),
        '"{0}" style is invalid, it must be one of {1}!',
    )

    def exponent_forward(x: NDArrayFloat) -> NDArrayFloat:
        """Return the input raised to the exponent value."""

        return x**exponent

    def exponent_reverse(y: NDArrayFloat) -> NDArrayFloat:
        """Return the input raised to the inverse exponent value."""

        return y ** (as_float_array(1) / exponent)

    y = zeros(x.shape)
    m_x = x >= 0
    if style == "basicfwd":
        y[m_x] = exponent_forward(x[m_x])
    elif style == "basicrev":
        y[m_x] = exponent_reverse(x[m_x])
    elif style == "basicmirrorfwd":
        y[m_x] = exponent_forward(x[m_x])
        y[~m_x] = -exponent_forward(-x[~m_x])
    elif style == "basicmirrorrev":
        y[m_x] = exponent_reverse(x[m_x])
        y[~m_x] = -exponent_reverse(-x[~m_x])
    elif style == "basicpassthrufwd":
        y[m_x] = exponent_forward(x[m_x])
        y[~m_x] = x[~m_x]
    else:  # style == 'basicpassthrurev'
        y[m_x] = exponent_reverse(x[m_x])
        y[~m_x] = x[~m_x]

    return as_float(y)


def exponent_function_monitor_curve(
    x: ArrayLike,
    exponent: ArrayLike = 1,
    offset: ArrayLike = 0,
    style: Literal[
        "monCurveFwd", "monCurveRev", "monCurveMirrorFwd", "monCurveMirrorRev"
    ]
    | str = "monCurveFwd",
) -> NDArrayFloat:
    """
    Define the *Monitor Curve* exponent transfer function.

    Parameters
    ----------
    x
        Data to undergo the monitor curve exponential conversion.
    exponent
        Exponent value used for the conversion.
    offset
        Offset value used for the conversion.
    style
        Defines the behaviour for the transfer function to operate:

        -   *monCurveFwd*: *Monitor Curve Forward* exponential behaviour
            where the definition applies a power law function with a linear
            segment near the origin.
        -   *monCurveRev*: *Monitor Curve Reverse* exponential behaviour
            where the definition applies a power law function with a linear
            segment near the origin.
        -   *monCurveMirrorFwd*: *Monitor Curve Mirror Forward* exponential
            behaviour where the definition applies a power law function with a
            linear segment near the origin and mirrors the function for values
            less than zero (i.e. rotationally symmetric around the origin).
        -   *monCurveMirrorRev*: *Monitor Curve Mirror Reverse* exponential
            behaviour where the definition applies a power law function with a
            linear segment near the origin and mirrors the function for values
            less than zero (i.e. rotationally symmetric around the origin).

    Returns
    -------
    :class:`numpy.ndarray`
        Exponentially converted data.

    Examples
    --------
    >>> exponent_function_monitor_curve(0.18, 2.2, 0.001)  # doctest: +ELLIPSIS
    0.0232240...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 0.001
    ... )
    -0.0002054...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 0.001, "monCurveRev"
    ... )
    0.4581151...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 0.001, "monCurveRev"
    ... )
    -157.7302795...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 2, "monCurveMirrorFwd"
    ... )
    0.1679399...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 0.001, "monCurveMirrorFwd"
    ... )
    -0.0232240...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 0.001, "monCurveMirrorRev"
    ... )
    0.4581151...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 0.001, "monCurveMirrorRev"
    ... )
    -0.4581151...
    """

    x = as_float_array(x)
    exponent = as_float_array(exponent)
    offset = as_float_array(offset)
    style = validate_method(
        style,
        (
            "monCurveFwd",
            "monCurveRev",
            "monCurveMirrorFwd",
            "monCurveMirrorRev",
        ),
        '"{0}" style is invalid, it must be one of {1}!',
    )

    with sdiv_mode():
        s = as_float_array(
            sdiv(exponent - 1, offset)
            * sdiv(exponent * offset, (exponent - 1) * (offset + 1))
            ** exponent
        )

    def monitor_curve_forward(
        x: NDArrayFloat, offset: NDArrayFloat, exponent: NDArrayFloat
    ) -> NDArrayFloat:
        """Define the *Monitor Curve Forward* function."""

        with sdiv_mode():
            x_break = sdiv(offset, exponent - 1)

        y = as_float_array(x * s)

        y[x >= x_break] = (
            (x[x >= x_break] + offset) / (1 + offset)
        ) ** exponent

        return y

    def monitor_curve_reverse(
        y: NDArrayFloat, offset: NDArrayFloat, exponent: NDArrayFloat
    ) -> NDArrayFloat:
        """Define the *Monitor Curve Reverse* function."""

        with sdiv_mode():
            y_break = (
                sdiv(exponent * offset, (exponent - 1) * (1 + offset))
            ) ** exponent

            x = as_float_array(y / s)

        x[y >= y_break] = (
            (1 + offset) * (y[y >= y_break] ** (1 / exponent))
        ) - offset

        return x

    y = zeros(x.shape)
    m_x = x >= 0
    if style == "moncurvefwd":
        y = monitor_curve_forward(x, offset, exponent)
    elif style == "moncurverev":
        y = monitor_curve_reverse(x, offset, exponent)
    elif style == "moncurvemirrorfwd":
        y[m_x] = monitor_curve_forward(x[m_x], offset, exponent)
        y[~m_x] = -monitor_curve_forward(-x[~m_x], offset, exponent)
    else:  # style == 'moncurvemirrorrev'
        y[m_x] = monitor_curve_reverse(x[m_x], offset, exponent)
        y[~m_x] = -monitor_curve_reverse(-x[~m_x], offset, exponent)

    return as_float(y)
