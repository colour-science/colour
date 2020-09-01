# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import colour.ndarray as np

from colour.utilities import as_float, as_float_array, suppress_warnings

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['exponent_function_basic', 'exponent_function_monitor_curve']


def exponent_function_basic(x, exponent=1, style='basicFwd'):
    """
    Defines the *basic* exponent transfer function.

    Parameters
    ----------
    x : numeric or array_like
        Data to undergo the basic exponent conversion.
    exponent : numeric or array_like, optional
        Exponent value used for the conversion.
    style : unicode, optional
        **{'basicFwd', 'basicRev', 'basicMirrorFwd', 'basicMirrorRev',
        'basicPassThruFwd', 'basicPassThruRev'}**,
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
    numeric or ndarray
        Exponentially converted data.

    Raises
    ------
    ValueError
        If the *style* is not defined.

    Examples
    --------
    >>> exponent_function_basic(0.18, 2.2)  # doctest: +ELLIPSIS
    0.0229932...
    >>> exponent_function_basic(-0.18, 2.2)
    0.0
    >>> exponent_function_basic(0.18, 2.2, 'basicRev')  # doctest: +ELLIPSIS
    0.4586564...
    >>> exponent_function_basic(-0.18, 2.2, 'basicRev')
    0.0
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 'basicMirrorFwd')
    0.0229932...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 'basicMirrorFwd')
    -0.0229932...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 'basicMirrorRev')
    0.4586564...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 'basicMirrorRev')
    -0.4586564...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 'basicPassThruFwd')
    0.0229932...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 'basicPassThruFwd')
    -0.1799999...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 'basicPassThruRev')
    0.4586564...
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 'basicPassThruRev')
    -0.1799999...
    """

    x = as_float_array(x)
    exponent = as_float_array(exponent)

    def exponent_forward(x):
        """
        Returns the input raised to the exponent value.
        """

        return x ** exponent

    def exponent_reverse(y):
        """
        Returns the input raised to the inverse exponent value.
        """

        return y ** (1 / exponent)

    style = style.lower()
    if style == 'basicfwd':
        return as_float(np.where(x >= 0, exponent_forward(x), 0))
    elif style == 'basicrev':
        return as_float(np.where(x >= 0, exponent_reverse(x), 0))
    elif style == 'basicmirrorfwd':
        return as_float(
            np.where(x >= 0, exponent_forward(x), -exponent_forward(-x)))
    elif style == 'basicmirrorrev':
        return as_float(
            np.where(x >= 0, exponent_reverse(x), -exponent_reverse(-x)))
    elif style == 'basicpassthrufwd':
        return as_float(np.where(x >= 0, exponent_forward(x), x))
    elif style == 'basicpassthrurev':
        return as_float(np.where(x >= 0, exponent_reverse(x), x))
    else:
        raise ValueError(
            'Undefined style used: "{0}", must be one of the following: '
            '"{1}".'.format(
                style, ', '.join([
                    'basicFwd', 'basicRev', 'basicMirrorFwd', 'basicMirrorRev',
                    'basicPassThruFwd', 'basicPassThruRev'
                ])))


def exponent_function_monitor_curve(x,
                                    exponent=1,
                                    offset=0,
                                    style='monCurveFwd'):
    """
    Defines the *Monitor Curve* exponent transfer function.

    Parameters
    ----------
    x : numeric or array_like
        Data to undergo the monitor curve exponential conversion.
    exponent : numeric or array_like, optional
        Exponent value used for the conversion.
    offset: numeric or array_like, optional
        Offset value used for the conversion.
    style : unicode, optional
        **{'monCurveFwd', 'monCurveRev', 'monCurveMirrorFwd',
        'monCurveMirrorRev'}**,
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
    numeric or ndarray
        Exponentially converted data.

    Raises
    ------
    ValueError
        If the *style* is not defined.

    Examples
    --------
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 0.001)
    0.0232240...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 0.001)
    -0.0002054...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 0.001, 'monCurveRev')
    0.4581151...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 0.001, 'monCurveRev')
    -157.7302795...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 2, 'monCurveMirrorFwd')
    0.1679399...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 0.001, 'monCurveMirrorFwd')
    -0.0232240...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 0.001, 'monCurveMirrorRev')
    0.4581151...
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 0.001, 'monCurveMirrorRev')
    -0.4581151...
    """

    x = as_float_array(x)
    exponent = as_float_array(exponent)
    offset = as_float_array(offset)

    with suppress_warnings(python_warnings=True):
        s = as_float_array(((exponent - 1) / offset) * ((exponent * offset) / (
            (exponent - 1) * (offset + 1))) ** exponent)

        s[np.isnan(s)] = 1

    def monitor_curve_forward(x):
        """
        Defines the *Monitor Curve Forward* function.
        """

        x_break = offset / (exponent - 1)

        return np.where(
            x >= x_break,
            ((x + offset) / (1 + offset)) ** exponent,
            x * s,
        )

    def monitor_curve_reverse(y):
        """
        Defines the *Monitor Curve Reverse* function.
        """

        y_break = ((exponent * offset) / (
            (exponent - 1) * (1 + offset))) ** exponent

        return np.where(
            y >= y_break,
            ((1 + offset) * (y ** (1 / exponent))) - offset,
            y / s,
        )

    style = style.lower()
    if style == 'moncurvefwd':
        return as_float(monitor_curve_forward(x))
    elif style == 'moncurverev':
        return as_float(monitor_curve_reverse(x))
    elif style == 'moncurvemirrorfwd':
        return as_float(
            np.where(
                x >= 0,
                monitor_curve_forward(x),
                -monitor_curve_forward(-x),
            ))
    elif style == 'moncurvemirrorrev':
        return as_float(
            np.where(
                x >= 0,
                monitor_curve_reverse(x),
                -monitor_curve_reverse(-x),
            ))
    else:
        raise ValueError(
            'Undefined style used: "{0}", must be one of the following: '
            '"{1}".'.format(
                style, ', '.join([
                    'monCurveFwd', 'monCurveRev', 'monCurveMirrorFwd',
                    'monCurveMirrorRev'
                ])))
