#
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

import numpy as np

from colour.utilities import as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['exponent_function_basic', 'exponent_function_monitor_curve']


def exponent_function_basic(x, exponent, style='basicFwd'):
    """
    Defines the *basic* exponent transfer function.

    Parameters
    ----------
    x : numeric or array_like
        Data to undergo the basic exponent conversion.
    exponent : numeric or array_like
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
    >>> exponent_function_basic(2, 2)
    array(4.0)
    >>> exponent_function_basic(-2, 2)
    array(0.0)
    >>> exponent_function_basic(4, 2, 'basicRev')
    array(2.0)
    >>> exponent_function_basic(-4, 2, 'basicRev')
    array(0.0)
    >>> exponent_function_basic(2, 2, 'basicMirrorFwd')
    array(4.0)
    >>> exponent_function_basic(-2, 2, 'basicMirrorFwd')
    array(-4.0)
    >>> exponent_function_basic(4, 2, 'basicMirrorRev')
    array(2.0)
    >>> exponent_function_basic(-4, 2, 'basicMirrorRev')
    array(-2.0)
    >>> exponent_function_basic(2, 2, 'basicPassThruFwd')
    array(4.0)
    >>> exponent_function_basic(-2, 2, 'basicPassThruFwd')
    array(-2.0)
    >>> exponent_function_basic(4, 2, 'basicPassThruRev')
    array(2.0)
    >>> exponent_function_basic(-4, 2, 'basicPassThruRev')
    array(-4.0)
    """

    x = as_float_array(x)
    exponent = as_float_array(exponent)

    def exponent_forward(x):
        """
        Returns the input raised to the exponent value.
        """

        y = x ** exponent
        return (y)

    def exponent_reverse(y):
        """
        Returns the input raised to the inverse exponent value.
        """

        return y ** (1 / exponent)

    style = style.lower()
    if style == 'basicfwd':
        return np.where(x > 0, exponent_forward(x), 0)
    elif style == 'basicrev':
        return np.where(x > 0, exponent_reverse(x), 0)
    elif style == 'basicmirrorfwd':
        return np.where(x >= 0, exponent_forward(x), -exponent_forward(-x))
    elif style == 'basicmirrorrev':
        return np.where(x >= 0, exponent_reverse(x), -exponent_reverse(-x))
    elif style == 'basicpassthrufwd':
        return np.where(x > 0, exponent_forward(x), x)
    elif style == 'basicpassthrurev':
        return np.where(x > 0, exponent_reverse(x), x)
    else:
        raise ValueError(
            'Undefined style used: "{0}", must be one of the following: '
            '"{1}".'.format(
                style, ', '.join([
                    'basicFwd', 'basicRev', 'basicMirrorFwd', 'basicMirrorRev',
                    'basicPassThruFwd', 'basicPassThruRev'
                ])))


def exponent_function_monitor_curve(x, exponent, offset, style='monCurveFwd'):
    """
    Defines the *Monitor Curve* exponent transfer function.

    Parameters
    ----------
    x : numeric or array_like
        Data to undergo the monitor curve exponential conversion.
    exponent : numeric or array_like
        Exponent value used for the conversion.
    offset: numeric or array_like
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
    >>> exponent_function_monitor_curve(2, 2, 2)
    array(1.7777777777777777)
    >>> exponent_function_monitor_curve(-2, 2, 2)
    array(-1.7777777777777777)
    >>> exponent_function_monitor_curve(\
            1.7777777777777777, 2, 2, 'monCurveRev')  # doctest: +ELLIPSIS
    array(2.0)
    >>> exponent_function_monitor_curve(\
            -1.7777777777777777, 2, 2, 'monCurveRev')  # doctest: +ELLIPSIS
    array(-2.0)
    >>> exponent_function_monitor_curve(\
            2, 2, 2, 'monCurveMirrorFwd')
    array(1.7777777777777777)
    >>> exponent_function_monitor_curve(\
            -2, 2, 2, 'monCurveMirrorFwd')
    array(-1.7777777777777777)
    >>> exponent_function_monitor_curve(\
            1.7777777777777777, 2, 2,\
            'monCurveMirrorRev')  # doctest: +ELLIPSIS
    array(2.0)
    >>> exponent_function_monitor_curve(\
            -1.7777777777777777, 2, 2,\
            'monCurveMirrorRev')  # doctest: +ELLIPSIS
    array(-2.0)
    """

    x = as_float_array(x)
    exponent = as_float_array(exponent)
    offset = as_float_array(offset)
    if offset == 0.0:
        print(
            "Setting the offest value as '0' can lead to a divide-by-zero error.\
                Try again!")
        return None
    if exponent == 1.0:
        print(
            "Setting the exponent value as '1' can lead to a divide-by-zero error.\
                Try again!")
        return None
    s = ((exponent - 1) / offset) * ((exponent * offset) / (
        (exponent - 1) * (offset + 1))) ** exponent

    def monitor_curve_forward(x):
        """
        Defines the *Monitor Curve Forward* function.
        """

        xBreak = offset / (exponent - 1)
        y = ((x + offset) / (1 + offset)) ** exponent
        return np.where(x >= xBreak, y, x * s)

    def monitor_curve_reverse(y):
        """
        Defines the *Monitor Curve Reverse* function.
        """

        yBreak = ((exponent * offset) / (
            (exponent - 1) * (offset + 1))) ** exponent
        return np.where(y >= yBreak,
                        ((1 + offset) * (y ** (1 / exponent))) - offset, y / s)

    style = style.lower()
    if style == 'moncurvefwd':
        return monitor_curve_forward(x)
    elif style == 'moncurverev':
        return monitor_curve_reverse(x)
    elif style == 'moncurvemirrorfwd':
        return np.where(x >= 0, monitor_curve_forward(x),
                        -monitor_curve_forward(-x))
    elif style == 'moncurvemirrorrev':
        return np.where(x >= 0, monitor_curve_reverse(x),
                        -monitor_curve_reverse(-x))
    else:
        raise ValueError(
            'Undefined style used: "{0}", must be one of the following: '
            '"{1}".'.format(
                style, ', '.join([
                    'monCurveFwd', 'monCurveRev', 'monCurveMirrorFwd',
                    'monCurveMirrorRev'
                ])))
