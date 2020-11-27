# -*- coding: utf-8 -*-
"""
Common Log Encodings
====================

Defines the common log encodings:

-   :func:`colour.models.logarithmic_function_basic`
-   :func:`colour.models.logarithmic_function_quasilog`
-   :func:`colour.models.logarithmic_function_camera`
-   :func:`colour.models.log_encoding_Log2`
-   :func:`colour.models.log_decoding_Log2`

References
----------
-   :cite:`TheAcademyofMotionPictureArtsandSciencesa` :
    The Academy of Motion Picture Arts and Sciences,
    Science and Technology Council,
    & Academy Color Encoding System (ACES) Project Subcommittee.(n.d.).
    ACESutil.Lin_to_Log2_param.ctl. Retrieved June 14, 2020,
    from https://github.com/ampas/aces-dev/blob/\
518c27f577e99cdecfddf2ebcfaa53444b1f9343/transforms/ctl/utilities/\
ACESutil.Lin_to_Log2_param.ctl
-   :cite:`TheAcademyofMotionPictureArtsandSciencesb` :
    The Academy of Motion Picture Arts and Sciences,
    Science and Technology Council,
    & Academy Color Encoding System (ACES) Project Subcommittee.(n.d.).
    ACESutil.Log2_to_Lin_param.ctl. Retrieved June 14, 2020,
    from https://github.com/ampas/aces-dev/blob/\
518c27f577e99cdecfddf2ebcfaa53444b1f9343/transforms/ctl/utilities/\
ACESutil.Log2_to_Lin_param.ctl
:   cite: `TheAcademyofMotionPictureArtsandSciences2020` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2020). Specification
    S-2014-006 - Common LUT Format (CLF) - A Common File Format for Look-Up
    Tables. Retrieved June 24, 2020, from http://j.mp/S-2014-006
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import (as_float, as_float_array, from_range_1,
                              to_domain_1)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'logarithmic_function_basic', 'logarithmic_function_quasilog',
    'logarithmic_function_camera', 'log_encoding_Log2', 'log_decoding_Log2'
]

FLT_MIN = 1.175494e-38


def logarithmic_function_basic(x, style='log2', base=2):
    """
    Defines the basic logarithmic function.

    Parameters
    ----------
    x : numeric
        The data to undergo basic logarithmic conversion.
    style : unicode, optional
        **{'log10', 'antiLog10', 'log2', 'antiLog2', 'logB', 'antiLogB'}**,
        Defines the behaviour for the logarithmic function to operate:

        -   *log10*: Applies a base 10 logarithm to the passed value.
        -   *antiLog10*: Applies a base 10 anti-logarithm to the passed value.
        -   *log2*: Applies a base 2 logarithm to the passed value.
        -   *antiLog2*: Applies a base 2 anti-logarithm to the passed value.
        -   *logB*: Applies an arbitrary base logarithm to the passed value.
        -   *antiLogB*: Applies an arbitrary base anti-logarithm to the passed
            value.
    base : numeric, optional
        Logarithmic base used for the conversion.

    Returns
    -------
    numeric or ndarray
        Logarithmically converted data.

    Raises
    ------
    ValueError
        If the *style* is not defined.

    Examples
    --------
    The basic logarithmic function *styles* operate as follows:

    >>> logarithmic_function_basic(0.18)  # doctest: +ELLIPSIS
    -2.4739311...
    >>> logarithmic_function_basic(0.18, 'log10')  # doctest: +ELLIPSIS
    -0.7447274...
    >>> logarithmic_function_basic(  # doctest: +ELLIPSIS
    ...    0.18, 'logB', 3)
    -1.5608767...
    >>> logarithmic_function_basic(  # doctest: +ELLIPSIS
    ...    -2.473931188332412, 'antiLog2')
    0.18000000...
    >>> logarithmic_function_basic(  # doctest: +ELLIPSIS
    ...    -0.7447274948966939, 'antiLog10')
    0.18000000...
    >>> logarithmic_function_basic(  # doctest: +ELLIPSIS
    ...    -1.5608767950073117, 'antiLogB', 3)
    0.18000000...
    """

    x = as_float_array(x)

    style = style.lower()
    if style == 'log10':
        return as_float(np.where(x >= FLT_MIN, np.log10(x), np.log10(FLT_MIN)))
    elif style == 'antilog10':
        return as_float(10 ** x)
    elif style == 'log2':
        return as_float(np.where(x >= FLT_MIN, np.log2(x), np.log2(FLT_MIN)))
    elif style == 'antilog2':
        return as_float(2 ** x)
    elif style == 'logb':
        return as_float(np.log(x) / np.log(base))
    elif style == 'antilogb':
        return as_float(base ** x)
    else:
        raise ValueError(
            'Undefined style used: "{0}", must be one of the following: '
            '"{1}".'.format(
                style, ', '.join([
                    'log10', 'antiLog10', 'log2', 'antiLog2', 'logB',
                    'antiLogB'
                ])))


def logarithmic_function_quasilog(x,
                                  style='linToLog',
                                  base=2,
                                  log_side_slope=1,
                                  lin_side_slope=1,
                                  log_side_offset=0,
                                  lin_side_offset=0):
    """
    Defines the quasilog logarithmic function.

    Parameters
    ----------
    x : numeric
        Linear/non-linear data to undergo encoding/decoding.
    style : unicode, optional
        **{'linToLog', 'logToLin'}**,
        Defines the behaviour for the logarithmic function to operate:

        -   *linToLog*: Applies a logarithm to convert linear data to
            logarithmic data.
        -   *logToLin*: Applies an anti-logarithm to convert logarithmic
            data to linear data.
    base : numeric, optional
        Logarithmic base used for the conversion.
    log_side_slope : numeric, optional
        Slope (or gain) applied to the log side of the logarithmic function.
        The default value is 1.
    lin_side_slope : numeric, optional
        Slope of the linear side of the logarithmic function. The default value
        is 1.
    log_side_offset : numeric, optional
        Offset applied to the log side of the logarithmic function. The default
        value is 0.
    lin_side_offset : numeric, optional
        Offset applied to the linear side of the logarithmic function. The
        default value is 0.

    Returns
    -------
    numeric or ndarray
        Encoded/Decoded data.

    Raises
    ------
    ValueError
        If the *style* is not defined.

    Examples
    --------
    >>> logarithmic_function_quasilog(  # doctest: +ELLIPSIS
    ...    0.18, 'linToLog')
    -2.4739311...
    >>> logarithmic_function_quasilog(  # doctest: +ELLIPSIS
    ...    -2.473931188332412, 'logToLin')
    0.18000000...
    """

    x = as_float_array(x)

    style = style.lower()
    if style == 'lintolog':
        return as_float((
            log_side_slope *
            (np.log(np.maximum(lin_side_slope * x + lin_side_offset, FLT_MIN))
             / np.log(base)) + log_side_offset))
    elif style == 'logtolin':
        return as_float(
            ((base **
              ((x - log_side_offset) / log_side_slope) - lin_side_offset) /
             lin_side_slope))
    else:
        raise ValueError(
            'Undefined style used: "{0}", must be one of the following: '
            '"{1}".'.format(style, ', '.join(['linToLog', 'logToLin'])))


def logarithmic_function_camera(x,
                                style='cameraLinToLog',
                                base=2,
                                log_side_slope=1,
                                lin_side_slope=1,
                                log_side_offset=0,
                                lin_side_offset=0,
                                lin_side_break=0.005,
                                linear_slope=None):
    """
    Defines the camera logarithmic function.

    Parameters
    ----------
    x : numeric
        Linear/non-linear data to undergo encoding/decoding.
    style : unicode, optional
        **{'cameraLinToLog', 'cameraLogToLin'}**,
        Defines the behaviour for the logarithmic function to operate:

        -   *cameraLinToLog*: Applies a piece-wise function with logarithmic
            and linear segments on linear values, converting them to non-linear
            values.
        -   *cameraLogToLin*: Applies a piece-wise function with logarithmic
            and linear segments on non-linear values, converting them to linear
            values.
    base : numeric, optional
        Logarithmic base used for the conversion.
    log_side_slope : numeric, optional
        Slope (or gain) applied to the log side of the logarithmic segment. The
        default value is 1.
    lin_side_slope : numeric, optional
        Slope of the linear side of the logarithmic segment. The default value
        is 1.
    log_side_offset : numeric, optional
        Offset applied to the log side of the logarithmic segment. The default
        value is 0.
    lin_side_offset : numeric, optional
        Offset applied to the linear side of the logarithmic segment. The
        default value is 0.
    lin_side_break : numeric
        Break-point, defined in linear space, at which the piece-wise function
        transitions between the logarithmic and linear segments.
    linear_slope : numeric, optional
        Slope of the linear portion of the curve. The default value is *None*.

    Returns
    -------
    numeric or ndarray
        Encoded/Decoded data.

    Raises
    ------
    ValueError
        If the *style* is not defined.

    Examples
    --------
    >>> logarithmic_function_camera(  # doctest: +ELLIPSIS
    ...    0.18, 'cameraLinToLog')
    -2.4739311...
    >>> logarithmic_function_camera(  # doctest: +ELLIPSIS
    ...    -2.4739311883324122, 'cameraLogToLin')
    0.1800000...
    """

    x = as_float_array(x)

    log_side_break = (
        log_side_slope *
        (np.log(lin_side_slope * lin_side_break + lin_side_offset) /
         np.log(base)) + log_side_offset)

    if linear_slope is None:
        linear_slope = (log_side_slope * (lin_side_slope / (
            (lin_side_slope * lin_side_break + lin_side_offset) * np.log(base))
                                          ))

    linear_offset = log_side_break - linear_slope * lin_side_break

    style = style.lower()
    if style == 'cameralintolog':
        return as_float(
            np.where(
                x <= lin_side_break, linear_slope * x + linear_offset,
                logarithmic_function_quasilog(
                    x, 'linToLog', base, log_side_slope, lin_side_slope,
                    log_side_offset, lin_side_offset)))
    elif style == 'cameralogtolin':
        return as_float(
            np.where(
                x <= log_side_break,
                (x - linear_offset) / linear_slope,
                logarithmic_function_quasilog(
                    x, 'logToLin', base, log_side_slope, lin_side_slope,
                    log_side_offset, lin_side_offset),
            ))
    else:
        raise ValueError(
            'Undefined style used: "{0}", must be one of the following: '
            '"{1}".'.format(style,
                            ', '.join(['cameraLinToLog', 'cameraLogToLin'])))


def log_encoding_Log2(lin,
                      middle_grey=0.18,
                      min_exposure=-6.5,
                      max_exposure=6.5):
    """
    Defines the common *Log2* encoding function.

    Parameters
    ----------
    lin : numeric or array_like
          Linear data to undergo encoding.
    middle_grey : numeric, optional
          *Middle Grey* exposure value.
    min_exposure : numeric, optional
          Minimum exposure level.
    max_exposure : numeric, optional
          Maximum exposure level.

    Returns
    -------
    numeric or ndarray
        Non-linear *Log2* encoded data.

    Notes
    -----
    -   The common *Log2* encoding function can be used to build linear to
        logarithmic shapers in the *ACES OCIO configuration*.
    -   A (48-nits OCIO) shaper having values in a linear domain, can be
        encoded to a logarithmic domain:

        +-------------------+-------------------+
        | **Shaper Domain** | **Shaper Range**  |
        +===================+===================+
        | [0.002, 16.291]   | [0, 1]            |
        +-------------------+-------------------+

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciencesa`

    Examples
    --------
    >>> log_encoding_Log2(0.18)
    0.5
    """

    lin = to_domain_1(lin)

    lg2 = np.log2(lin / middle_grey)
    log_norm = (lg2 - min_exposure) / (max_exposure - min_exposure)

    return as_float(from_range_1(log_norm))


def log_decoding_Log2(log_norm,
                      middle_grey=0.18,
                      min_exposure=-6.5,
                      max_exposure=6.5):
    """
    Defines the common *Log2* decoding function.

    Parameters
    ----------
    log_norm : numeric or array_like
        Logarithmic data to undergo decoding.
    middle_grey : numeric, optional
        *Middle Grey* exposure value.
    min_exposure : numeric, optional
        Minimum exposure level.
    max_exposure : numeric, optional
        Maximum exposure level.

    Returns
    -------
    numeric or ndarray
        Linear *Log2* decoded data.

    Notes
    -----
    -   The common *Log2* decoding function can be used to build logarithmic to
        linear shapers in the *ACES OCIO configuration*.
    -   The shaper with logarithmic encoded values can be decoded back to
        linear domain:

        +-------------------+-------------------+
        | **Shaper Range**  | **Shaper Domain** |
        +===================+===================+
        | [0, 1]            | [0.002, 16.291]   |
        +-------------------+-------------------+

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciencesb`

    Examples
    --------
    >>> log_decoding_Log2(0.5)  # doctest: +ELLIPSIS
    0.1799999...
    """

    log_norm = to_domain_1(log_norm)

    lg2 = log_norm * (max_exposure - min_exposure) + min_exposure
    lin = (2 ** lg2) * middle_grey

    return as_float(from_range_1(lin))
