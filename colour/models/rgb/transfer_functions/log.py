# -*- coding: utf-8 -*-
"""
Common Log Encodings
====================

Defines the common log encodings:

-   :func:`colour.models.logarithmic_function_basic`
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
from colour.utilities import as_float, from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'logarithmic_function_basic', 'logarithmic_function_camera',
    'log_encoding_Log2', 'log_decoding_Log2'
]

FLT_MIN = 1.175494e-38


def logarithmic_function_basic(x, base=2, style='log2'):
    """
    Defines the basic logarithmic function.

    Parameters
    ----------
    x : numeric
        The data to undergo basic logarithmic conversion.
    base : numeric, optional
        The base value used for the conversion.
    style : unicode, optional
        **{'log10', 'antiLog10', 'log2', 'antiLog2', 'logN', 'antiLogN'}**,
        Defines the behaviour for the logarithmic function to operate:

        -   *log10*: Applies a base 10 logarithm to the passed value.
        -   *antiLog10*: Applies a base 10 anti-logarithm to the passed value.
        -   *log2*: Applies a base 2 logarithm to the passed value.
        -   *antiLog2*: Applies a base 2 anti-logarithm to the passed value.
        -   *LogN*: Applies an arbitrary base logarithm to the passed value.
        -   *antiLogN*: Applies an arbitrary base anti-logarithm to the passed
            value.

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
    >>> logarithmic_function_basic(0.18, 10, 'log10')  # doctest: +ELLIPSIS
    -0.7447274...
    >>> logarithmic_function_basic(  # doctest: +ELLIPSIS
    ...    0.18, 2.2 , 'LogN')
    -2.17487782...
    >>> logarithmic_function_basic(  # doctest: +ELLIPSIS
    ...    -2.473931188332412, 2, 'antiLog2')
    0.18000000...
    >>> logarithmic_function_basic(  # doctest: +ELLIPSIS
    ...    -0.7447274948966939, 10, 'antiLog10')
    0.18000000...
    >>> logarithmic_function_basic(  # doctest: +ELLIPSIS
    ...    -2.1748778238301729 , 2.2 , 'antiLogN')
    0.18000000...
    """

    style = style.lower()
    if style == 'log10':
        return as_float(np.where(x >= FLT_MIN, np.log10(x), np.log10(FLT_MIN)))
    elif style == 'antilog10':
        return as_float(10 ** x)
    elif style == 'log2':
        return as_float(np.where(x >= FLT_MIN, np.log2(x), np.log2(FLT_MIN)))
    elif style == 'antilog2':
        return as_float(2 ** x)
    elif style == 'logn':
        return as_float(np.log(x) / np.log(base))
    elif style == 'antilogn':
        return as_float(base ** x)
    else:
        raise ValueError(
            'Undefined style used: "{0}", must be one of the following: '
            '"{1}".'.format(
                style, ', '.join([
                    'log10', 'antiLog10', 'log2', 'antiLog2', 'logN',
                    'antiLogN'
                ])))


def logarithmic_function_camera(x,
                                lin_side_break=0,
                                style='linToLog',
                                base=2,
                                log_side_slope=1,
                                lin_side_slope=1,
                                log_side_offset=0,
                                lin_side_offset=0):
    """
    Defines the camera logarithmic function.

    Parameters
    ----------
    x : numeric
        Linear/non-linear data to undergo encoding/decoding.
    lin_side_break : numeric
        It is the the break-point, defined in linear space,
        at which the piece-wise function transitions between
        the logarithmic and linear segments.

    style : unicode, optional
        **{'linToLog', 'logToLin', 'cameraLinToLog', 'cameraLogToLin'}**,
        Defines the behaviour for the logarithmic function to operate:

        -   *linToLog*: Applies a logarithm to convert linear data to
            logarithmic data.

        -   *logToLin*: Applies an anti-logarithm to convert logarithmic
            data to linear data.

        -   *cameraLinToLog*: Applies a piece-wise function with logarithmic
            and linear segments on linear values, converting them to non-linear
            values.

        -   *cameraLogToLin*: Applies a piece-wise function with logarithmic
            and linear segments on non-linear values, converting them to linear
            values.

    base : numeric, optional
        The base value used for the transfer.
    log_side_slope : numeric, optional
        It is the slope (or gain) applied to the log side
        of the logarithmic segment. Its default value is 1.
    lin_side_slope : numeric, optional
        It is the slope of the linear side of
        the logarithmic segment. Its default value is 1.
    log_side_offset : numeric, optional
        It is the offset applied to the log side
        of the logarithmic segment. Its default value is 0.
    lin_side_offset : numeric, optional
        It is the offset applied to the linear side
        of the logarithmic segment. Its default value is 0.

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
    ...    0.18, style='linToLog')
    -2.4739311...
    >>> logarithmic_function_camera(  # doctest: +ELLIPSIS
    ...    -2.47393118833, style='logToLin')
    0.18000000...
    >>> logarithmic_function_camera(  # doctest: +ELLIPSIS
    ...    0.18, 2.2, style='cameraLinToLog')
    array(-0.187152831975386)
    >>> logarithmic_function_camera(  # doctest: +ELLIPSIS
    ...    -0.187152831975, 2.2, style='cameraLogToLin')
    array(0.18000000000058866)
    """

    def lin_to_log(x,
                   base=2,
                   log_side_slope=1,
                   lin_side_slope=1,
                   log_side_offset=0,
                   lin_side_offset=0):
        """
        Defines the linear to logarithm encoding transfer function.
        """

        return as_float((log_side_slope * (np.log(
            max(lin_side_slope * x + lin_side_offset, FLT_MIN)) / np.log(base))
                         + log_side_offset))

    def log_to_lin(x,
                   base=2,
                   log_side_slope=1,
                   lin_side_slope=1,
                   log_side_offset=0,
                   lin_side_offset=0):
        """
        Defines the logarithmic to linear decoding transfer function.
        """

        return as_float(
            ((base **
              ((x - log_side_offset) / log_side_slope) - lin_side_offset) /
             lin_side_slope))

    log_side_break = (
        log_side_slope *
        (np.log(lin_side_slope * lin_side_break + lin_side_offset) /
         np.log(base)) + log_side_offset)

    linear_slope = (log_side_slope * (lin_side_slope / (
        (lin_side_slope * lin_side_break + lin_side_offset) * np.log(base))))

    linear_offset = log_side_break - linear_slope * lin_side_break

    style = style.lower()
    if style == 'lintolog':
        return lin_to_log(x, base, log_side_slope, lin_side_slope,
                          log_side_offset, lin_side_offset)
    elif style == 'logtolin':
        return log_to_lin(x, base, log_side_slope, lin_side_slope,
                          log_side_offset, lin_side_offset)
    elif style == 'cameralintolog':
        return np.where(x <= lin_side_break, linear_slope * x + linear_offset,
                        lin_to_log(x))
    elif style == 'cameralogtolin':
        return np.where(x <= log_side_break,
                        (x - linear_offset) / linear_slope, log_to_lin(x))
    else:
        raise ValueError(
            'Undefined style used: "{0}", must be one of the following: '
            '"{1}".'.format(
                style, ', '.join([
                    'logToLog', 'logToLin', 'cameraLinToLog', 'cameraLogToLin'
                ])))


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
