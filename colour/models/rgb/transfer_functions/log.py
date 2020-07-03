# -*- coding: utf-8 -*-
"""
Common Log Encodings
====================

Defines the common log encodings:

-   :func:`colour.models.logarithm_basic`
-   :func:`colour.models.logarithm_lin_to_log`
-   :func:`colour.models.logarithm_log_to_lin`
-   :func:`colour.models.logarithm_camera_lin_to_log`
-   :func:`colour.models.logarithm_camera_log_to_lin`
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
from colour.utilities import from_range_1, to_domain_1, as_float

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'logarithm_basic', 'logarithm_lin_to_log', 'logarithm_log_to_lin',
    'logarithm_camera_lin_to_log', 'logarithm_camera_log_to_lin',
    'log_encoding_Log2', 'log_decoding_Log2'
]

FLT_MIN = 1.175494e-38


def logarithm_basic(x, base=2, style='log2'):
    """
    Defines the basic logarithmic function.

    Parameters
    ----------
    x : numeric
        The data to undergo basic logarithmic conversion.
    base : numeric, optional
        The base value used for the conversion.
    style : unicode, optional
        **{'log10', 'antiLog10', 'log2', 'antiLog2'}**,
        Defines the behaviour for the logarithmic function to operate:

        -   *log10*: Applies a base 10 logarithm to the passed value.
        -   *antiLog10*: Applies a base 10 anti-logarithm to the passed value.
        -   *log2*: Applies a base 2 logarithm to the passed value.
        -   *antiLog2*: Applies a base 2 anti-logarithm to the passed value.

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

    >>> logarithm_basic(0.18)  # doctest: +ELLIPSIS
    -2.4739311...
    >>> logarithm_basic(0.18, 10, 'log10')  # doctest: +ELLIPSIS
    -0.7447274...
    >>> logarithm_basic(  # doctest: +ELLIPSIS
        -2.473931188332412, 2, 'antiLog2')
    0.18000000...
    >>> logarithm_basic(  # doctest: +ELLIPSIS
        -0.7447274948966939, 10, 'antiLog10')
    0.18000000...
    """

    def log_base(x, base=2):
        """
        Returns the (base) logarithm of the passed value.
        """

        return (np.log(x) / np.log(base))

    def antilog_base(x, base=2):
        """
        Returns the (base) anti-logarithm of the passed value.
        """

        return base ** x

    style = style.lower()
    if style == 'log10':
        return as_float(
            np.where(x >= FLT_MIN, log_base(x, 10), log_base(FLT_MIN, 10)))
    elif style == 'antilog10':
        return antilog_base(x, 10)
    elif style == 'log2':
        return as_float(np.where(x >= FLT_MIN, log_base(x), log_base(FLT_MIN)))
    elif style == 'antilog2':
        return antilog_base(x)
    else:
        raise ValueError(
            'Undefined style used: "{0}", must be one of the following: '
            '"{1}".'.format(
                style, ', '.join(['log10', 'antiLog10', 'log2', 'antiLog2'])))


def logarithm_lin_to_log(x,
                         base=2,
                         log_side_slope=1,
                         lin_side_slope=1,
                         log_side_offset=0,
                         lin_side_offset=0):
    """
    Defines the linear to logarithm encoding transfer function.

    Parameters
    ----------
    x : numeric
        Linear data to undergo encoding.
    base : numeric, optional
        The base value used for the transfer.
    log_side_slope : numeric, optional
        It is the slope (or gain) applied to the log side
        of the logarithmic segment. Its default value is 1.
    lin_side_slope : numeric, optional
        It is the slope of the linear side of the
        logarithmic segment. Its default value is 1.
    log_side_offset : numeric, optional
        It is the offset applied to the log side
        of the logarithmic segment. Its default value is 0.
    lin_side_offset : numeric, optional
        It is the offset applied to the linaer side
        of the logarithmic segment. Its default value is 0.

    Returns
    -------
    numeric or ndarray
        Logarithmic encoded data.

    Example
    -------
    >>> logarithm_lin_to_log(0.18)  # doctest: +ELLIPSIS
    -2.4739311...
    """

    return as_float((log_side_slope * (np.log(
        max(lin_side_slope * x + lin_side_offset, FLT_MIN)) / np.log(base)) +
                     log_side_offset))


def logarithm_log_to_lin(x,
                         base=2,
                         log_side_slope=1,
                         lin_side_slope=1,
                         log_side_offset=0,
                         lin_side_offset=0):
    """
    Defines the logarithmic to linear decoding transfer function.

    Parameters
    ----------
    x : numeric
        Logarithmic data to undergo decoding.
    base : numeric, optional
        The base value used for the transfer.
    log_side_slope : numeric, optional
        It is the slope (or gain) applied to the log side
        of the logarithmic segment. Its default value is 1.
    lin_side_slope : numeric, optional
        It is the slope of the linear side
        of the logarithmic segment. Its default value is 1.
    log_side_offset : numeric, optional
        It is the offset applied to the log side
        of the logarithmic segment. Its default value is 0.
    lin_side_offset : numeric, optional
        It is the offset applied to the linaer side
        of the logarithmic segment. Its default value is 0.

    Returns
    -------
    numeric or ndarray
        Linear decoded data.

    Example
    -------
    >>> logarithm_log_to_lin(-2.47393118833)  # doctest: +ELLIPSIS
    0.18000000...
    """

    return as_float(
        ((base ** ((x - log_side_offset) / log_side_slope) - lin_side_offset) /
         lin_side_slope))


def logarithm_camera_lin_to_log(x,
                                lin_side_break,
                                base=2,
                                log_side_slope=1,
                                lin_side_slope=1,
                                log_side_offset=0,
                                lin_side_offset=0):
    """
    Defines the parametrized camera log encoding function,
    which does the linear to logarithmic conversion.

    Parameters
    ----------
    x : numeric
        Linear data to undergo encdoing.
    lin_side_break : numeric
        It is the the break-point, defined in linear space,
        at which the piece-wise function transitions between
        the logarithmic and linear segments.
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
        It is the offset applied to the linaer side
        of the logarithmic segment. Its default value is 0.

    Returns
    -------
    numeric or ndarray
        Logarithmic encoded data.

    Example
    -------
    >>> logarithm_camera_lin_to_log(0.18, 2.2)  # doctest: +ELLIPSIS
    -0.1871528...
    """

    log_side_break = (
        log_side_slope *
        (np.log(lin_side_slope * lin_side_break + lin_side_offset) /
         np.log(base)) + log_side_offset)
    linear_slope = (log_side_slope * (lin_side_slope / (
        (lin_side_slope * lin_side_break + lin_side_offset) * np.log(base))))
    linear_offset = log_side_break - linear_slope * lin_side_break

    return as_float(
        np.where(x <= lin_side_break, linear_slope * x + linear_offset,
                 logarithm_lin_to_log(x)))


def logarithm_camera_log_to_lin(x,
                                lin_side_break,
                                base=2,
                                log_side_slope=1,
                                lin_side_slope=1,
                                log_side_offset=0,
                                lin_side_offset=0):
    """
    Defines the parametrized camera log decoding function,
    which does the logarithmic to linear conversion.

    Parameters
    ----------
    x : numeric
        Logarithmic data to undergo decoding.
    lin_side_break : numeric
        It is the the break-point, defined in linear space,
        at which the piece-wise function transitions between
        the logarithmic and linear segments.
    base : numeric, optional
        The base value used for the transfer.
    log_side_slope : numeric, optional
        It is the slope (or gain) applied to the log side
        of the logarithmic segment. Its default value is 1.
    lin_side_slope : numeric, optional
        It is the slope of the linear side
        of the logarithmic segment. Its default value is 1.
    log_side_offset : numeric, optional
        It is the offset applied to the log side
        of the logarithmic segment. Its default value is 0.
    lin_side_offset : numeric, optional
        It is the offset applied to the linaer side
        of the logarithmic segment. Its default value is 0.

    Returns
    -------
    numeric or ndarray
        Linear decoded data.

    Example
    -------
    >>> logarithm_camera_log_to_lin(-0.187152831975, 2.2)  # doctest: +ELLIPSIS
    0.18000000...
    """

    log_side_break = (
        log_side_slope *
        (np.log(lin_side_slope * lin_side_break + lin_side_offset) /
         np.log(base)) + log_side_offset)
    linear_slope = (log_side_slope * (lin_side_slope / (
        (lin_side_slope * lin_side_break + lin_side_offset) * np.log(base))))
    linear_offset = log_side_break - linear_slope * lin_side_break

    return as_float(
        np.where(x <= log_side_break, (x - linear_offset) / linear_slope,
                 logarithm_log_to_lin(x)))


def log_encoding_Log2(lin,
                      middle_grey=0.18,
                      min_exposure=0.18 * 2 ** -6.5,
                      max_exposure=0.18 * 2 ** 6.5):
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

    The common *Log2* encoding function can be used
    to build linear to logarithmic shapers in the
    ACES OCIO configuration.

    A (48-nits OCIO) shaper having values in a linear
    domain, can be encoded to a logarithmic domain:

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
    Linear numeric input is encoded as follows:

    >>> log_encoding_Log2(18)
    0.40773288970434662

    Linear array-like input is encoded as follows:

    >>> log_encoding_Log2(np.linspace(1, 2, 3))
    array([ 0.15174832,  0.18765817,  0.21313661])
    """

    lin = to_domain_1(lin)

    lg2 = np.log2(lin / middle_grey)
    log_norm = (lg2 - min_exposure) / (max_exposure - min_exposure)

    return from_range_1(log_norm)


def log_decoding_Log2(log_norm,
                      middle_grey=0.18,
                      min_exposure=0.18 * 2 ** -6.5,
                      max_exposure=0.18 * 2 ** 6.5):
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

    The common *Log2* decoding function can be used
    to build logarithmic to linear shapers in the
    ACES OCIO configuration.

    The shaper with logarithmic encoded values can be
    decoded back to linear domain:

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
    Logarithmic input is decoded as follows:

    >>> log_decoding_Log2(0.40773288970434662)
    17.999999999999993

    Linear array-like input is encoded as follows:

    >>> log_decoding_Log2(np.linspace(0, 1, 4))
    array([  1.80248299e-01,   7.77032379e+00,   3.34970882e+02,
             1.44402595e+04])
    """

    log_norm = to_domain_1(log_norm)

    lg2 = log_norm * (max_exposure - min_exposure) + min_exposure
    lin = (2 ** lg2) * middle_grey

    return from_range_1(lin)
