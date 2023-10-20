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

from __future__ import annotations

import numpy as np

from colour.algebra import sdiv, sdiv_mode
from colour.hints import (
    ArrayLike,
    Literal,
    NDArrayFloat,
    cast,
)
from colour.utilities import (
    as_float,
    as_float_array,
    from_range_1,
    optional,
    to_domain_1,
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
    "logarithmic_function_basic",
    "logarithmic_function_quasilog",
    "logarithmic_function_camera",
    "log_encoding_Log2",
    "log_decoding_Log2",
]

FLT_MIN = 1.175494e-38


def logarithmic_function_basic(
    x: ArrayLike,
    style: Literal[
        "log10", "antiLog10", "log2", "antiLog2", "logB", "antiLogB"
    ]
    | str = "log2",
    base: int = 2,
) -> NDArrayFloat:
    """
    Define the basic logarithmic function.

    Parameters
    ----------
    x
        The data to undergo basic logarithmic conversion.
    style
        Defines the behaviour for the logarithmic function to operate:

        -   *log10*: Applies a base 10 logarithm to the passed value.
        -   *antiLog10*: Applies a base 10 anti-logarithm to the passed value.
        -   *log2*: Applies a base 2 logarithm to the passed value.
        -   *antiLog2*: Applies a base 2 anti-logarithm to the passed value.
        -   *logB*: Applies an arbitrary base logarithm to the passed value.
        -   *antiLogB*: Applies an arbitrary base anti-logarithm to the passed
            value.
    base
        Logarithmic base used for the conversion.

    Returns
    -------
    :class:`numpy.ndarray`
        Logarithmically converted data.

    Examples
    --------
    The basic logarithmic function *styles* operate as follows:

    >>> logarithmic_function_basic(0.18)  # doctest: +ELLIPSIS
    -2.4739311...
    >>> logarithmic_function_basic(0.18, "log10")  # doctest: +ELLIPSIS
    -0.7447274...
    >>> logarithmic_function_basic(0.18, "logB", 3)  # doctest: +ELLIPSIS
    -1.5608767...
    >>> logarithmic_function_basic(  # doctest: +ELLIPSIS
    ...     -2.473931188332412, "antiLog2"
    ... )
    0.18000000...
    >>> logarithmic_function_basic(  # doctest: +ELLIPSIS
    ...     -0.7447274948966939, "antiLog10"
    ... )
    0.18000000...
    >>> logarithmic_function_basic(  # doctest: +ELLIPSIS
    ...     -1.5608767950073117, "antiLogB", 3
    ... )
    0.18000000...
    """

    x = as_float_array(x)
    style = validate_method(
        style,
        ("log10", "antiLog10", "log2", "antiLog2", "logB", "antiLogB"),
        '"{0}" style is invalid, it must be one of {1}!',
    )

    if style == "log10":
        return as_float(np.where(x >= FLT_MIN, np.log10(x), np.log10(FLT_MIN)))
    elif style == "antilog10":
        return as_float(10**x)
    elif style == "log2":
        return as_float(np.where(x >= FLT_MIN, np.log2(x), np.log2(FLT_MIN)))
    elif style == "antilog2":
        return as_float(2**x)
    elif style == "logb":
        return as_float(np.log(x) / np.log(base))
    else:  # style == 'antilogb'
        return as_float(base**x)


def logarithmic_function_quasilog(
    x: ArrayLike,
    style: Literal["linToLog", "logToLin"] | str = "linToLog",
    base: int = 2,
    log_side_slope: float = 1,
    lin_side_slope: float = 1,
    log_side_offset: float = 0,
    lin_side_offset: float = 0,
) -> NDArrayFloat:
    """
    Define the quasilog logarithmic function.

    Parameters
    ----------
    x
        Linear/non-linear data to undergo encoding/decoding.
    style
        Defines the behaviour for the logarithmic function to operate:

        -   *linToLog*: Applies a logarithm to convert linear data to
            logarithmic data.
        -   *logToLin*: Applies an anti-logarithm to convert logarithmic
            data to linear data.
    base
        Logarithmic base used for the conversion.
    log_side_slope
        Slope (or gain) applied to the log side of the logarithmic function.
        The default value is 1.
    lin_side_slope
        Slope of the linear side of the logarithmic function. The default value
        is 1.
    log_side_offset
        Offset applied to the log side of the logarithmic function. The default
        value is 0.
    lin_side_offset
        Offset applied to the linear side of the logarithmic function. The
        default value is 0.

    Returns
    -------
    :class:`numpy.ndarray`
        Encoded/Decoded data.

    Examples
    --------
    >>> logarithmic_function_quasilog(0.18, "linToLog")  # doctest: +ELLIPSIS
    -2.4739311...
    >>> logarithmic_function_quasilog(  # doctest: +ELLIPSIS
    ...     -2.473931188332412, "logToLin"
    ... )
    0.18000000...
    """

    x = as_float_array(x)
    style = validate_method(
        style,
        ("lintolog", "logtolin"),
        '"{0}" style is invalid, it must be one of {1}!',
    )

    if style == "lintolog":
        y = (
            log_side_slope
            * (
                np.log(
                    np.maximum(lin_side_slope * x + lin_side_offset, FLT_MIN)
                )
                / np.log(base)
            )
            + log_side_offset
        )
    else:  # style == 'logtolin'
        with sdiv_mode():
            y = sdiv(
                base ** sdiv(x - log_side_offset, log_side_slope)
                - lin_side_offset,
                lin_side_slope,
            )

    return as_float(y)


def logarithmic_function_camera(
    x: ArrayLike,
    style: Literal["cameraLinToLog", "cameraLogToLin"]
    | str = "cameraLinToLog",
    base: int = 2,
    log_side_slope: float = 1,
    lin_side_slope: float = 1,
    log_side_offset: float = 0,
    lin_side_offset: float = 0,
    lin_side_break: float = 0.005,
    linear_slope: float | None = None,
) -> NDArrayFloat:
    """
    Define the camera logarithmic function.

    Parameters
    ----------
    x
        Linear/non-linear data to undergo encoding/decoding.
    style
        Defines the behaviour for the logarithmic function to operate:

        -   *cameraLinToLog*: Applies a piece-wise function with logarithmic
            and linear segments on linear values, converting them to non-linear
            values.
        -   *cameraLogToLin*: Applies a piece-wise function with logarithmic
            and linear segments on non-linear values, converting them to linear
            values.
    base
        Logarithmic base used for the conversion.
    log_side_slope
        Slope (or gain) applied to the log side of the logarithmic segment. The
        default value is 1.
    lin_side_slope
        Slope of the linear side of the logarithmic segment. The default value
        is 1.
    log_side_offset
        Offset applied to the log side of the logarithmic segment. The default
        value is 0.
    lin_side_offset
        Offset applied to the linear side of the logarithmic segment. The
        default value is 0.
    lin_side_break
        Break-point, defined in linear space, at which the piece-wise function
        transitions between the logarithmic and linear segments.
    linear_slope
        Slope of the linear portion of the curve. The default value is *None*.

    Returns
    -------
    :class:`numpy.ndarray`
        Encoded/Decoded data.

    Examples
    --------
    >>> logarithmic_function_camera(  # doctest: +ELLIPSIS
    ...     0.18, "cameraLinToLog"
    ... )
    -2.4739311...
    >>> logarithmic_function_camera(  # doctest: +ELLIPSIS
    ...     -2.4739311883324122, "cameraLogToLin"
    ... )
    0.1800000...
    """

    x = as_float_array(x)
    style = validate_method(
        style,
        ("cameraLinToLog", "cameraLogToLin"),
        '"{0}" style is invalid, it must be one of {1}!',
    )

    log_side_break = (
        log_side_slope
        * (
            np.log(lin_side_slope * lin_side_break + lin_side_offset)
            / np.log(base)
        )
        + log_side_offset
    )

    with sdiv_mode():
        linear_slope = cast(
            float,
            optional(
                linear_slope,
                (
                    log_side_slope
                    * (
                        sdiv(
                            lin_side_slope,
                            (lin_side_slope * lin_side_break + lin_side_offset)
                            * np.log(base),
                        )
                    )
                ),
            ),
        )

    linear_offset = log_side_break - linear_slope * lin_side_break

    y = zeros(x.shape)
    if style == "cameralintolog":
        m_x = x <= lin_side_break
        y[m_x] = linear_slope * x[m_x] + linear_offset
        y[~m_x] = logarithmic_function_quasilog(
            x[~m_x],
            "linToLog",
            base,
            log_side_slope,
            lin_side_slope,
            log_side_offset,
            lin_side_offset,
        )
    else:  # style == 'cameralogtolin'
        with sdiv_mode():
            m_x = x <= log_side_break
            y[m_x] = sdiv(x[m_x] - linear_offset, linear_slope)
            y[~m_x] = logarithmic_function_quasilog(
                x[~m_x],
                "logToLin",
                base,
                log_side_slope,
                lin_side_slope,
                log_side_offset,
                lin_side_offset,
            )

    return as_float(y)


def log_encoding_Log2(
    lin: ArrayLike,
    middle_grey: float = 0.18,
    min_exposure: float = -6.5,
    max_exposure: float = 6.5,
) -> NDArrayFloat:
    """
    Define the common *Log2* encoding function.

    Parameters
    ----------
    lin
          Linear data to undergo encoding.
    middle_grey
          *Middle Grey* exposure value.
    min_exposure
          Minimum exposure level.
    max_exposure
          Maximum exposure level.

    Returns
    -------
    :class:`numpy.ndarray`
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


def log_decoding_Log2(
    log_norm: ArrayLike,
    middle_grey: float = 0.18,
    min_exposure: float = -6.5,
    max_exposure: float = 6.5,
) -> NDArrayFloat:
    """
    Define the common *Log2* decoding function.

    Parameters
    ----------
    log_norm
        Logarithmic data to undergo decoding.
    middle_grey
        *Middle Grey* exposure value.
    min_exposure
        Minimum exposure level.
    max_exposure
        Maximum exposure level.

    Returns
    -------
    :class:`numpy.ndarray`
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
    lin = (2**lg2) * middle_grey

    return as_float(from_range_1(lin))
