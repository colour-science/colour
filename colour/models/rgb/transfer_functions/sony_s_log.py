"""
Sony S-Log Encodings
====================

Defines the *Sony S-Log* log encodings:

-   :func:`colour.models.log_encoding_SLog`
-   :func:`colour.models.log_decoding_SLog`
-   :func:`colour.models.log_encoding_SLog2`
-   :func:`colour.models.log_decoding_SLog2`
-   :func:`colour.models.log_encoding_SLog3`
-   :func:`colour.models.log_decoding_SLog3`

References
----------
-   :cite:`SonyCorporation2012a` : Sony Corporation. (2012). S-Log2 Technical
    Paper (pp. 1-9). https://drive.google.com/file/d/\
1Q1RYri6BaxtYYxX0D4zVD6lAmbwmgikc/view?usp=sharing
-   :cite:`SonyCorporationd` : Sony Corporation. (n.d.). Technical Summary
    for S-Gamut3.Cine/S-Log3 and S-Gamut3/S-Log3 (pp. 1-7).
    http://community.sony.com/sony/attachments/sony/\
large-sensor-camera-F5-F55/12359/2/TechnicalSummary_for_S-Gamut3Cine_S-Gamut3_S-Log3_V1_00.pdf
"""

from __future__ import annotations

import numpy as np

from colour.hints import (
    Boolean,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Integer,
)
from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import (
    as_float,
    as_float_array,
    domain_range_scale,
    from_range_1,
    to_domain_1,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "log_encoding_SLog",
    "log_decoding_SLog",
    "log_encoding_SLog2",
    "log_decoding_SLog2",
    "log_encoding_SLog3",
    "log_decoding_SLog3",
]


def log_encoding_SLog(
    x: FloatingOrArrayLike,
    bit_depth: Integer = 10,
    out_normalised_code_value: Boolean = True,
    in_reflection: Boolean = True,
) -> FloatingOrNDArray:
    """
    Define the *Sony S-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.
    bit_depth
        Bit depth used for conversion.
    out_normalised_code_value
        Whether the non-linear *Sony S-Log* data :math:`y` is encoded as
        normalised code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear *Sony S-Log* data :math:`y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporation2012a`

    Examples
    --------
    >>> log_encoding_SLog(0.18)  # doctest: +ELLIPSIS
    0.3849708...

    The values of *IRE and CV of S-Log2 @ISO800* table in
    :cite:`SonyCorporation2012a` are obtained as follows:

    >>> x = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_SLog(x, 10, False) * 100).astype(np.int)
    array([ 3, 38, 65])
    >>> np.around(log_encoding_SLog(x) * (2 ** 10 - 1)).astype(np.int)
    array([ 90, 394, 636])
    """

    x = to_domain_1(x)

    if in_reflection:
        x = x / 0.9

    y = np.where(
        x >= 0,
        ((0.432699 * np.log10(x + 0.037584) + 0.616596) + 0.03),
        x * 5 + 0.030001222851889303,
    )

    y_cv = full_to_legal(y, bit_depth) if out_normalised_code_value else y

    return as_float(from_range_1(y_cv))


def log_decoding_SLog(
    y: FloatingOrArrayLike,
    bit_depth: Integer = 10,
    in_normalised_code_value: Boolean = True,
    out_reflection: Boolean = True,
) -> FloatingOrNDArray:
    """
    Define the *Sony S-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y
        Non-linear *Sony S-Log* data :math:`y`.
    bit_depth
        Bit depth used for conversion.
    in_normalised_code_value
        Whether the non-linear *Sony S-Log* data :math:`y` is encoded as
        normalised code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporation2012a`

    Examples
    --------
    >>> log_decoding_SLog(0.384970815928670)  # doctest: +ELLIPSIS
    0.1...
    """

    y = to_domain_1(y)

    x = legal_to_full(y, bit_depth) if in_normalised_code_value else y

    with domain_range_scale("ignore"):
        x = np.where(
            y >= log_encoding_SLog(0.0, bit_depth, in_normalised_code_value),
            10 ** ((x - 0.616596 - 0.03) / 0.432699) - 0.037584,
            (x - 0.030001222851889303) / 5.0,
        )

    if out_reflection:
        x = x * 0.9

    return as_float(from_range_1(x))


def log_encoding_SLog2(
    x: FloatingOrArrayLike,
    bit_depth: Integer = 10,
    out_normalised_code_value: Boolean = True,
    in_reflection: Boolean = True,
) -> FloatingOrNDArray:
    """
    Define the *Sony S-Log2* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.
    bit_depth
        Bit depth used for conversion.
    out_normalised_code_value
        Whether the non-linear *Sony S-Log2* data :math:`y` is encoded as
        normalised code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear *Sony S-Log2* data :math:`y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporation2012a`

    Examples
    --------
    >>> log_encoding_SLog2(0.18)  # doctest: +ELLIPSIS
    0.3395325...

    The values of *IRE and CV of S-Log2 @ISO800* table in
    :cite:`SonyCorporation2012a` are obtained as follows:

    >>> x = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_SLog2(x, 10, False) * 100).astype(np.int)
    array([ 3, 32, 59])
    >>> np.around(log_encoding_SLog2(x) * (2 ** 10 - 1)).astype(np.int)
    array([ 90, 347, 582])
    """

    x = as_float_array(x)

    return log_encoding_SLog(
        x * 155 / 219, bit_depth, out_normalised_code_value, in_reflection
    )


def log_decoding_SLog2(
    y: FloatingOrArrayLike,
    bit_depth: Integer = 10,
    in_normalised_code_value: Boolean = True,
    out_reflection: Boolean = True,
) -> FloatingOrNDArray:
    """
    Define the *Sony S-Log2* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y
        Non-linear *Sony S-Log2* data :math:`y`.
    bit_depth
        Bit depth used for conversion.
    in_normalised_code_value
        Whether the non-linear *Sony S-Log2* data :math:`y` is encoded as
        normalised code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporation2012a`

    Examples
    --------
    >>> log_decoding_SLog2(0.339532524633774)  # doctest: +ELLIPSIS
    0.1...
    """

    return (
        219
        * log_decoding_SLog(
            y, bit_depth, in_normalised_code_value, out_reflection
        )
        / 155
    )


def log_encoding_SLog3(
    x: FloatingOrArrayLike,
    bit_depth: Integer = 10,
    out_normalised_code_value: Boolean = True,
    in_reflection: Boolean = True,
) -> FloatingOrNDArray:
    """
    Define the *Sony S-Log3* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.
    bit_depth
        Bit depth used for conversion.
    out_normalised_code_value
        Whether the non-linear *Sony S-Log3* data :math:`y` is encoded as
        normalised code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear *Sony S-Log3* data :math:`y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporationd`

    Examples
    --------
    >>> log_encoding_SLog3(0.18)  # doctest: +ELLIPSIS
    0.4105571...

    The values of *S-Log3 10bit code values (18%, 90%)* table in
    :cite:`SonyCorporationd` are obtained as follows:

    >>> x = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_SLog3(x, 10, False) * 100).astype(np.int)
    array([ 4, 41, 61])
    >>> np.around(log_encoding_SLog3(x) * (2 ** 10 - 1)).astype(np.int)
    array([ 95, 420, 598])
    """

    x = to_domain_1(x)

    if not in_reflection:
        x = x * 0.9

    y = np.where(
        x >= 0.01125000,
        (420 + np.log10((x + 0.01) / (0.18 + 0.01)) * 261.5) / 1023,
        (x * (171.2102946929 - 95) / 0.01125000 + 95) / 1023,
    )

    y_cv = y if out_normalised_code_value else legal_to_full(y, bit_depth)

    return as_float(from_range_1(y_cv))


def log_decoding_SLog3(
    y: FloatingOrArrayLike,
    bit_depth: Integer = 10,
    in_normalised_code_value: Boolean = True,
    out_reflection: Boolean = True,
) -> FloatingOrNDArray:
    """
    Define the *Sony S-Log3* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y
        Non-linear *Sony S-Log3* data :math:`y`.
    bit_depth
        Bit depth used for conversion.
    in_normalised_code_value
        Whether the non-linear *Sony S-Log3* data :math:`y` is encoded as
        normalised code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporationd`

    Examples
    --------
    >>> log_decoding_SLog3(0.410557184750733)  # doctest: +ELLIPSIS
    0.1...
    """

    y = to_domain_1(y)

    y = y if in_normalised_code_value else full_to_legal(y, bit_depth)

    x = np.where(
        y >= 171.2102946929 / 1023,
        ((10 ** ((y * 1023 - 420) / 261.5)) * (0.18 + 0.01) - 0.01),
        (y * 1023 - 95) * 0.01125000 / (171.2102946929 - 95),
    )

    if not out_reflection:
        x = x / 0.9

    return as_float(from_range_1(x))
