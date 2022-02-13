"""
Academy Color Encoding System - Log Encodings
=============================================

Defines the *Academy Color Encoding System* (ACES) log encodings:

-   :func:`colour.models.log_encoding_ACESproxy`
-   :func:`colour.models.log_decoding_ACESproxy`
-   :func:`colour.models.log_encoding_ACEScc`
-   :func:`colour.models.log_decoding_ACEScc`
-   :func:`colour.models.log_encoding_ACEScct`
-   :func:`colour.models.log_decoding_ACEScct`

References
----------
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014q` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Technical
    Bulletin TB-2014-004 - Informative Notes on SMPTE ST 2065-1 - Academy Color
    Encoding Specification (ACES) (pp. 1-40). Retrieved December 19, 2014, from
    http://j.mp/TB-2014-004
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014r` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Technical
    Bulletin TB-2014-012 - Academy Color Encoding System Version 1.0 Component
    Names (pp. 1-8). Retrieved December 19, 2014, from http://j.mp/TB-2014-012
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014s` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2013). Specification
    S-2013-001 - ACESproxy, an Integer Log Encoding of ACES Image Data.
    Retrieved December 19, 2014, from http://j.mp/S-2013-001
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014t` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Specification
    S-2014-003 - ACEScc, A Logarithmic Encoding of ACES Data for use within
    Color Grading Systems (pp. 1-12). Retrieved December 19, 2014, from
    http://j.mp/S-2014-003
-   :cite:`TheAcademyofMotionPictureArtsandSciences2016c` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project. (2016). Specification S-2016-001 -
    ACEScct, A Quasi-Logarithmic Encoding of ACES Data for use within Color
    Grading Systems. Retrieved October 10, 2016, from http://j.mp/S-2016-001
-   :cite:`TheAcademyofMotionPictureArtsandSciencese` : The Academy of Motion
    Picture Arts and Sciences, Science and Technology Council, & Academy Color
    Encoding System (ACES) Project Subcommittee. (n.d.). Academy Color Encoding
    System. Retrieved February 24, 2014, from
    http://www.oscars.org/science-technology/council/projects/aces.html
"""

from __future__ import annotations

import numpy as np

from colour.hints import (
    Boolean,
    Dict,
    Floating,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    IntegerOrArrayLike,
    IntegerOrNDArray,
    Literal,
    Union,
)
from colour.utilities import (
    Structure,
    as_float,
    as_int,
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
    "CONSTANTS_ACES_PROXY_10",
    "CONSTANTS_ACES_PROXY_12",
    "CONSTANTS_ACES_PROXY",
    "CONSTANTS_ACES_CCT",
    "log_encoding_ACESproxy",
    "log_decoding_ACESproxy",
    "log_encoding_ACEScc",
    "log_decoding_ACEScc",
    "log_encoding_ACEScct",
    "log_decoding_ACEScct",
]

CONSTANTS_ACES_PROXY_10: Structure = Structure(
    CV_min=64,
    CV_max=940,
    steps_per_stop=50,
    mid_CV_offset=425,
    mid_log_offset=2.5,
)
"""*ACESproxy* 10 bit colourspace constants."""

CONSTANTS_ACES_PROXY_12: Structure = Structure(
    CV_min=256,
    CV_max=3760,
    steps_per_stop=200,
    mid_CV_offset=1700,
    mid_log_offset=2.5,
)
"""*ACESproxy* 12 bit colourspace constants."""

CONSTANTS_ACES_PROXY: Dict = {
    10: CONSTANTS_ACES_PROXY_10,
    12: CONSTANTS_ACES_PROXY_12,
}
"""Aggregated *ACESproxy* colourspace constants."""

CONSTANTS_ACES_CCT: Structure = Structure(
    X_BRK=0.0078125,
    Y_BRK=0.155251141552511,
    A=10.5402377416545,
    B=0.0729055341958355,
)
"""*ACEScct* colourspace constants."""


# pylint: disable=W0102
def log_encoding_ACESproxy(
    lin_AP1: FloatingOrArrayLike,
    bit_depth: Literal[10, 12] = 10,
    out_int: Boolean = False,
    constants: Dict = CONSTANTS_ACES_PROXY,
) -> Union[FloatingOrNDArray, IntegerOrNDArray]:
    """
    Define the *ACESproxy* colourspace log encoding curve / opto-electronic
    transfer function.

    Parameters
    ----------
    lin_AP1
        *lin_AP1* value.
    bit_depth
        *ACESproxy* bit depth.
    out_in
        Whether to return value as integer code value or float equivalent of a
        code value at a given bit depth.
    constants
        *ACESproxy* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.integer` or :class:`numpy.ndarray`
        *ACESproxy* non-linear value.

    Notes
    -----
    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``lin_AP1``    | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``ACESproxy``  | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    \\* This definition has an output integer switch, thus the domain-range
    scale information is only given for the floating point mode.

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014s`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> log_encoding_ACESproxy(0.18)  # doctest: +ELLIPSIS
    0.4164222...
    >>> log_encoding_ACESproxy(0.18, out_int=True)
    426
    """

    lin_AP1 = to_domain_1(lin_AP1)

    CV_min = constants[bit_depth].CV_min
    CV_max = constants[bit_depth].CV_max
    mid_CV_offset = constants[bit_depth].mid_CV_offset
    mid_log_offset = constants[bit_depth].mid_log_offset
    steps_per_stop = constants[bit_depth].steps_per_stop

    def float_2_cv(x: Floating) -> Floating:
        """Convert given numeric to code value."""

        return np.maximum(CV_min, np.minimum(CV_max, np.round(x)))

    ACESproxy = np.where(
        lin_AP1 > 2**-9.72,
        float_2_cv(
            (np.log2(lin_AP1) + mid_log_offset) * steps_per_stop
            + mid_CV_offset
        ),
        np.resize(CV_min, lin_AP1.shape),
    )

    if out_int:
        return as_int(np.round(ACESproxy))
    else:
        return as_float(from_range_1(ACESproxy / (2**bit_depth - 1)))


# pylint: disable=W0102
def log_decoding_ACESproxy(
    ACESproxy: Union[FloatingOrArrayLike, IntegerOrArrayLike],
    bit_depth: Literal[10, 12] = 10,
    in_int: Boolean = False,
    constants: Dict = CONSTANTS_ACES_PROXY,
) -> FloatingOrNDArray:
    """
    Define the *ACESproxy* colourspace log decoding curve / electro-optical
    transfer function.

    Parameters
    ----------
    ACESproxy
        *ACESproxy* non-linear value.
    bit_depth
        *ACESproxy* bit depth.
    in_int
        Whether to treat the input value as integer code value or float
        equivalent of a code value at a given bit depth.
    constants
        *ACESproxy* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *lin_AP1* value.

    Notes
    -----
    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``ACESproxy``  | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``lin_AP1``    | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    \\* This definition has an input integer switch, thus the domain-range
    scale information is only given for the floating point mode.

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014s`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> log_decoding_ACESproxy(0.416422287390029)  # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding_ACESproxy(426, in_int=True)  # doctest: +ELLIPSIS
    0.1...
    """

    ACESproxy = to_domain_1(ACESproxy)

    mid_CV_offset = constants[bit_depth].mid_CV_offset
    mid_log_offset = constants[bit_depth].mid_log_offset
    steps_per_stop = constants[bit_depth].steps_per_stop

    if not in_int:
        ACESproxy = ACESproxy * (2**bit_depth - 1)

    lin_AP1 = 2 ** (
        (ACESproxy - mid_CV_offset) / steps_per_stop - mid_log_offset
    )

    return as_float(from_range_1(lin_AP1))


def log_encoding_ACEScc(lin_AP1: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Define the *ACEScc* colourspace log encoding / opto-electronic transfer
    function.

    Parameters
    ----------
    lin_AP1
        *lin_AP1* value.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *ACEScc* non-linear value.

    Notes
    -----
    +-------------+-----------------------+---------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``lin_AP1`` | [0, 1]                | [0, 1]        |
    +-------------+-----------------------+---------------+

    +-------------+-----------------------+---------------+
    | **Range**   | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``ACEScc``  | [0, 1]                | [0, 1]        |
    +-------------+-----------------------+---------------+

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014t`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> log_encoding_ACEScc(0.18)  # doctest: +ELLIPSIS
    0.4135884...
    """

    lin_AP1 = to_domain_1(lin_AP1)

    ACEScc = np.where(
        lin_AP1 < 0,
        (np.log2(2**-16) + 9.72) / 17.52,
        (np.log2(2**-16 + lin_AP1 * 0.5) + 9.72) / 17.52,
    )
    ACEScc = np.where(
        lin_AP1 >= 2**-15,
        (np.log2(lin_AP1) + 9.72) / 17.52,
        ACEScc,
    )

    return as_float(from_range_1(ACEScc))


def log_decoding_ACEScc(ACEScc: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Define the *ACEScc* colourspace log decoding / electro-optical transfer
    function.

    Parameters
    ----------
    ACEScc
        *ACEScc* non-linear value.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *lin_AP1* value.

    Notes
    -----
    +-------------+-----------------------+---------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``ACEScc``  | [0, 1]                | [0, 1]        |
    +-------------+-----------------------+---------------+

    +-------------+-----------------------+---------------+
    | **Range**   | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``lin_AP1`` | [0, 1]                | [0, 1]        |
    +-------------+-----------------------+---------------+

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014t`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> log_decoding_ACEScc(0.413588402492442)  # doctest: +ELLIPSIS
    0.1799999...
    """

    ACEScc = to_domain_1(ACEScc)

    lin_AP1 = np.where(
        ACEScc < (9.72 - 15) / 17.52,
        (2 ** (ACEScc * 17.52 - 9.72) - 2**-16) * 2,
        2 ** (ACEScc * 17.52 - 9.72),
    )
    lin_AP1 = np.where(
        ACEScc >= (np.log2(65504) + 9.72) / 17.52,
        65504,
        lin_AP1,
    )

    return as_float(from_range_1(lin_AP1))


# pylint: disable=W0102
def log_encoding_ACEScct(
    lin_AP1: FloatingOrArrayLike, constants: Structure = CONSTANTS_ACES_CCT
) -> FloatingOrNDArray:
    """
    Define the *ACEScct* colourspace log encoding / opto-electronic transfer
    function.

    Parameters
    ----------
    lin_AP1
        *lin_AP1* value.
    constants
        *ACEScct* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *ACEScct* non-linear value.

    Notes
    -----
    +-------------+-----------------------+---------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``lin_AP1`` | [0, 1]                | [0, 1]        |
    +-------------+-----------------------+---------------+

    +-------------+-----------------------+---------------+
    | **Range**   | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``ACEScct`` | [0, 1]                | [0, 1]        |
    +-------------+-----------------------+---------------+

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2016c`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> log_encoding_ACEScct(0.18)  # doctest: +ELLIPSIS
    0.4135884...
    """

    lin_AP1 = to_domain_1(lin_AP1)

    ACEScct = np.where(
        lin_AP1 <= constants.X_BRK,
        constants.A * lin_AP1 + constants.B,
        (np.log2(lin_AP1) + 9.72) / 17.52,
    )

    return as_float(from_range_1(ACEScct))


# pylint: disable=W0102
def log_decoding_ACEScct(
    ACEScct: FloatingOrArrayLike, constants: Structure = CONSTANTS_ACES_CCT
) -> FloatingOrNDArray:
    """
    Define the *ACEScct* colourspace log decoding / electro-optical transfer
    function.

    Parameters
    ----------
    ACEScct
        *ACEScct* non-linear value.
    constants
        *ACEScct* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *lin_AP1* value.

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2016c`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Notes
    -----
    +-------------+-----------------------+---------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``ACEScct`` | [0, 1]                | [0, 1]        |
    +-------------+-----------------------+---------------+

    +-------------+-----------------------+---------------+
    | **Range**   | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``lin_AP1`` | [0, 1]                | [0, 1]        |
    +-------------+-----------------------+---------------+

    Examples
    --------
    >>> log_decoding_ACEScct(0.413588402492442)  # doctest: +ELLIPSIS
    0.1799999...
    """

    ACEScct = to_domain_1(ACEScct)

    lin_AP1 = np.where(
        ACEScct > constants.Y_BRK,
        2 ** (ACEScct * 17.52 - 9.72),
        (ACEScct - constants.B) / constants.A,
    )

    return as_float(from_range_1(lin_AP1))
