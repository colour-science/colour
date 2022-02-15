"""
Nikon N-Log Log Encoding
========================

Defines the *Nikon N-Log* log encoding:

-   :func:`colour.models.log_encoding_NLog`
-   :func:`colour.models.log_decoding_NLog`

References
----------
-   :cite:`Nikon2018` : Nikon. (2018). N-Log Specification Document - Version
    1.0.0 (pp. 1-5). Retrieved September 9, 2019, from
    http://download.nikonimglib.com/archive3/hDCmK00m9JDI03RPruD74xpoU905/\
N-Log_Specification_(En)01.pdf
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow
from colour.hints import (
    Boolean,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Integer,
)
from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import Structure, as_float, from_range_1, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - http://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "NLOG_CONSTANTS",
    "log_encoding_NLog",
    "log_decoding_NLog",
]

NLOG_CONSTANTS: Structure = Structure(
    cut1=0.328,
    cut2=(452 / 1023),
    a=(650 / 1023),
    b=0.0075,
    c=(150 / 1023),
    d=(619 / 1023),
)
"""*Nikon N-Log* colourspace constants."""


def log_encoding_NLog(
    in_r: FloatingOrArrayLike,
    bit_depth: Integer = 10,
    out_normalised_code_value: Boolean = True,
    in_reflection: Boolean = True,
    constants: Structure = NLOG_CONSTANTS,
) -> FloatingOrNDArray:
    """
    Define the *Nikon N-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    in_r
        Linear reflection data :math`in`.
    bit_depth
        Bit depth used for conversion.
    out_normalised_code_value
        Whether the non-linear *Nikon N-Log* data :math:`out` is encoded as
        normalised code values.
    in_reflection
        Whether the light level :math`in` to a camera is reflection.
    constants
        *Nikon N-Log* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear data :math:`out`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``in_r``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``out_r``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Nikon2018`

    Examples
    --------
    >>> log_encoding_NLog(0.18)  # doctest: +ELLIPSIS
    0.3636677...
    """

    in_r = to_domain_1(in_r)

    if not in_reflection:
        in_r = in_r * 0.9

    cut1 = constants.cut1
    a = constants.a
    b = constants.b
    c = constants.c
    d = constants.d

    out_r = np.where(
        in_r < cut1,
        a * spow(in_r + b, 1 / 3),
        c * np.log(in_r) + d,
    )

    out_r_cv = (
        out_r if out_normalised_code_value else legal_to_full(out_r, bit_depth)
    )

    return as_float(from_range_1(out_r_cv))


def log_decoding_NLog(
    out_r: FloatingOrArrayLike,
    bit_depth: Integer = 10,
    in_normalised_code_value: Boolean = True,
    out_reflection: Boolean = True,
    constants: Structure = NLOG_CONSTANTS,
) -> FloatingOrNDArray:
    """
    Define the *Nikon N-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    out_r
        Non-linear data :math:`out`.
    bit_depth
        Bit depth used for conversion.
    in_normalised_code_value
        Whether the non-linear *Nikon N-Log* data :math:`out` is encoded as
        normalised code values.
    out_reflection
        Whether the light level :math`in` to a camera is reflection.
    constants
        *Nikon N-Log* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear reflection data :math`in`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``out_r``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``in_r``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Nikon2018`

    Examples
    --------
    >>> log_decoding_NLog(0.36366777011713869)  # doctest: +ELLIPSIS
    0.1799999...
    """

    out_r = to_domain_1(out_r)

    out_r = (
        out_r if in_normalised_code_value else full_to_legal(out_r, bit_depth)
    )

    cut2 = constants.cut2
    a = constants.a
    b = constants.b
    c = constants.c
    d = constants.d

    in_r = np.where(
        out_r < cut2,
        spow(out_r / a, 3) - b,
        np.exp((out_r - d) / c),
    )

    if not out_reflection:
        in_r = in_r / 0.9

    return as_float(from_range_1(in_r))
