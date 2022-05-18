"""
Leica L-Log Log Encoding
========================

Defines the *Leica L-Log* log encoding:

-   :func:`colour.models.log_encoding_LLog`
-   :func:`colour.models.log_decoding_LLog`

References
----------
-   :cite:`LeicaCameraAG2022` : Leica Camera AG. (2022). Leica L-Log Reference
    Manual. https://leica-camera.com/sites/default/files/\
pm-65976-210914__L-Log_Reference_Manual_EN.pdf
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
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANTS_LLOG",
    "log_encoding_LLog",
    "log_decoding_LLog",
]

CONSTANTS_LLOG: Structure = Structure(
    cut1=0.006,
    cut2=0.1380,
    a=8,
    b=0.09,
    c=0.27,
    d=1.3,
    e=0.0115,
    f=0.6,
)
"""*Leica L-Log* colourspace constants."""


def log_encoding_LLog(
    LSR: FloatingOrArrayLike,
    bit_depth: Integer = 10,
    out_normalised_code_value: Boolean = True,
    in_reflection: Boolean = True,
    constants: Structure = CONSTANTS_LLOG,
) -> FloatingOrNDArray:
    """
    Define the *Leica L-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    LSR
        Linear scene reflection :math:`LSR` values.
    bit_depth
        Bit depth used for conversion.
    out_normalised_code_value
        Whether the non-linear *Leica L-Log* data :math:`L-Log` is encoded as
        normalised code values.
    in_reflection
        Whether the light level :math`in` to a camera is reflection.
    constants
        *Leica L-Log* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *L-Log* 10-bit equivalent code value :math:`L-Log`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``LSR``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``LLog``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`LeicaCameraAG2022`

    Examples
    --------
    >>> log_encoding_LLog(0.18)  # doctest: +ELLIPSIS
    0.4353139...
    """

    LSR = to_domain_1(LSR)

    if not in_reflection:
        LSR = LSR * 0.9

    cut1 = constants.cut1
    a = constants.a
    b = constants.b
    c = constants.c
    d = constants.d
    e = constants.e
    f = constants.f

    LLog = np.where(
        LSR <= cut1,
        a * LSR + b,
        c * np.log10(d * LSR + e) + f,
    )

    LLog_cv = (
        LLog if out_normalised_code_value else legal_to_full(LLog, bit_depth)
    )

    return as_float(from_range_1(LLog_cv))


def log_decoding_LLog(
    LLog: FloatingOrArrayLike,
    bit_depth: Integer = 10,
    in_normalised_code_value: Boolean = True,
    out_reflection: Boolean = True,
    constants: Structure = CONSTANTS_LLOG,
) -> FloatingOrNDArray:
    """
    Define the *Leica L-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    LLog
        *L-Log* 10-bit equivalent code value :math:`L-Log`.
    bit_depth
        Bit depth used for conversion.
    in_normalised_code_value
        Whether the non-linear *Leica L-Log* data :math:`L-Log` is encoded as
        normalised code values.
    out_reflection
        Whether the light level :math`in` to a camera is reflection.
    constants
        *Leica L-Log* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear scene reflection :math:`LSR` values.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``LLog``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``LSR``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`LeicaCameraAG2022`

    Examples
    --------
    >>> log_decoding_LLog(0.43531390404392656)  # doctest: +ELLIPSIS
    0.1800000...
    """

    LLog = to_domain_1(LLog)

    LLog = LLog if in_normalised_code_value else full_to_legal(LLog, bit_depth)

    cut2 = constants.cut2
    a = constants.a
    b = constants.b
    c = constants.c
    d = constants.d
    e = constants.e
    f = constants.f

    LSR = np.where(
        LLog <= cut2,
        (LLog - b) / a,
        (spow(10, (LLog - f) / c) - e) / d,
    )

    if not out_reflection:
        LSR = LSR / 0.9

    return as_float(from_range_1(LSR))
