"""
Nikon N-Log Log Encoding
========================

Define the *Nikon N-Log* log encoding:

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
from colour.hints import ArrayLike, NDArrayFloat
from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import Structure, as_float, from_range_1, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANTS_NLOG",
    "log_encoding_NLog",
    "log_decoding_NLog",
]

CONSTANTS_NLOG: Structure = Structure(
    cut1=0.328,
    cut2=(452 / 1023),
    a=(650 / 1023),
    b=0.0075,
    c=(150 / 1023),
    d=(619 / 1023),
)
"""*Nikon N-Log* constants."""


def log_encoding_NLog(
    y: ArrayLike,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
    constants: Structure = CONSTANTS_NLOG,
) -> NDArrayFloat:
    """
    Define the *Nikon N-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    y
        Reflectance :math:`y`, "y = 0.18" is equivalent to Stop 0.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the non-linear *Nikon N-Log* data :math:`x` is encoded as
        normalised code values.
    in_reflection
        Whether the light level :math`in` to a camera is reflection.
    constants
        *Nikon N-Log* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        *N-Log* 10-bit equivalent code value :math:`x`.

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
    :cite:`Nikon2018`

    Examples
    --------
    >>> log_encoding_NLog(0.18)  # doctest: +ELLIPSIS
    0.3636677...
    """

    y = to_domain_1(y)

    if not in_reflection:
        y = y * 0.9

    cut1 = constants.cut1
    a = constants.a
    b = constants.b
    c = constants.c
    d = constants.d

    x = np.where(
        y < cut1,
        a * spow(y + b, 1 / 3),
        c * np.log(y) + d,
    )

    x_cv = x if out_normalised_code_value else legal_to_full(x, bit_depth)

    return as_float(from_range_1(x_cv))


def log_decoding_NLog(
    x: ArrayLike,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
    constants: Structure = CONSTANTS_NLOG,
) -> NDArrayFloat:
    """
    Define the *Nikon N-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    x
        *N-Log* 10-bit equivalent code value :math:`x`
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the non-linear *Nikon N-Log* data :math:`x` is encoded as
        normalised code values.
    out_reflection
        Whether the light level :math`in` to a camera is reflection.
    constants
        *Nikon N-Log* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Reflectance :math:`y`.

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
    :cite:`Nikon2018`

    Examples
    --------
    >>> log_decoding_NLog(0.36366777011713869)  # doctest: +ELLIPSIS
    0.1799999...
    """

    x = to_domain_1(x)

    x = x if in_normalised_code_value else full_to_legal(x, bit_depth)

    cut2 = constants.cut2
    a = constants.a
    b = constants.b
    c = constants.c
    d = constants.d

    y = np.where(
        x < cut2,
        spow(x / a, 3) - b,
        np.exp((x - d) / c),
    )

    if not out_reflection:
        y = y / 0.9

    return as_float(from_range_1(y))
