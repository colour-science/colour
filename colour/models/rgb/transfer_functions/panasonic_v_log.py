"""
Panasonic V-Log Log Encoding
============================

Defines the *Panasonic V-Log* log encoding:

-   :func:`colour.models.log_encoding_VLog`
-   :func:`colour.models.log_decoding_VLog`

References
----------
-   :cite:`Panasonic2014a` : Panasonic. (2014). VARICAM V-Log/V-Gamut (pp.
    1-7).
    http://pro-av.panasonic.net/en/varicam/common/pdf/VARICAM_V-Log_V-Gamut.pdf
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, NDArrayFloat
from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import Structure, as_float, from_range_1, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANTS_VLOG",
    "log_encoding_VLog",
    "log_decoding_VLog",
]

CONSTANTS_VLOG: Structure = Structure(
    cut1=0.01, cut2=0.181, b=0.00873, c=0.241514, d=0.598206
)
"""*Panasonic V-Log* constants."""


def log_encoding_VLog(
    L_in: ArrayLike,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
    constants: Structure = CONSTANTS_VLOG,
) -> NDArrayFloat:
    """
    Define the *Panasonic V-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    L_in
        Linear reflection data :math`L_{in}`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the non-linear *Panasonic V-Log* data :math:`V_{out}` is
        encoded as normalised code values.
    in_reflection
        Whether the light level :math`L_{in}` to a camera is reflection.
    constants
        *Panasonic V-Log* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Non-linear data :math:`V_{out}`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_in``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V_out``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Panasonic2014a`

    Examples
    --------
    >>> log_encoding_VLog(0.18)  # doctest: +ELLIPSIS
    0.4233114...

    The values of *Fig.2.2 V-Log Code Value* table in :cite:`Panasonic2014a`
    are obtained as follows:

    >>> L_in = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_VLog(L_in, 10, False) * 100).astype(np.int_)
    array([ 7, 42, 61])
    >>> np.around(log_encoding_VLog(L_in) * (2**10 - 1)).astype(np.int_)
    array([128, 433, 602])
    >>> np.around(log_encoding_VLog(L_in) * (2**12 - 1)).astype(np.int_)
    array([ 512, 1733, 2409])

    Note that some values in the last column values of
    *Fig.2.2 V-Log Code Value* table in :cite:`Panasonic2014a` are different
    by a code: [512, 1732, 2408].
    """

    L_in = to_domain_1(L_in)

    if not in_reflection:
        L_in = L_in * 0.9

    cut1 = constants.cut1
    b = constants.b
    c = constants.c
    d = constants.d

    V_out = np.where(
        L_in < cut1,
        5.6 * L_in + 0.125,
        c * np.log10(L_in + b) + d,
    )

    V_out_cv = (
        V_out if out_normalised_code_value else legal_to_full(V_out, bit_depth)
    )

    return as_float(from_range_1(V_out_cv))


def log_decoding_VLog(
    V_out: ArrayLike,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
    constants: Structure = CONSTANTS_VLOG,
) -> NDArrayFloat:
    """
    Define the *Panasonic V-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    V_out
        Non-linear data :math:`V_{out}`.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the non-linear *Panasonic V-Log* data :math:`V_{out}` is
        encoded as normalised code values.
    out_reflection
        Whether the light level :math`L_{in}` to a camera is reflection.
    constants
        *Panasonic V-Log* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear reflection data :math`L_{in}`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V_out``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_in``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Panasonic2014a`

    Examples
    --------
    >>> log_decoding_VLog(0.423311448760136)  # doctest: +ELLIPSIS
    0.1799999...
    """

    V_out = to_domain_1(V_out)

    V_out = (
        V_out if in_normalised_code_value else full_to_legal(V_out, bit_depth)
    )

    cut2 = constants.cut2
    b = constants.b
    c = constants.c
    d = constants.d

    L_in = np.where(
        V_out < cut2,
        (V_out - 0.125) / 5.6,
        10 ** ((V_out - d) / c) - b,
    )

    if not out_reflection:
        L_in = L_in / 0.9

    return as_float(from_range_1(L_in))
