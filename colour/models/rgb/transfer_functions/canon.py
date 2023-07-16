"""
Canon Log Encodings
===================

Defines the *Canon Log* encodings:

-   :attr:`colour.models.CANON_LOG_ENCODING_METHODS`
-   :func:`colour.models.log_encoding_CanonLog`
-   :attr:`colour.models.CANON_LOG_DECODING_METHODS`
-   :func:`colour.models.log_decoding_CanonLog`
-   :attr:`colour.models.CANON_LOG_2_ENCODING_METHODS`
-   :func:`colour.models.log_encoding_CanonLog2`
-   :attr:`colour.models.CANON_LOG_2_DECODING_METHODS`
-   :func:`colour.models.log_decoding_CanonLog2`
-   :attr:`colour.models.CANON_LOG_3_ENCODING_METHODS`
-   :func:`colour.models.log_encoding_CanonLog3`
-   :attr:`colour.models.CANON_LOG_3_DECODING_METHODS`
-   :func:`colour.models.log_decoding_CanonLog3`

Notes
-----
-   :cite:`Canon2016` is available as a *Drivers & Downloads* *Software* for
    Windows 7 *Operating System*, a copy of the archive is hosted at
    this url: https://drive.google.com/open?id=0B_IQZQdc4Vy8ZGYyY29pMEVwZU0
-   :cite:`Canon2020` is available as a *Drivers & Downloads* *Software* for
    Windows 10 *Operating System*, a copy of the archive is hosted at
    this url: https://drive.google.com/open?id=1Vcz8RVIXgXL54lhZsOwGUjjVZRObZSc5

References
----------
-   :cite:`Canon2016` : Canon. (2016). Input Transform Version 201612 for EOS
    C300 Mark II. Retrieved August 23, 2016, from https://www.usa.canon.com/\
internet/portal/us/home/support/details/cameras/cinema-eos/eos-c300-mark-ii
-   :cite:`Canon2020` : Canon. (2020). Input Transform Version 202007 for EOS
    C300 Mark II. Retrieved July 16, 2023, from https://www.usa.canon.com/\
internet/portal/us/home/support/details/cameras/cinema-eos/eos-c300-mark-ii
-   :cite:`Thorpe2012a` : Thorpe, L. (2012). CANON-LOG TRANSFER CHARACTERISTIC.
    Retrieved September 25, 2014, from
    http://downloads.canon.com/CDLC/Canon-Log_Transfer_Characteristic_6-20-2012.pdf
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, Literal, NDArrayFloat
from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import (
    CanonicalMapping,
    as_float,
    domain_range_scale,
    from_range_1,
    to_domain_1,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "log_encoding_CanonLog_v1",
    "log_decoding_CanonLog_v1",
    "log_encoding_CanonLog_v1_2",
    "log_decoding_CanonLog_v1_2",
    "CANON_LOG_ENCODING_METHODS",
    "log_encoding_CanonLog",
    "CANON_LOG_DECODING_METHODS",
    "log_decoding_CanonLog",
    "log_encoding_CanonLog2_v1",
    "log_decoding_CanonLog2_v1",
    "log_encoding_CanonLog2_v1_2",
    "log_decoding_CanonLog2_v1_2",
    "CANON_LOG_2_ENCODING_METHODS",
    "log_encoding_CanonLog2",
    "CANON_LOG_2_DECODING_METHODS",
    "log_decoding_CanonLog2",
    "log_encoding_CanonLog3_v1",
    "log_decoding_CanonLog3_v1",
    "log_encoding_CanonLog3_v1_2",
    "log_decoding_CanonLog3_v1_2",
    "CANON_LOG_3_ENCODING_METHODS",
    "log_encoding_CanonLog3",
    "CANON_LOG_3_DECODING_METHODS",
    "log_decoding_CanonLog3",
]


def log_encoding_CanonLog_v1(
    x: ArrayLike,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> NDArrayFloat:
    """
    Define the *Canon Log* v1 log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log* non-linear data is encoded as normalised code
        values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log* non-linear data.

    References
    ----------
    :cite:`Canon2016`, :cite:`Thorpe2012a`

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
    | ``clog``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_encoding_CanonLog_v1(0.18) * 100  # doctest: +ELLIPSIS
    34.3389651...

    The values of *Table 2 Canon-Log Code Values* table in :cite:`Thorpe2012a`
    are obtained as follows:

    >>> x = np.array([0, 2, 18, 90, 720]) / 100
    >>> np.around(log_encoding_CanonLog_v1(x) * (2**10 - 1)).astype(np.int_)
    array([ 128,  169,  351,  614, 1016])
    >>> np.around(log_encoding_CanonLog_v1(x, 10, False) * 100, 1)
    array([   7.3,   12. ,   32.8,   62.7,  108.7])
    """

    x = to_domain_1(x)

    if in_reflection:
        x = x / 0.9

    with domain_range_scale("ignore"):
        clog = np.where(
            x < log_decoding_CanonLog_v1(0.0730597, bit_depth, False),
            -(0.529136 * (np.log10(-x * 10.1596 + 1)) - 0.0730597),
            0.529136 * np.log10(10.1596 * x + 1) + 0.0730597,
        )

    clog_cv = (
        full_to_legal(clog, bit_depth) if out_normalised_code_value else clog
    )

    return as_float(from_range_1(clog_cv))


def log_decoding_CanonLog_v1(
    clog: ArrayLike,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> NDArrayFloat:
    """
    Define the *Canon Log* v1 log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog
        *Canon Log* non-linear data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log* non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`, :cite:`Thorpe2012a`

    Examples
    --------
    >>> log_decoding_CanonLog_v1(
    ...     34.338965172606912 / 100
    ... )  # doctest: +ELLIPSIS
    0.17999999...
    """

    clog = to_domain_1(clog)

    clog = legal_to_full(clog, bit_depth) if in_normalised_code_value else clog

    x = np.where(
        clog < 0.0730597,
        -(10 ** ((0.0730597 - clog) / 0.529136) - 1) / 10.1596,
        (10 ** ((clog - 0.0730597) / 0.529136) - 1) / 10.1596,
    )

    if out_reflection:
        x = x * 0.9

    return as_float(from_range_1(x))


def log_encoding_CanonLog_v1_2(
    x: ArrayLike,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> NDArrayFloat:
    """
    Define the *Canon Log* v1.2 log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log* non-linear data is encoded as normalised code
        values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log* non-linear data.

    References
    ----------
    :cite:`Canon2020`

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
    | ``clog``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_encoding_CanonLog_v1_2(0.18) * 100  # doctest: +ELLIPSIS
    34.3389649...
    """

    x = to_domain_1(x)

    if in_reflection:
        x = x / 0.9

    with domain_range_scale("ignore"):
        clog = np.where(
            x < (log_decoding_CanonLog_v1_2(0.12512248, bit_depth, True)),
            -(0.45310179 * (np.log10(-x * 10.1596 + 1)) - 0.12512248),
            0.45310179 * np.log10(10.1596 * x + 1) + 0.12512248,
        )

    # NOTE: *Canon Log* v1.2 constants are expressed in legal range
    # (studio swing).
    clog_cv = (
        clog if out_normalised_code_value else legal_to_full(clog, bit_depth)
    )

    return as_float(from_range_1(clog_cv))


def log_decoding_CanonLog_v1_2(
    clog: ArrayLike,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> NDArrayFloat:
    """
    Define the *Canon Log* v1.2 log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog
        *Canon Log* non-linear data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log* non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2020`

    Examples
    --------
    >>> log_decoding_CanonLog_v1_2(34.338964929528061 / 100)
    ... # doctest: +ELLIPSIS
    0.17999999...
    """

    clog = to_domain_1(clog)

    # NOTE: *Canon Log* v1.2 constants are expressed in legal range
    # (studio swing).
    clog = clog if in_normalised_code_value else full_to_legal(clog, bit_depth)

    x = np.where(
        clog < 0.12512248,
        -(10 ** ((0.12512248 - clog) / 0.45310179) - 1) / 10.1596,
        (10 ** ((clog - 0.12512248) / 0.45310179) - 1) / 10.1596,
    )

    if out_reflection:
        x = x * 0.9

    return as_float(from_range_1(x))


CANON_LOG_ENCODING_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "v1": log_encoding_CanonLog_v1,
        "v1.2": log_encoding_CanonLog_v1_2,
    }
)
CANON_LOG_ENCODING_METHODS.__doc__ = """
Supported *CanonLog* log encoding curve / opto-electronic transfer function
methods.

References
----------
:cite:`Canon2016`, :cite:`Canon2020`
"""


def log_encoding_CanonLog(
    x: ArrayLike,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
    method: Literal["v1", "v1.2"] | str = "v1.2",
) -> NDArrayFloat:
    """
    Define the *Canon Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log* non-linear data is encoded as normalised code
        values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log* non-linear data.

    References
    ----------
    :cite:`Canon2016`, :cite:`Canon2020`, :cite:`Thorpe2012a`

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
    | ``clog``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_encoding_CanonLog(0.18) * 100  # doctest: +ELLIPSIS
    34.3389649...
    >>> log_encoding_CanonLog(0.18, method="v1") * 100  # doctest: +ELLIPSIS
    34.3389651...

    The values of *Table 2 Canon-Log Code Values* table in :cite:`Thorpe2012a`
    are obtained as follows:

    >>> x = np.array([0, 2, 18, 90, 720]) / 100
    >>> np.around(
    ...     log_encoding_CanonLog(x, method="v1") * (2**10 - 1)
    ... ).astype(np.int_)
    array([ 128,  169,  351,  614, 1016])
    >>> np.around(log_encoding_CanonLog(x, 10, False, method="v1") * 100, 1)
    array([   7.3,   12. ,   32.8,   62.7,  108.7])
    """

    method = validate_method(method, tuple(CANON_LOG_ENCODING_METHODS))

    return CANON_LOG_ENCODING_METHODS[method](
        x, bit_depth, out_normalised_code_value, in_reflection
    )


CANON_LOG_DECODING_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "v1": log_decoding_CanonLog_v1,
        "v1.2": log_decoding_CanonLog_v1_2,
    }
)
CANON_LOG_DECODING_METHODS.__doc__ = """
Supported *CanonLog* log decoding curve / electro-optical transfer function
methods.

References
----------
:cite:`Canon2016`, :cite:`Canon2020`
"""


def log_decoding_CanonLog(
    clog: ArrayLike,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
    method: Literal["v1", "v1.2"] | str = "v1.2",
) -> NDArrayFloat:
    """
    Define the *Canon Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog
        *Canon Log* non-linear data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log* non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`, :cite:`Canon2020`, :cite:`Thorpe2012a`

    Examples
    --------
    >>> log_decoding_CanonLog(34.338964929528061 / 100)  # doctest: +ELLIPSIS
    0.17999999...
    >>> log_decoding_CanonLog(34.338965172606912 / 100, method="v1")
    ... # doctest: +ELLIPSIS
    0.17999999...
    """

    method = validate_method(method, tuple(CANON_LOG_DECODING_METHODS))

    return CANON_LOG_DECODING_METHODS[method](
        clog, bit_depth, in_normalised_code_value, out_reflection
    )


def log_encoding_CanonLog2_v1(
    x: ArrayLike,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> NDArrayFloat:
    """
    Define the *Canon Log 2* v1 log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log 2* non-linear data is encoded as normalised
        code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log 2* non-linear data.

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
    | ``clog2``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`

    Examples
    --------
    >>> log_encoding_CanonLog2_v1(0.18) * 100  # doctest: +ELLIPSIS
    39.8254694...
    """

    x = to_domain_1(x)

    if in_reflection:
        x = x / 0.9

    with domain_range_scale("ignore"):
        clog2 = np.where(
            x < log_decoding_CanonLog2_v1(0.035388128, bit_depth, False),
            -(0.281863093 * (np.log10(-x * 87.09937546 + 1)) - 0.035388128),
            0.281863093 * np.log10(x * 87.09937546 + 1) + 0.035388128,
        )

    clog2_cv = (
        full_to_legal(clog2, bit_depth) if out_normalised_code_value else clog2
    )

    return as_float(from_range_1(clog2_cv))


def log_decoding_CanonLog2_v1(
    clog2: ArrayLike,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> NDArrayFloat:
    """
    Define the *Canon Log 2* v1 log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog2
        *Canon Log 2* non-linear data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log 2* non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog2``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`

    Examples
    --------
    >>> log_decoding_CanonLog2_v1(
    ...     39.825469498316735 / 100
    ... )  # doctest: +ELLIPSIS
    0.1799999...
    """

    clog2 = to_domain_1(clog2)

    clog2 = (
        legal_to_full(clog2, bit_depth) if in_normalised_code_value else clog2
    )

    x = np.where(
        clog2 < 0.035388128,
        -(10 ** ((0.035388128 - clog2) / 0.281863093) - 1) / 87.09937546,
        (10 ** ((clog2 - 0.035388128) / 0.281863093) - 1) / 87.09937546,
    )

    if out_reflection:
        x = x * 0.9

    return as_float(from_range_1(x))


def log_encoding_CanonLog2_v1_2(
    x: ArrayLike,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> NDArrayFloat:
    """
    Define the *Canon Log 2* v1.2 log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log 2* non-linear data is encoded as normalised
        code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log 2* non-linear data.

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
    | ``clog2``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2020`

    Examples
    --------
    >>> log_encoding_CanonLog2_v1_2(0.18) * 100  # doctest: +ELLIPSIS
    39.8254692...
    """

    x = to_domain_1(x)

    if in_reflection:
        x = x / 0.9

    with domain_range_scale("ignore"):
        clog2 = np.where(
            x < (log_decoding_CanonLog2_v1_2(0.092864125, bit_depth, True)),
            -(0.24136077 * (np.log10(-x * 87.09937546 + 1)) - 0.092864125),
            0.24136077 * np.log10(x * 87.09937546 + 1) + 0.092864125,
        )

    # NOTE: *Canon Log 2* v1.2 constants are expressed in legal range
    # (studio swing).
    clog2_cv = (
        clog2 if out_normalised_code_value else legal_to_full(clog2, bit_depth)
    )

    return as_float(from_range_1(clog2_cv))


def log_decoding_CanonLog2_v1_2(
    clog2: ArrayLike,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> NDArrayFloat:
    """
    Define the *Canon Log 2* v1.2 log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog2
        *Canon Log 2* non-linear data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log 2* non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog2``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2020`

    Examples
    --------
    >>> log_decoding_CanonLog2_v1_2(39.825469256149191 / 100)
    ... # doctest: +ELLIPSIS
    0.1799999...
    """

    clog2 = to_domain_1(clog2)

    # NOTE: *Canon Log 2* v1.2 constants are expressed in legal range
    # (studio swing).
    clog2 = (
        clog2 if in_normalised_code_value else full_to_legal(clog2, bit_depth)
    )

    x = np.where(
        clog2 < 0.092864125,
        -(10 ** ((0.092864125 - clog2) / 0.24136077) - 1) / 87.09937546,
        (10 ** ((clog2 - 0.092864125) / 0.24136077) - 1) / 87.09937546,
    )

    if out_reflection:
        x = x * 0.9

    return as_float(from_range_1(x))


CANON_LOG_2_ENCODING_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "v1": log_encoding_CanonLog2_v1,
        "v1.2": log_encoding_CanonLog2_v1_2,
    }
)
CANON_LOG_2_ENCODING_METHODS.__doc__ = """
Supported *Canon Log 2* log encoding curve / opto-electronic transfer function
methods.

References
----------
:cite:`Canon2016`, :cite:`Canon2020`
"""


def log_encoding_CanonLog2(
    x: ArrayLike,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
    method: Literal["v1", "v1.2"] | str = "v1.2",
) -> NDArrayFloat:
    """
    Define the *Canon Log 2* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log 2* non-linear data is encoded as normalised
        code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log 2* non-linear data.

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
    | ``clog2``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`, :cite:`Canon2020`

    Examples
    --------
    >>> log_encoding_CanonLog2(0.18) * 100  # doctest: +ELLIPSIS
    39.8254692...
    """

    method = validate_method(method, tuple(CANON_LOG_2_ENCODING_METHODS))

    return CANON_LOG_2_ENCODING_METHODS[method](
        x, bit_depth, out_normalised_code_value, in_reflection
    )


CANON_LOG_2_DECODING_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "v1": log_decoding_CanonLog2_v1,
        "v1.2": log_decoding_CanonLog2_v1_2,
    }
)
CANON_LOG_2_DECODING_METHODS.__doc__ = """
Supported *Canon Log 2* log decoding curve / electro-optical transfer function
methods.

References
----------
:cite:`Canon2016`, :cite:`Canon2020`
"""


def log_decoding_CanonLog2(
    clog2: ArrayLike,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
    method: Literal["v1", "v1.2"] | str = "v1.2",
) -> NDArrayFloat:
    """
    Define the *Canon Log 2* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog2
        *Canon Log 2* non-linear data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log 2* non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog2``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`, :cite:`Canon2020`

    Examples
    --------
    >>> log_decoding_CanonLog2(39.825469256149191 / 100)  # doctest: +ELLIPSIS
    0.1799999...
    """

    method = validate_method(method, tuple(CANON_LOG_2_DECODING_METHODS))

    return CANON_LOG_2_DECODING_METHODS[method](
        clog2, bit_depth, in_normalised_code_value, out_reflection
    )


def log_encoding_CanonLog3_v1(
    x: ArrayLike,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> NDArrayFloat:
    """
    Define the *Canon Log 3* v1 log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log 3* non-linear data is encoded as normalised code
        values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log 3* non-linear data.

    Notes
    -----
    -   Introspection of the grafting points by Shaw, N. (2018) shows that the
        *Canon Log 3* v1 IDT was likely derived from its encoding curve as the
        latter is grafted at *+/-0.014*::

            >>> clog3 = 0.04076162
            >>> (clog3 - 0.073059361) / 2.3069815
            -0.014000000000000002
            >>> clog3 = 0.105357102
            >>> (clog3 - 0.073059361) / 2.3069815
            0.013999999999999997

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog3``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`

    Examples
    --------
    >>> log_encoding_CanonLog3_v1(0.18) * 100  # doctest: +ELLIPSIS
    34.3389369...
    """

    x = to_domain_1(x)

    if in_reflection:
        x = x / 0.9

    with domain_range_scale("ignore"):
        clog3 = np.select(
            (
                x
                < log_decoding_CanonLog3_v1(
                    0.04076162, bit_depth, False, False
                ),
                x
                <= log_decoding_CanonLog3_v1(
                    0.105357102, bit_depth, False, False
                ),
                x
                > log_decoding_CanonLog3_v1(
                    0.105357102, bit_depth, False, False
                ),
            ),
            (
                -0.42889912 * np.log10(-x * 14.98325 + 1) + 0.07623209,
                2.3069815 * x + 0.073059361,
                0.42889912 * np.log10(x * 14.98325 + 1) + 0.069886632,
            ),
        )

    clog3_cv = (
        full_to_legal(clog3, bit_depth) if out_normalised_code_value else clog3
    )

    return as_float(from_range_1(clog3_cv))


def log_decoding_CanonLog3_v1(
    clog3: ArrayLike,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> NDArrayFloat:
    """
    Define the *Canon Log 3* v1 log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog3
        *Canon Log 3* non-linear data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log 3* non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog3``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`

    Examples
    --------
    >>> log_decoding_CanonLog3_v1(
    ...     34.338936938868677 / 100
    ... )  # doctest: +ELLIPSIS
    0.1800000...
    """

    clog3 = to_domain_1(clog3)

    clog3 = (
        legal_to_full(clog3, bit_depth) if in_normalised_code_value else clog3
    )

    x = np.select(
        (clog3 < 0.04076162, clog3 <= 0.105357102, clog3 > 0.105357102),
        (
            -(10 ** ((0.07623209 - clog3) / 0.42889912) - 1) / 14.98325,
            (clog3 - 0.073059361) / 2.3069815,
            (10 ** ((clog3 - 0.069886632) / 0.42889912) - 1) / 14.98325,
        ),
    )

    if out_reflection:
        x = x * 0.9

    return as_float(from_range_1(x))


def log_encoding_CanonLog3_v1_2(
    x: ArrayLike,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> NDArrayFloat:
    """
    Define the *Canon Log 3* v1.2 log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log 3* non-linear data is encoded as normalised code
        values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log 3* non-linear data.

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
    | ``clog3``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2020`

    Examples
    --------
    >>> log_encoding_CanonLog3_v1_2(0.18) * 100  # doctest: +ELLIPSIS
    34.3389370...
    """

    x = to_domain_1(x)

    if in_reflection:
        x = x / 0.9

    with domain_range_scale("ignore"):
        clog3 = np.select(
            (
                x
                < log_decoding_CanonLog3_v1_2(
                    0.097465473, bit_depth, True, False
                ),
                x
                <= log_decoding_CanonLog3_v1_2(
                    0.15277891, bit_depth, True, False
                ),
                x
                > log_decoding_CanonLog3_v1_2(
                    0.15277891, bit_depth, True, False
                ),
            ),
            (
                -0.36726845 * np.log10(-x * 14.98325 + 1) + 0.12783901,
                1.9754798 * x + 0.12512219,
                0.36726845 * np.log10(x * 14.98325 + 1) + 0.12240537,
            ),
        )

    # NOTE: *Canon Log 3* v1.2 constants are expressed in legal range
    # (studio swing).
    clog3_cv = (
        clog3 if out_normalised_code_value else legal_to_full(clog3, bit_depth)
    )

    return as_float(from_range_1(clog3_cv))


def log_decoding_CanonLog3_v1_2(
    clog3: ArrayLike,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> NDArrayFloat:
    """
    Define the *Canon Log 3* v1.2 log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog3
        *Canon Log 3* non-linear data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log 3* non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog3``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2020`

    Examples
    --------
    >>> log_decoding_CanonLog3_v1_2(34.338937037393549 / 100)
    ... # doctest: +ELLIPSIS
    0.1799999...
    """

    clog3 = to_domain_1(clog3)

    # NOTE: *Canon Log 3* v1.2 constants are expressed in legal range
    # (studio swing).
    clog3 = (
        clog3 if in_normalised_code_value else full_to_legal(clog3, bit_depth)
    )

    x = np.select(
        (clog3 < 0.097465473, clog3 <= 0.15277891, clog3 > 0.15277891),
        (
            -(10 ** ((0.12783901 - clog3) / 0.36726845) - 1) / 14.98325,
            (clog3 - 0.12512219) / 1.9754798,
            (10 ** ((clog3 - 0.12240537) / 0.36726845) - 1) / 14.98325,
        ),
    )

    if out_reflection:
        x = x * 0.9

    return as_float(from_range_1(x))


CANON_LOG_3_ENCODING_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "v1": log_encoding_CanonLog3_v1,
        "v1.2": log_encoding_CanonLog3_v1_2,
    }
)
CANON_LOG_3_ENCODING_METHODS.__doc__ = """
Supported *Canon Log 3* log encoding curve / opto-electronic transfer function
methods.

References
----------
:cite:`Canon2016`, :cite:`Canon2020`
"""


def log_encoding_CanonLog3(
    x: ArrayLike,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
    method: Literal["v1", "v1.2"] | str = "v1.2",
) -> NDArrayFloat:
    """
    Define the *Canon Log 3* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log 3* non-linear data is encoded as normalised
        code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log 3* non-linear data.

    Notes
    -----
    -   Introspection of the grafting points by Shaw, N. (2018) shows that the
        *Canon Log 3* v1 IDT was likely derived from its encoding curve as the
        latter is grafted at *+/-0.014*::

            >>> clog3 = 0.04076162
            >>> (clog3 - 0.073059361) / 2.3069815
            -0.014000000000000002
            >>> clog3 = 0.105357102
            >>> (clog3 - 0.073059361) / 2.3069815
            0.013999999999999997

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog2``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`, :cite:`Canon2020`

    Examples
    --------
    >>> log_encoding_CanonLog3(0.18) * 100  # doctest: +ELLIPSIS
    34.3389370...
    """

    method = validate_method(method, tuple(CANON_LOG_3_ENCODING_METHODS))

    return CANON_LOG_3_ENCODING_METHODS[method](
        x, bit_depth, out_normalised_code_value, in_reflection
    )


CANON_LOG_3_DECODING_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "v1": log_decoding_CanonLog3_v1,
        "v1.2": log_decoding_CanonLog3_v1_2,
    }
)
CANON_LOG_3_DECODING_METHODS.__doc__ = """
Supported *Canon Log 3* log decoding curve / electro-optical transfer function
methods.

References
----------
:cite:`Canon2016`, :cite:`Canon2020`
"""


def log_decoding_CanonLog3(
    clog3: ArrayLike,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
    method: Literal["v1", "v1.2"] | str = "v1.2",
) -> NDArrayFloat:
    """
    Define the *Canon Log 3* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog3
        *Canon Log 3* non-linear data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log 3* non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog2``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`, :cite:`Canon2020`

    Examples
    --------
    >>> log_decoding_CanonLog3(34.338937037393549 / 100)  # doctest: +ELLIPSIS
    0.1799999...
    """

    method = validate_method(method, tuple(CANON_LOG_3_DECODING_METHODS))

    return CANON_LOG_3_DECODING_METHODS[method](
        clog3, bit_depth, in_normalised_code_value, out_reflection
    )
