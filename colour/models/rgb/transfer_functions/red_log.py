# -*- coding: utf-8 -*-
"""
RED Log Encodings
=================

Defines the *RED* log encodings:

-   :func:`colour.models.log_encoding_REDLog`
-   :func:`colour.models.log_decoding_REDLog`
-   :func:`colour.models.log_encoding_REDLogFilm`
-   :func:`colour.models.log_decoding_REDLogFilm`
-   :func:`colour.models.log_encoding_Log3G10_v1`
-   :func:`colour.models.log_decoding_Log3G10_v1`
-   :func:`colour.models.log_encoding_Log3G10_v2`
-   :func:`colour.models.log_decoding_Log3G10_v2`
-   :func:`colour.models.log_encoding_Log3G10_v3`
-   :func:`colour.models.log_decoding_Log3G10_v3`
-   :attr:`colour.models.LOG3G10_ENCODING_METHODS`
-   :func:`colour.models.log_encoding_Log3G10`
-   :attr:`colour.models.LOG3G10_DECODING_METHODS`
-   :func:`colour.models.log_decoding_Log3G10`
-   :func:`colour.models.log_encoding_Log3G12`
-   :func:`colour.models.log_decoding_Log3G12`

References
----------
-   :cite:`Nattress2016a` : Nattress, G. (2016). Private Discussion with Shaw,
    N.
-   :cite:`REDDigitalCinema2017` : RED Digital Cinema. (2017). White Paper on
    REDWideGamutRGB and Log3G10. Retrieved January 16, 2021, from
    https://www.red.com/download/white-paper-on-redwidegamutrgb-and-log3g10
-   :cite:`SonyImageworks2012a` : Sony Imageworks. (2012). make.py. Retrieved
    November 27, 2014, from
    https://github.com/imageworks/OpenColorIO-Configs/blob/master/\
nuke-default/make.py
"""

from __future__ import annotations

import numpy as np

from colour.hints import (
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Literal,
    Union,
)
from colour.models.rgb.transfer_functions import (
    log_encoding_Cineon,
    log_decoding_Cineon,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    as_float,
    as_float_array,
    from_range_1,
    to_domain_1,
    validate_method,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'log_encoding_REDLog',
    'log_decoding_REDLog',
    'log_encoding_REDLogFilm',
    'log_decoding_REDLogFilm',
    'log_encoding_Log3G10_v1',
    'log_decoding_Log3G10_v1',
    'log_encoding_Log3G10_v2',
    'log_decoding_Log3G10_v2',
    'log_encoding_Log3G10_v3',
    'log_decoding_Log3G10_v3',
    'LOG3G10_ENCODING_METHODS',
    'log_encoding_Log3G10',
    'LOG3G10_DECODING_METHODS',
    'log_decoding_Log3G10',
    'log_encoding_Log3G12',
    'log_decoding_Log3G12',
]


def log_encoding_REDLog(x: FloatingOrArrayLike,
                        black_offset: FloatingOrArrayLike = 10
                        ** ((0 - 1023) / 511)) -> FloatingOrNDArray:
    """
    Defines the *REDLog* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    black_offset
        Black offset.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear data :math:`y`.

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
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_encoding_REDLog(0.18)  # doctest: +ELLIPSIS
    0.6376218...
    """

    x = to_domain_1(x)
    black_offset = as_float_array(black_offset)

    y = (1023 + 511 * np.log10(x * (1 - black_offset) + black_offset)) / 1023

    return as_float(from_range_1(y))


def log_decoding_REDLog(y: FloatingOrArrayLike,
                        black_offset: FloatingOrArrayLike = 10
                        ** ((0 - 1023) / 511)) -> FloatingOrNDArray:
    """
    Defines the *REDLog* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y
        Non-linear data :math:`y`.
    black_offset
        Black offset.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear data :math:`x`.

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
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_decoding_REDLog(0.637621845988175)  # doctest: +ELLIPSIS
    0.1...
    """

    y = to_domain_1(y)
    black_offset = as_float_array(black_offset)

    x = ((10 ** ((1023 * y - 1023) / 511)) - black_offset) / (1 - black_offset)

    return as_float(from_range_1(x))


def log_encoding_REDLogFilm(x: FloatingOrArrayLike,
                            black_offset: FloatingOrArrayLike = 10
                            ** ((95 - 685) / 300)) -> FloatingOrNDArray:
    """
    Defines the *REDLogFilm* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    black_offset
        Black offset.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear data :math:`y`.

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
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_encoding_REDLogFilm(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    """

    return log_encoding_Cineon(x, black_offset)


def log_decoding_REDLogFilm(y: FloatingOrArrayLike,
                            black_offset: FloatingOrArrayLike = 10
                            ** ((95 - 685) / 300)) -> FloatingOrNDArray:
    """
    Defines the *REDLogFilm* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y
        Non-linear data :math:`y`.
    black_offset
        Black offset.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear data :math:`x`.

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
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_decoding_REDLogFilm(0.457319613085418)  # doctest: +ELLIPSIS
    0.1799999...
    """

    return log_decoding_Cineon(y, black_offset)


def log_encoding_Log3G10_v1(x: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines the *Log3G10* *v1* log encoding curve / opto-electronic transfer
    function, the curve used in *REDCINE-X PRO Beta 42* and *Resolve 12.5.2*.

    Parameters
    ----------
    x
        Linear data :math:`x`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear data :math:`y`.

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
    :cite:`Nattress2016a`

    Examples
    --------
    >>> log_encoding_Log3G10_v1(0.18)  # doctest: +ELLIPSIS
    0.3333336...
    """

    x = to_domain_1(x)

    y = np.sign(x) * 0.222497 * np.log10((np.abs(x) * 169.379333) + 1)

    return as_float(from_range_1(y))


def log_decoding_Log3G10_v1(y: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines the *Log3G10* *v1* log decoding curve / electro-optical transfer
    function, the curve used in *REDCINE-X PRO Beta 42* and *Resolve 12.5.2*.

    Parameters
    ----------
    y
        Non-linear data :math:`y`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear data :math:`x`.

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
    :cite:`Nattress2016a`

    Examples
    --------
    >>> log_decoding_Log3G10_v1(1.0 / 3)  # doctest: +ELLIPSIS
    0.1799994...
    """

    y = to_domain_1(y)

    x = (np.sign(y) * (10.0 ** (np.abs(y) / 0.222497) - 1) / 169.379333)

    return as_float(from_range_1(x))


def log_encoding_Log3G10_v2(x: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines the *Log3G10* *v2* log encoding curve / opto-electronic transfer
    function, the current curve in *REDCINE-X PRO*.

    Parameters
    ----------
    x
        Linear data :math:`x`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear data :math:`y`.

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
    :cite:`Nattress2016a`

    Examples
    --------
    >>> log_encoding_Log3G10_v2(0.0)  # doctest: +ELLIPSIS
    0.0915514...
    """

    x = to_domain_1(x)

    y = (np.sign(x + 0.01) * 0.224282 *
         np.log10((np.abs(x + 0.01) * 155.975327) + 1))

    return as_float(from_range_1(y))


def log_decoding_Log3G10_v2(y: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines the *Log3G10* *v2* log decoding curve / electro-optical transfer
    function, the current curve in *REDCINE-X PRO*.

    Parameters
    ----------
    y
        Non-linear data :math:`y`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear data :math:`x`.

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
    :cite:`Nattress2016a`

    Examples
    --------
    >>> log_decoding_Log3G10_v2(1.0)  # doctest: +ELLIPSIS
    184.3223476...
    """

    y = to_domain_1(y)

    x = (np.sign(y) * (10.0 ** (np.abs(y) / 0.224282) - 1) / 155.975327) - 0.01

    return as_float(from_range_1(x))


def log_encoding_Log3G10_v3(x: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines the *Log3G10* *v3* log encoding curve / opto-electronic transfer
    function, the curve described in the *RedLog3G10* Whitepaper.

    Parameters
    ----------
    x
        Linear data :math:`x`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear data :math:`y`.

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
    :cite:`REDDigitalCinema2017`

    Examples
    --------
    >>> log_encoding_Log3G10_v3(0.0)  # doctest: +ELLIPSIS
    0.09155148...
    """

    a = 0.224282
    b = 155.975327
    c = 0.01
    g = 15.1927

    x = to_domain_1(x)

    x = x + c

    y = np.where(x < 0.0, x * g,
                 np.sign(x) * a * np.log10((np.abs(x) * b) + 1.0))

    return as_float(from_range_1(y))


def log_decoding_Log3G10_v3(y: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines the *Log3G10* *v3* log decoding curve / electro-optical transfer
    function, the curve described in the *RedLog3G10* whitepaper.

    Parameters
    ----------
    y
        Non-linear data :math:`y`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear data :math:`x`.

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
    :cite:`REDDigitalCinema2017`

    Examples
    --------
    >>> log_decoding_Log3G10_v3(1.0)  # doctest: +ELLIPSIS
    184.32234764...
    """

    a = 0.224282
    b = 155.975327
    c = 0.01
    g = 15.1927

    y = to_domain_1(y)

    x = np.where(y < 0.0, (y / g) - c,
                 np.sign(y) * (10 ** (np.abs(y) / a) - 1.0) / b - c)

    return as_float(from_range_1(x))


LOG3G10_ENCODING_METHODS: CaseInsensitiveMapping = CaseInsensitiveMapping({
    'v1': log_encoding_Log3G10_v1,
    'v2': log_encoding_Log3G10_v2,
    'v3': log_encoding_Log3G10_v3,
})
LOG3G10_ENCODING_METHODS.__doc__ = """
Supported *Log3G10* log encoding curve / opto-electronic transfer function
methods.

References
----------
:cite:`Nattress2016a`, :cite:`REDDigitalCinema2017`
"""


def log_encoding_Log3G10(x: FloatingOrArrayLike,
                         method: Union[Literal['v1', 'v2', 'v3'], str] = 'v3'
                         ) -> FloatingOrNDArray:
    """
    Defines the *Log3G10* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear data :math:`y`.

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

    -   The *Log3G10* *v1* log encoding curve is the one used in
        *REDCINE-X Beta 42*. *Resolve 12.5.2* also uses the *v1* curve. *RED*
        is planning to use the *Log3G10* *v2* log encoding curve in the release
        version of the *RED SDK*.
    -   The intent of the *Log3G10* *v1* log encoding curve is that zero maps
        to zero, 0.18 maps to 1/3, and 10 stops above 0.18 maps to 1.0.
        The name indicates this in a similar way to the naming conventions of
        *Sony HyperGamma* curves.

        The constants used in the functions do not in fact quite hit these
        values, but rather than use corrected constants, the functions here
        use the official *RED* values, in order to match the output of the
        *RED SDK*.

        For those interested, solving for constants which exactly hit 1/3
        and 1.0 yields the following values::

            B = 25 * (np.sqrt(4093.0) - 3) / 9
            A = 1 / np.log10(B * 184.32 + 1)

        where the function takes the form::

            Log3G10(x) = A * np.log10(B * x + 1)

        Similarly for *Log3G12*, the values which hit exactly 1/3 and 1.0
        are::

            B = 25 * (np.sqrt(16381.0) - 3) / 9
            A = 1 / np.log10(B * 737.28 + 1)

    References
    ----------
    :cite:`Nattress2016a`, :cite:`REDDigitalCinema2017`

    Examples
    --------
    >>> log_encoding_Log3G10(0.0)  # doctest: +ELLIPSIS
    0.09155148...
    >>> log_encoding_Log3G10(0.18, method='v1')  # doctest: +ELLIPSIS
    0.3333336...
    """

    method = validate_method(method, LOG3G10_ENCODING_METHODS)

    return LOG3G10_ENCODING_METHODS[method](x)


LOG3G10_DECODING_METHODS = CaseInsensitiveMapping({
    'v1': log_decoding_Log3G10_v1,
    'v2': log_decoding_Log3G10_v2,
    'v3': log_decoding_Log3G10_v3,
})
LOG3G10_DECODING_METHODS.__doc__ = """
Supported *Log3G10* log decoding curve / electro-optical transfer function
methods.

References
----------
:cite:`Nattress2016a`, :cite:`REDDigitalCinema2017`
"""


def log_decoding_Log3G10(y,
                         method: Union[Literal['v1', 'v2', 'v3'], str] = 'v3'):
    """
    Defines the *Log3G10* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y
        Non-linear data :math:`y`.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear data :math:`x`.

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
    :cite:`Nattress2016a`, :cite:`REDDigitalCinema2017`

    Examples
    --------
    >>> log_decoding_Log3G10(1.0)  # doctest: +ELLIPSIS
    184.3223476...
    >>> log_decoding_Log3G10(1.0 / 3, method='v1')  # doctest: +ELLIPSIS
    0.1799994...
    """

    method = validate_method(method, LOG3G10_DECODING_METHODS)

    return LOG3G10_DECODING_METHODS[method](y)


def log_encoding_Log3G12(x: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines the *Log3G12* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear data :math:`y`.

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
    :cite:`Nattress2016a`, :cite:`REDDigitalCinema2017`

    Examples
    --------
    >>> log_encoding_Log3G12(0.18)  # doctest: +ELLIPSIS
    0.3333326...
    """

    x = to_domain_1(x)

    y = np.sign(x) * 0.184904 * np.log10((np.abs(x) * 347.189667) + 1)

    return as_float(from_range_1(y))


def log_decoding_Log3G12(y: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines the *Log3G12* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y
        Non-linear data :math:`y`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear data :math:`x`.

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
    :cite:`Nattress2016a`, :cite:`REDDigitalCinema2017`

    Examples
    --------
    >>> log_decoding_Log3G12(1.0 / 3)  # doctest: +ELLIPSIS
    0.1800015...
    """

    y = to_domain_1(y)

    x = np.sign(y) * (10.0 ** (np.abs(y) / 0.184904) - 1) / 347.189667

    return as_float(from_range_1(x))
