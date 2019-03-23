# -*- coding: utf-8 -*-
"""
RIMM, ROMM and ERIMM Encodings
==============================

Defines the *RIMM, ROMM and ERIMM* encodings opto-electrical transfer functions
(OETF / OECF) and electro-optical transfer functions (EOTF / EOCF):

-   :func:`colour.models.oetf_ROMMRGB`
-   :func:`colour.models.eotf_ROMMRGB`
-   :func:`colour.models.oetf_ProPhotoRGB`
-   :func:`colour.models.eotf_ProPhotoRGB`
-   :func:`colour.models.oetf_RIMMRGB`
-   :func:`colour.models.eotf_RIMMRGB`
-   :func:`colour.models.log_encoding_ERIMMRGB`
-   :func:`colour.models.log_decoding_ERIMMRGB`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`ANSI2003a` : ANSI. (2003). Specification of ROMM RGB. Retrieved from
    http://www.color.org/ROMMRGB.pdf
-   :cite:`Spaulding2000b` : Spaulding, K. E., Woolfe, G. J., &
    Giorgianni, E. J. (2000). Reference Input/Output Medium Metric RGB Color
    Encodings (RIMM/ROMM RGB). Retrieved from
    http://www.photo-lovers.org/pdf/color/romm.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import spow
from colour.utilities import (as_float, as_int, domain_range_scale,
                              from_range_1, to_domain_1)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'oetf_ROMMRGB', 'eotf_ROMMRGB', 'oetf_ProPhotoRGB', 'eotf_ProPhotoRGB',
    'oetf_RIMMRGB', 'eotf_RIMMRGB', 'log_encoding_ERIMMRGB',
    'log_decoding_ERIMMRGB'
]


def oetf_ROMMRGB(X, bit_depth=8, out_int=False):
    """
    Defines the *ROMM RGB* encoding opto-electronic transfer function
    (OETF / OECF).

    Parameters
    ----------
    X : numeric or array_like
        Linear data :math:`X_{ROMM}`.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_int : bool, optional
        Whether to return value as integer code value or float equivalent of a
        code value at a given bit depth.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`X'_{ROMM}`.

    Notes
    -----

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``X``          | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``X_p``        | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    -   \\* This definition has an output integer switch, thus the domain-range
        scale information is only given for the floating point mode.

    References
    ----------
    :cite:`ANSI2003a`, :cite:`Spaulding2000b`

    Examples
    --------
    >>> oetf_ROMMRGB(0.18)  # doctest: +ELLIPSIS
    0.3857114...
    >>> oetf_ROMMRGB(0.18, out_int=True)
    98
    """

    X = to_domain_1(X)

    I_max = 2 ** bit_depth - 1

    E_t = 16 ** (1.8 / (1 - 1.8))

    X_p = np.where(X < E_t, X * 16 * I_max, spow(X, 1 / 1.8) * I_max)

    if out_int:
        return as_int(np.round(X_p))
    else:
        return as_float(from_range_1(X_p / I_max))


def eotf_ROMMRGB(X_p, bit_depth=8, in_int=False):
    """
    Defines the *ROMM RGB* encoding electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    X_p : numeric or array_like
        Non-linear data :math:`X'_{ROMM}`.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_int : bool, optional
        Whether to treat the input value as integer code value or float
        equivalent of a code value at a given bit depth.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`X_{ROMM}`.

    Notes
    -----

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``X_p``        | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``X``          | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    -   \\* This definition has an input integer switch, thus the domain-range
        scale information is only given for the floating point mode.

    References
    ----------
    :cite:`ANSI2003a`, :cite:`Spaulding2000b`

    Examples
    --------
    >>> eotf_ROMMRGB(0.385711424751138) # doctest: +ELLIPSIS
    0.1...
    >>> eotf_ROMMRGB(98, in_int=True) # doctest: +ELLIPSIS
    0.1...
    """

    X_p = to_domain_1(X_p)

    I_max = 2 ** bit_depth - 1

    if not in_int:
        X_p = X_p * I_max

    E_t = 16 ** (1.8 / (1 - 1.8))

    X = np.where(
        X_p < 16 * E_t * I_max,
        X_p / (16 * I_max),
        spow(X_p / I_max, 1.8),
    )

    return as_float(from_range_1(X))


oetf_ProPhotoRGB = oetf_ROMMRGB
eotf_ProPhotoRGB = eotf_ROMMRGB


def oetf_RIMMRGB(X, bit_depth=8, out_int=False, E_clip=2.0):
    """
    Defines the *RIMM RGB* encoding opto-electronic transfer function
    (OETF / OECF).

    *RIMM RGB* encoding non-linearity is based on that specified by
    *Recommendation ITU-R BT.709-6*.

    Parameters
    ----------
    X : numeric or array_like
        Linear data :math:`X_{RIMM}`.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_int : bool, optional
        Whether to return value as integer code value or float equivalent of a
        code value at a given bit depth.
    E_clip : numeric, optional
        Maximum exposure level.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`X'_{RIMM}`.

    Notes
    -----

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``X``          | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``X_p``        | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    -   \\* This definition has an output integer switch, thus the domain-range
        scale information is only given for the floating point mode.

    References
    ----------
    :cite:`Spaulding2000b`

    Examples
    --------
    >>> oetf_RIMMRGB(0.18)  # doctest: +ELLIPSIS
    0.2916737...
    >>> oetf_RIMMRGB(0.18, out_int=True)
    74
    """

    X = to_domain_1(X)

    I_max = 2 ** bit_depth - 1

    V_clip = 1.099 * spow(E_clip, 0.45) - 0.099
    q = I_max / V_clip

    X_p = q * np.select([X < 0.0, X < 0.018, X >= 0.018, X > E_clip],
                        [0, 4.5 * X, 1.099 * spow(X, 0.45) - 0.099, I_max])

    if out_int:
        return as_int(np.round(X_p))
    else:
        return as_float(from_range_1(X_p / I_max))


def eotf_RIMMRGB(X_p, bit_depth=8, in_int=False, E_clip=2.0):
    """
    Defines the *RIMM RGB* encoding electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    X_p : numeric or array_like
        Non-linear data :math:`X'_{RIMM}`.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_int : bool, optional
        Whether to treat the input value as integer code value or float
        equivalent of a code value at a given bit depth.
    E_clip : numeric, optional
        Maximum exposure level.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`X_{RIMM}`.

    Notes
    -----

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``X_p``        | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``X``          | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    -   \\* This definition has an input integer switch, thus the domain-range
        scale information is only given for the floating point mode.

    References
    ----------
    :cite:`Spaulding2000b`

    Examples
    --------
    >>> eotf_RIMMRGB(0.291673732475746)  # doctest: +ELLIPSIS
    0.1...
    >>> eotf_RIMMRGB(74, in_int=True)  # doctest: +ELLIPSIS
    0.1...
    """

    X_p = to_domain_1(X_p)

    I_max = 2 ** bit_depth - 1

    if not in_int:
        X_p = X_p * I_max

    V_clip = 1.099 * spow(E_clip, 0.45) - 0.099

    m = V_clip * X_p / I_max

    with domain_range_scale('ignore'):
        X = np.where(
            X_p / I_max < oetf_RIMMRGB(0.018, bit_depth, E_clip=E_clip),
            m / 4.5,
            spow((m + 0.099) / 1.099, 1 / 0.45),
        )

    return as_float(from_range_1(X))


def log_encoding_ERIMMRGB(X,
                          bit_depth=8,
                          out_int=False,
                          E_min=0.001,
                          E_clip=316.2):
    """
    Defines the *ERIMM RGB* log encoding curve / opto-electronic transfer
    function (OETF / OECF).

    Parameters
    ----------
    X : numeric or array_like
        Linear data :math:`X_{ERIMM}`.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_int : bool, optional
        Whether to return value as integer code value or float equivalent of a
        code value at a given bit depth.
    E_min : numeric, optional
        Minimum exposure limit.
    E_clip : numeric, optional
        Maximum exposure limit.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`X'_{ERIMM}`.

    Notes
    -----

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``X``          | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``X_p``        | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    -   \\* This definition has an output integer switch, thus the domain-range
        scale information is only given for the floating point mode.

    References
    ----------
    :cite:`Spaulding2000b`

    Examples
    --------
    >>> log_encoding_ERIMMRGB(0.18)  # doctest: +ELLIPSIS
    0.4100523...
    >>> log_encoding_ERIMMRGB(0.18, out_int=True)
    105
    """

    X = to_domain_1(X)

    I_max = 2 ** bit_depth - 1

    E_t = np.exp(1) * E_min

    X_p = np.select([
        X < 0.0,
        X <= E_t,
        X > E_t,
        X > E_clip,
    ], [
        0,
        I_max * ((np.log(E_t) - np.log(E_min)) /
                 (np.log(E_clip) - np.log(E_min))) * (X / E_t),
        I_max * (
            (np.log(X) - np.log(E_min)) / (np.log(E_clip) - np.log(E_min))),
        I_max,
    ])

    if out_int:
        return as_int(np.round(X_p))
    else:
        return as_float(from_range_1(X_p / I_max))


def log_decoding_ERIMMRGB(X_p,
                          bit_depth=8,
                          in_int=False,
                          E_min=0.001,
                          E_clip=316.2):
    """
    Defines the *ERIMM RGB* log decoding curve / electro-optical transfer
    function (EOTF / EOCF).

    Parameters
    ----------
    X_p : numeric or array_like
        Non-linear data :math:`X'_{ERIMM}`.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_int : bool, optional
        Whether to treat the input value as integer code value or float
        equivalent of a code value at a given bit depth.
    E_min : numeric, optional
        Minimum exposure limit.
    E_clip : numeric, optional
        Maximum exposure limit.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`X_{ERIMM}`.

    Notes
    -----

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``X_p``        | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``X``          | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    -   \\* This definition has an input integer switch, thus the domain-range
        scale information is only given for the floating point mode.

    References
    ----------
    :cite:`Spaulding2000b`

    Examples
    --------
    >>> log_decoding_ERIMMRGB(0.410052389492129) # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding_ERIMMRGB(105, in_int=True) # doctest: +ELLIPSIS
    0.1...
    """

    X_p = to_domain_1(X_p)

    I_max = 2 ** bit_depth - 1

    if not in_int:
        X_p = X_p * I_max

    E_t = np.exp(1) * E_min

    X = np.where(
        X_p <= I_max * (
            (np.log(E_t) - np.log(E_min)) / (np.log(E_clip) - np.log(E_min))),
        ((np.log(E_clip) - np.log(E_min)) / (np.log(E_t) - np.log(E_min))) * (
            (X_p * E_t) / I_max),
        np.exp((X_p / I_max) * (np.log(E_clip) - np.log(E_min)) +
               np.log(E_min)),
    )

    return as_float(from_range_1(X))
