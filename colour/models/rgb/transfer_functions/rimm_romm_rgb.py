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

from colour.utilities import (as_numeric, domain_range_scale, from_range_1,
                              from_range_int, to_domain_1, to_domain_int)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'oetf_ROMMRGB', 'eotf_ROMMRGB', 'oetf_ProPhotoRGB', 'eotf_ProPhotoRGB',
    'oetf_RIMMRGB', 'eotf_RIMMRGB', 'log_encoding_ERIMMRGB',
    'log_decoding_ERIMMRGB'
]


def oetf_ROMMRGB(X, I_max=255):
    """
    Defines the *ROMM RGB* encoding opto-electronic transfer function
    (OETF / OECF).

    Parameters
    ----------
    X : numeric or array_like
        Linear data :math:`X_{ROMM}`.
    I_max : numeric, optional
        Maximum code value: 255, 4095 and 65535 for respectively 8-bit,
        12-bit and 16-bit per channel.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`X'_{ROMM}`.

    References
    ----------
    -   :cite:`ANSI2003a`
    -   :cite:`Spaulding2000b`

    Examples
    --------
    >>> oetf_ROMMRGB(0.18)  # doctest: +ELLIPSIS
    98.3564133...
    """

    X = to_domain_1(X)

    E_t = 16 ** (1.8 / (1 - 1.8))

    X_p = np.where(X < E_t, X * 16 * I_max, X ** (1 / 1.8) * I_max)

    return as_numeric(from_range_int(X_p, np.log2(I_max + 1)))


def eotf_ROMMRGB(X_p, I_max=255):
    """
    Defines the *ROMM RGB* encoding electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    X_p : numeric or array_like
        Non-linear data :math:`X'_{ROMM}`.
    I_max : numeric, optional
        Maximum code value: 255, 4095 and 65535 for respectively 8-bit,
        12-bit and 16-bit per channel.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`X_{ROMM}`.

    References
    ----------
    -   :cite:`ANSI2003a`
    -   :cite:`Spaulding2000b`

    Examples
    --------
    >>> eotf_ROMMRGB(98.356413311540095) # doctest: +ELLIPSIS
    0.1...
    """

    X_p = to_domain_int(X_p, np.log2(I_max + 1))

    E_t = 16 ** (1.8 / (1 - 1.8))

    X = np.where(X_p < 16 * E_t * I_max, X_p / (16 * I_max), (X_p / I_max)
                 ** 1.8)

    return as_numeric(from_range_1(X))


oetf_ProPhotoRGB = oetf_ROMMRGB
eotf_ProPhotoRGB = eotf_ROMMRGB


def oetf_RIMMRGB(X, I_max=255, E_clip=2.0):
    """
    Defines the *RIMM RGB* encoding opto-electronic transfer function
    (OETF / OECF).

    *RIMM RGB* encoding non-linearity is based on that specified by
    *Recommendation ITU-R BT.709-6*.

    Parameters
    ----------
    X : numeric or array_like
        Linear data :math:`X_{RIMM}`.
    I_max : numeric, optional
        Maximum code value: 255, 4095 and 65535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    E_clip : numeric, optional
        Maximum exposure level.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`X'_{RIMM}`.

    References
    ----------
    -   :cite:`Spaulding2000b`

    Examples
    --------
    >>> oetf_RIMMRGB(0.18)  # doctest: +ELLIPSIS
    74.3768017...
    """

    X = to_domain_1(X)

    V_clip = 1.099 * E_clip ** 0.45 - 0.099
    q = I_max / V_clip

    X_p = np.select([X < 0.0, X < 0.018, X >= 0.018, X > E_clip],
                    [0, 4.5 * X, 1.099 * (X ** 0.45) - 0.099, I_max])

    return as_numeric(from_range_int(q * X_p, np.log2(I_max + 1)))


def eotf_RIMMRGB(X_p, I_max=255, E_clip=2.0):
    """
    Defines the *RIMM RGB* encoding electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    X_p : numeric or array_like
        Non-linear data :math:`X'_{RIMM}`.
    I_max : numeric, optional
        Maximum code value: 255, 4095 and 65535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    E_clip : numeric, optional
        Maximum exposure level.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`X_{RIMM}`.

    References
    ----------
    -   :cite:`Spaulding2000b`

    Examples
    --------
    >>> eotf_RIMMRGB(74.37680178131521)  # doctest: +ELLIPSIS
    0.1...
    """

    X_p = to_domain_int(X_p, np.log2(I_max + 1))

    V_clip = 1.099 * E_clip ** 0.45 - 0.099

    m = V_clip * X_p / I_max

    with domain_range_scale('ignore'):
        X = np.where(X_p < oetf_RIMMRGB(0.018), m / 4.5, ((m + 0.099) / 1.099)
                     ** (1 / 0.45))

    return as_numeric(from_range_1(X))


def log_encoding_ERIMMRGB(X, I_max=255, E_min=0.001, E_clip=316.2):
    """
    Defines the *ERIMM RGB* log encoding curve / opto-electronic transfer
    function (OETF / OECF).

    Parameters
    ----------
    X : numeric or array_like
        Linear data :math:`X_{ERIMM}`.
    I_max : numeric, optional
        Maximum code value: 255, 4095 and 65535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    E_min : numeric, optional
        Minimum exposure limit.
    E_clip : numeric, optional
        Maximum exposure limit.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`X'_{ERIMM}`.

    References
    ----------
    -   :cite:`Spaulding2000b`

    Examples
    --------
    >>> log_encoding_ERIMMRGB(0.18)  # doctest: +ELLIPSIS
    104.5633593...
    """

    X = to_domain_1(X)

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
        I_max * ((np.log(X) - np.log(E_min)) /
                 (np.log(E_clip) - np.log(E_min))),
        I_max,
    ])

    return as_numeric(from_range_int(X_p, np.log2(I_max + 1)))


def log_decoding_ERIMMRGB(X_p, I_max=255, E_min=0.001, E_clip=316.2):
    """
    Defines the *ERIMM RGB* log decoding curve / electro-optical transfer
    function (EOTF / EOCF).

    Parameters
    ----------
    X_p : numeric or array_like
        Non-linear data :math:`X'_{ERIMM}`.
    I_max : numeric, optional
        Maximum code value: 255, 4095 and 65535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    E_min : numeric, optional
        Minimum exposure limit.
    E_clip : numeric, optional
        Maximum exposure limit.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`X_{ERIMM}`.

    References
    ----------
    -   :cite:`Spaulding2000b`

    Examples
    --------
    >>> log_decoding_ERIMMRGB(104.56335932049294) # doctest: +ELLIPSIS
    0.1...
    """

    X_p = to_domain_int(X_p, np.log2(I_max + 1))

    E_t = np.exp(1) * E_min

    X = np.where(
        X_p <= I_max * ((np.log(E_t) - np.log(E_min)) /
                        (np.log(E_clip) - np.log(E_min))),
        ((np.log(E_clip) - np.log(E_min)) / (np.log(E_t) - np.log(E_min))) *
        ((X_p * E_t) / I_max),
        np.exp((X_p / I_max) *
               (np.log(E_clip) - np.log(E_min)) + np.log(E_min)),
    )

    return as_numeric(from_range_1(X))
