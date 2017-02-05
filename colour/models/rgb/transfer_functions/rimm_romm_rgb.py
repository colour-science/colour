#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RIMM, ROMM and ERIMM Encodings
==============================

Defines the *RIMM, ROMM and ERIMM* encodings opto-electrical transfer functions
(OETF / OECF) and electro-optical transfer functions (EOTF / EOCF):

-   :func:`oetf_ROMMRGB`
-   :func:`eotf_ROMMRGB`
-   :func:`oetf_ProPhotoRGB`
-   :func:`eotf_ProPhotoRGB`
-   :func:`oetf_RIMMRGB`
-   :func:`eotf_RIMMRGB`
-   :func:`log_encoding_ERIMMRGB`
-   :func:`log_decoding_ERIMMRGB`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Spaulding, K. E., Woolfe, G. J., & Giorgianni, E. J. (2000). Reference
        Input/Output Medium Metric RGB Color Encodings (RIMM/ROMM RGB), 1–8.
        Retrieved from http://www.photo-lovers.org/pdf/color/romm.pdf
.. [3]  ANSI. (2003). Specification of ROMM RGB. Retrieved from
        http://www.color.org/ROMMRGB.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['oetf_ROMMRGB',
           'eotf_ROMMRGB',
           'oetf_ProPhotoRGB',
           'eotf_ProPhotoRGB',
           'oetf_RIMMRGB',
           'eotf_RIMMRGB',
           'log_encoding_ERIMMRGB',
           'log_decoding_ERIMMRGB']


def oetf_ROMMRGB(X, I_max=255):
    """
    Defines the *ROMM RGB* encoding opto-electronic transfer function
    (OETF / OECF).

    Parameters
    ----------
    X : numeric or array_like
        Linear data :math:`X_{ROMM}`.
    I_max : numeric, optional
        Maximum code value: 255, 4095 and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`X'_{ROMM}`.

    Examples
    --------
    >>> oetf_ROMMRGB(0.18)  # doctest: +ELLIPSIS
    98.3564133...
    """

    X = np.asarray(X)

    E_t = 16 ** (1.8 / (1 - 1.8))

    return as_numeric(np.clip(
        np.where(X < E_t,
                 X * 16 * I_max,
                 X ** (1 / 1.8) * I_max), 0, I_max))


def eotf_ROMMRGB(X_p, I_max=255):
    """
    Defines the *ROMM RGB* encoding electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    X_p : numeric or array_like
        Non-linear data :math:`X'_{ROMM}`.
    I_max : numeric, optional
        Maximum code value: 255, 4095 and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`X_{ROMM}`.

    Examples
    --------
    >>> eotf_ROMMRGB(98.356413311540095) # doctest: +ELLIPSIS
    0.1...
    """

    X_p = np.asarray(X_p)

    E_t = 16 ** (1.8 / (1 - 1.8))

    return as_numeric(np.where(
        X_p < 16 * E_t * I_max,
        X_p / (16 * I_max),
        (X_p / I_max) ** 1.8))


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
        Maximum code value: 255, 4095 and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    E_clip : numeric, optional
        Maximum exposure level.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`X'_{RIMM}`.

    Examples
    --------
    >>> oetf_RIMMRGB(0.18)  # doctest: +ELLIPSIS
    74.3768017...
    """

    X = np.asarray(X)

    V_clip = 1.099 * E_clip ** 0.45 - 0.099
    q = I_max / V_clip

    X_p_RIMM = np.select(
        [X < 0.0,
         X < 0.018, X >= 0.018,
         X > E_clip],
        [0, 4.5 * X, 1.099 * (X ** 0.45) - 0.099, I_max])

    return as_numeric(q * X_p_RIMM)


def eotf_RIMMRGB(X_p, I_max=255, E_clip=2.0):
    """
    Defines the *RIMM RGB* encoding electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    X_p : numeric or array_like
        Non-linear data :math:`X'_{RIMM}`.
    I_max : numeric, optional
        Maximum code value: 255, 4095 and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    E_clip : numeric, optional
        Maximum exposure level.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`X_{RIMM}`.

    Examples
    --------
    >>> eotf_RIMMRGB(74.37680178131521)  # doctest: +ELLIPSIS
    0.1...
    """

    X_p = np.asarray(X_p)

    V_clip = 1.099 * E_clip ** 0.45 - 0.099

    m = V_clip * X_p / I_max

    X_RIMM = np.where(
        X_p < oetf_RIMMRGB(0.018),
        m / 4.5, ((m + 0.099) / 1.099) ** (1 / 0.45))

    return as_numeric(X_RIMM)


def log_encoding_ERIMMRGB(X,
                          I_max=255,
                          E_min=0.001,
                          E_clip=316.2):
    """
    Defines the *ERIMM RGB* log encoding curve / opto-electronic transfer
    function (OETF / OECF).

    Parameters
    ----------
    X : numeric or array_like
        Linear data :math:`X_{ERIMM}`.
    I_max : numeric, optional
        Maximum code value: 255, 4095 and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    E_min : numeric, optional
        Minimum exposure limit.
    E_clip : numeric, optional
        Maximum exposure limit.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`X'_{ERIMM}`.

    Examples
    --------
    >>> log_encoding_ERIMMRGB(0.18)  # doctest: +ELLIPSIS
    104.5633593...
    """

    X = np.asarray(X)

    E_t = np.exp(1) * E_min

    X_p = np.select(
        [X < 0.0,
         X <= E_t, X > E_t,
         X > E_clip],
        [0,
         I_max * ((np.log(E_t) - np.log(E_min)) /
                  (np.log(E_clip) - np.log(E_min))) * (X / E_t),
         I_max * ((np.log(X) - np.log(E_min)) /
                  (np.log(E_clip) - np.log(E_min))),
         I_max])

    return as_numeric(X_p)


def log_decoding_ERIMMRGB(X_p,
                          I_max=255,
                          E_min=0.001,
                          E_clip=316.2):
    """
    Defines the *ERIMM RGB* log decoding curve / electro-optical transfer
    function (EOTF / EOCF).

    Parameters
    ----------
    X_p : numeric or array_like
        Non-linear data :math:`X'_{ERIMM}`.
    I_max : numeric, optional
        Maximum code value: 255, 4095 and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    E_min : numeric, optional
        Minimum exposure limit.
    E_clip : numeric, optional
        Maximum exposure limit.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`X_{ERIMM}`.

    Examples
    --------
    >>> log_decoding_ERIMMRGB(104.56335932049294) # doctest: +ELLIPSIS
    0.1...
    """

    X_p = np.asarray(X_p)

    E_t = np.exp(1) * E_min

    X = np.where(
        X_p <= I_max * ((np.log(E_t) - np.log(E_min)) /
                        (np.log(E_clip) - np.log(E_min))),
        (((np.log(E_clip) - np.log(E_min)) / (np.log(E_t) - np.log(E_min))) *
         ((X_p * E_t) / I_max)),
        np.exp((X_p / I_max) *
               (np.log(E_clip) - np.log(E_min)) + np.log(E_min)))

    return as_numeric(X)
