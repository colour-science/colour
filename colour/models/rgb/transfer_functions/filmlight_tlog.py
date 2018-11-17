# -*- coding: utf-8 -*-
"""
FilmLight T-Log Log Encoding
============================

Defines the *FilmLight T-Log* log encoding:

-   :func:`colour.models.log_encoding_FilmLight_T_Log`
-   :func:`colour.models.log_decoding_FilmLight_T_Log`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Siragusano2018a` : Siragusano, D (2018). Private Discussion with
    Shaw, N.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_float, from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['log_encoding_FilmLight_T_Log', 'log_decoding_FilmLight_T_Log']


def log_encoding_FilmLight_T_Log(x, w=128.0, g=16.0, o=0.075):
    """
    Defines the *FilmLight T-Log* log encoding curve.

    Parameters
    ----------
    x : numeric or array_like
        Linear reflection data :math`x`.
    w : numeric, optional
        Value of :math:`x` for :math:`t = 1.0`.
    g : numeric, optional
        Gradient at :math:`x = 0.0`.
    o : numeric, optional
        Value of :math:`t` for :math:`x = 0.0`.

    Returns
    -------
    numeric or ndarray
        *FilmLight T-Log* encoded data :math:`t`.

    References
    ----------
    :cite:`Siragusano2018a`

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
    | ``t``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_encoding_FilmLight_T_Log(0.18)  # doctest: +ELLIPSIS
    0.3965678...
    """

    b = 1.0 / (0.7107 + 1.2359 * np.log(w * g))
    gs = g / (1.0 - o)
    C = b / gs
    a = 1.0 - b * np.log(w + C)
    y0 = a + b * np.log(C)
    s = (1.0 - o) / (1.0 - y0)
    A = 1.0 + (a - 1.0) * s
    B = b * s
    G = gs * s

    x = to_domain_1(x)

    t = np.where(
        x < 0.0,
        G * x + o,
        np.log(x + C)*B + A
    )

    return as_float(from_range_1(t))


def log_decoding_FilmLight_T_Log(t, w=128.0, g=16.0, o=0.075):
    """
    Defines the *FilmLight T-Log* log decoding curve.

    Parameters
    ----------
    t : numeric or array_like
        Non-linear data :math:`t`.
    w : numeric, optional
        Value of :math:`x` for :math:`t = 1.0`.
    g : numeric, optional
        Gradient at :math:`x = 0.0`.
    o : numeric, optional
        Value of :math:`t` for :math:`x = 0.0`.

    Returns
    -------
    numeric or ndarray
        Linear reflection data :math`x`.

    References
    ----------
    :cite:`Siragusano2018a`

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``t``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_decoding_FilmLight_T_Log(0.39656780129833191)
    0.1800000...
    """

    b = 1.0 / (0.7107 + 1.2359 * np.log(w * g))
    gs = g / (1.0 - o)
    C = b / gs
    a = 1.0 - b * np.log(w + C)
    y0 = a + b * np.log(C)
    s = (1.0 - o) / (1.0 - y0)
    A = 1.0 + (a - 1.0) * s
    B = b * s
    G = gs * s

    t = to_domain_1(t)

    x = np.where(
        t < o,
        (t - o) / G,
        np.exp((t - A) / B) - C
    )

    return as_float(from_range_1(x))
