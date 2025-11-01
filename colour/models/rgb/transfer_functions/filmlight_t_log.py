"""
FilmLight T-Log Log Encoding
============================

Define the *FilmLight T-Log* log encoding.

-   :func:`colour.models.log_encoding_FilmLightTLog`
-   :func:`colour.models.log_decoding_FilmLightTLog`

References
----------
-   :cite:`Siragusano2018a` : Siragusano, D. (2018). Private Discussion with
    Shaw, Nick.
"""

from __future__ import annotations

import numpy as np

from colour.hints import (  # noqa: TC001
    Domain1,
    Range1,
)
from colour.utilities import as_float, from_range_1, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "log_encoding_FilmLightTLog",
    "log_decoding_FilmLightTLog",
]


def log_encoding_FilmLightTLog(
    x: Domain1,
    w: float = 128.0,
    g: float = 16.0,
    o: float = 0.075,
) -> Range1:
    """
    Apply the *FilmLight T-Log* log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    x
        Linear reflection data :math:`x`.
    w
        Value of :math:`x` for which :math:`t = 1.0`.
    g
        Gradient at :math:`x = 0.0`.
    o
        Value of :math:`t` at :math:`x = 0.0`.

    Returns
    -------
    :class:`numpy.ndarray`
        *FilmLight T-Log* encoded data :math:`t`.

    References
    ----------
    :cite:`Siragusano2018a`

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``t``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    -   The following is an excerpt from the FilmLight colour space file
        `./etc/colourspaces/FilmLight_TLog_EGamut.flspace` which can be
        obtained by installing the free Baselight for Nuke plugin::

            T-Log, a cineon driven log tone-curve developed by FilmLight.
            The colour space is designed to be used as *Working Colour Space*.

            Version 10.0
            This is similar to Cineon LogC function.

            The formula is...
            y = A + B*log(x + C)
            ...where x,y are the log and linear values.

            A,B,C are constants calculated from...
            w = x value for y = 1.0
            g = the gradient at x=0
            o = y value for x = 0.0

            We do not have an exact solution but the
            formula for b gives an approximation. The
            gradient is not g, but should be within a
            few percent for most sensible values of (w*g).

    Examples
    --------
    >>> log_encoding_FilmLightTLog(0.18)  # doctest: +ELLIPSIS
    0.3965678...
    """

    x = to_domain_1(x)

    b = 1 / (0.7107 + 1.2359 * np.log(w * g))
    gs = g / (1 - o)
    C = b / gs
    a = 1 - b * np.log(w + C)
    y0 = a + b * np.log(C)
    s = (1 - o) / (1 - y0)
    A = 1 + (a - 1) * s
    B = b * s
    G = gs * s

    t = np.where(
        x < 0.0,
        G * x + o,
        np.log(x + C) * B + A,
    )

    return as_float(from_range_1(t))


def log_decoding_FilmLightTLog(
    t: Domain1,
    w: float = 128.0,
    g: float = 16.0,
    o: float = 0.075,
) -> Range1:
    """
    Apply the *FilmLight T-Log* log decoding inverse opto-electronic transfer

    function (OETF).

    Parameters
    ----------
    t
        *FilmLight T-Log* encoded data :math:`t`.
    w
        Value of :math:`x` for which :math:`t = 1.0`.
    g
        Gradient at :math:`x = 0.0`.
    o
        Value of :math:`t` at :math:`x = 0.0`.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear reflection data :math:`x`.

    References
    ----------
    :cite:`Siragusano2018a`

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``t``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    -   The following is an excerpt from the FilmLight colour space file
        `./etc/colourspaces/FilmLight_TLog_EGamut.flspace` which can be
        obtained by installing the free Baselight for Nuke plugin::

            T-Log, a cineon driven log tone-curve developed by FilmLight.
            The colour space is designed to be used as *Working Colour Space*.

            Version 10.0
            This is similar to Cineon LogC function.

            The formula is...
            y = A + B*log(x + C)
            ...where x,y are the log and linear values.

            A,B,C are constants calculated from...
            w = x value for y = 1.0
            g = the gradient at x=0
            o = y value for x = 0.0

            We do not have an exact solution but the
            formula for b gives an approximation. The
            gradient is not g, but should be within a
            few percent for most sensible values of (w*g).

    Examples
    --------
    >>> log_decoding_FilmLightTLog(0.396567801298332)  # doctest: +ELLIPSIS
    0.1800000...
    """

    t = to_domain_1(t)

    b = 1 / (0.7107 + 1.2359 * np.log(w * g))
    gs = g / (1 - o)
    C = b / gs
    a = 1 - b * np.log(w + C)
    y0 = a + b * np.log(C)
    s = (1 - o) / (1 - y0)
    A = 1 + (a - 1) * s
    B = b * s
    G = gs * s

    x = np.where(
        t < o,
        (t - o) / G,
        np.exp((t - A) / B) - C,
    )

    return as_float(from_range_1(x))
