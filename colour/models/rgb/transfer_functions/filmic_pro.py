"""
FiLMiC Pro 6 Encoding
=====================

Defines the *FiLMiC Pro 6* encoding:

-   :func:`colour.models.log_encoding_FilmicPro6`
-   :func:`colour.models.log_decoding_FilmicPro6`

References
----------
-   :cite:`FiLMiCInc2017` : FiLMiC Inc. (2017). FiLMiC Pro - User Manual v6 -
    Revision 1 (pp. 1-46). http://www.filmicpro.com/FilmicProUserManualv6.pdf
"""

from __future__ import annotations

import numpy as np

from colour.algebra import Extrapolator, LinearInterpolator
from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import as_float, from_range_1, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "log_encoding_FilmicPro6",
    "log_decoding_FilmicPro6",
]


def log_encoding_FilmicPro6(t: ArrayLike) -> NDArrayFloat:
    """
    Define the *FiLMiC Pro 6* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    t
        Linear data :math:`t`.

    Returns
    -------
    :class:`numpy.ndarray`
        Non-linear data :math:`y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``t``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    -   The *FiLMiC Pro 6* log encoding curve / opto-electronic transfer
        function is only defined for domain (0, 1].

    References
    ----------
    :cite:`FiLMiCInc2017`

    Warnings
    --------
    The *FiLMiC Pro 6* log encoding curve / opto-electronic transfer function
    was fitted with poor precision and has :math:`Y=1.000000819999999` value
    for :math:`t=1`. It also has no linear segment near zero and will thus be
    undefined for :math:`t=0` when computing its logarithm.

    Examples
    --------
    >>> log_encoding_FilmicPro6(0.18)  # doctest: +ELLIPSIS
    0.6066345...
    """

    t = to_domain_1(t)

    y = 0.371 * (np.sqrt(t) + 0.28257 * np.log(t) + 1.69542)

    return as_float(from_range_1(y))


_CACHE_LOG_DECODING_FILMICPRO_INTERPOLATOR: Extrapolator | None = None


def _log_decoding_FilmicPro6_interpolator() -> Extrapolator:
    """
    Return the *FiLMiC Pro 6* log decoding curve / electro-optical transfer
    function interpolator and caches it if not existing.

    Returns
    -------
    :class:`colour.Extrapolator`
        *FiLMiC Pro 6* log decoding curve / electro-optical transfer
        function interpolator.
    """

    global _CACHE_LOG_DECODING_FILMICPRO_INTERPOLATOR

    t = np.arange(0, 1, 0.0001)
    if _CACHE_LOG_DECODING_FILMICPRO_INTERPOLATOR is None:
        _CACHE_LOG_DECODING_FILMICPRO_INTERPOLATOR = Extrapolator(
            LinearInterpolator(log_encoding_FilmicPro6(t), t)
        )

    return _CACHE_LOG_DECODING_FILMICPRO_INTERPOLATOR


def log_decoding_FilmicPro6(y: ArrayLike) -> NDArrayFloat:
    """
    Define the *FiLMiC Pro 6* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y
        Non-linear data :math:`y`.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`t`.

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
    | ``t``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    -   The *FiLMiC Pro 6* log decoding curve / electro-optical transfer
        function is only defined for domain (0, 1].

    References
    ----------
    :cite:`FiLMiCInc2017`

    Warnings
    --------
    The *FiLMiC Pro 6* log encoding curve / opto-electronic transfer function
    has no inverse in :math:`R`, we thus use a *LUT* based inversion.

    Examples
    --------
    >>> log_decoding_FilmicPro6(0.6066345199247033)  # doctest: +ELLIPSIS
    0.1800000...
    """

    y = to_domain_1(y)

    t = _log_decoding_FilmicPro6_interpolator()(y)

    return as_float(from_range_1(t))
