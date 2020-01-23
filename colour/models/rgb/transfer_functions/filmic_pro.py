# -*- coding: utf-8 -*-
"""
FiLMiC Pro 6 Encoding
=============--======

Defines the *FiLMiC Pro 6* encoding:

-   :func:`colour.models.log_encoding_FilmicPro6`
-   :func:`colour.models.log_decoding_FilmicPro6`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`FiLMiCInc2017` : FiLMiC Inc. (2017). FiLMiC Pro - User Manual v6 -
    Revision 1. Retrieved from http://www.filmicpro.com/\
FilmicProUserManualv6.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import Extrapolator, LinearInterpolator
from colour.utilities import from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['log_encoding_FilmicPro6', 'log_decoding_FilmicPro6']


def log_encoding_FilmicPro6(t):
    """
    Defines the *FiLMiC Pro 6* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    t : numeric or array_like
        Linear data :math:`t`.

    Returns
    -------
    numeric or ndarray
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

    return from_range_1(y)


_LOG_DECODING_FILMICPRO_INTERPOLATOR_CACHE = None


def _log_decoding_FilmicPro6_interpolator():
    """
    Returns the *FiLMiC Pro 6* log decoding curve / electro-optical transfer
    function interpolator and caches it if not existing.

    Returns
    -------
    Extrapolator
        *FiLMiC Pro 6* log decoding curve / electro-optical transfer
        function interpolator.
    """

    global _LOG_DECODING_FILMICPRO_INTERPOLATOR_CACHE

    t = np.arange(0, 1, 0.0001)
    if _LOG_DECODING_FILMICPRO_INTERPOLATOR_CACHE is None:
        _LOG_DECODING_FILMICPRO_INTERPOLATOR_CACHE = Extrapolator(
            LinearInterpolator(log_encoding_FilmicPro6(t), t))

    return _LOG_DECODING_FILMICPRO_INTERPOLATOR_CACHE


def log_decoding_FilmicPro6(y):
    """
    Defines the *FiLMiC Pro 6* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear data :math:`y`.

    Returns
    -------
    numeric or ndarray
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

    return from_range_1(t)
