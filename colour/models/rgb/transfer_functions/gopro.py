"""
GoPro Encoding
==============

Defines the *GoPro* *Protune* encoding:

-   :func:`colour.models.log_encoding_Protune`
-   :func:`colour.models.log_decoding_Protune`

References
----------
-   :cite:`GoPro2016a` : GoPro, Duiker, H.-P., & Mansencal, T. (2016).
    gopro.py. Retrieved April 12, 2017, from
    https://github.com/hpd/OpenColorIO-Configs/blob/master/aces_1.0.3/python/\
aces_ocio/colorspaces/gopro.py
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import as_float, from_range_1, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "log_encoding_Protune",
    "log_decoding_Protune",
]


def log_encoding_Protune(x: ArrayLike) -> NDArrayFloat:
    """
    Define the *Protune* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`GoPro2016a`

    Examples
    --------
    >>> log_encoding_Protune(0.18)  # doctest: +ELLIPSIS
    0.6456234...
    """

    x = to_domain_1(x)

    y = np.log1p(x * 112) / np.log(113)

    return as_float(from_range_1(y))


def log_decoding_Protune(y: ArrayLike) -> NDArrayFloat:
    """
    Define the *Protune* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y
        Non-linear data :math:`y`.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`GoPro2016a`

    Examples
    --------
    >>> log_decoding_Protune(0.645623486803636)  # doctest: +ELLIPSIS
    0.1...
    """

    y = to_domain_1(y)

    x = (113**y - 1) / 112

    return as_float(from_range_1(x))
