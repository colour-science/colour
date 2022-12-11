"""
Prismatic Colourspace
=====================

Defines the *Prismatic* colourspace transformations:

-   :func:`colour.RGB_to_Prismatic`
-   :func:`colour.Prismatic_to_RGB`

References
----------
-   :cite:`Shirley2015a` : Shirley, P., & Hart, D. (2015). The prismatic color
    space for rgb computations (pp. 2-7).
"""

from __future__ import annotations

import numpy as np

from colour.algebra import sdiv, sdiv_mode
from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import from_range_1, to_domain_1, tsplit, tstack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "RGB_to_Prismatic",
    "Prismatic_to_RGB",
]


def RGB_to_Prismatic(RGB: ArrayLike) -> NDArrayFloat:
    """
    Convert from *RGB* colourspace to *Prismatic* :math:`L\\rho\\gamma\\beta`
    colourspace array.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *Prismatic* :math:`L\\rho\\gamma\\beta` colourspace array.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Lrgb``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Shirley2015a`

    Examples
    --------
    >>> RGB = np.array([0.25, 0.50, 0.75])
    >>> RGB_to_Prismatic(RGB)  # doctest: +ELLIPSIS
    array([ 0.75...   ,  0.1666666...,  0.3333333...,  0.5...   ])

    Adjusting saturation of given *RGB* colourspace array:
    >>> saturation = 0.5
    >>> Lrgb = RGB_to_Prismatic(RGB)
    >>> Lrgb[..., 1:] = 1 / 3 + saturation * (Lrgb[..., 1:] - 1 / 3)
    >>> Prismatic_to_RGB(Lrgb)  # doctest: +ELLIPSIS
    array([ 0.45...,  0.6...,  0.75...])
    """

    RGB = to_domain_1(RGB)

    L = np.max(RGB, axis=-1)
    s = np.sum(RGB, axis=-1)

    with sdiv_mode():
        one_s = sdiv(1, s[..., None])

    r, g, b = tsplit(one_s * RGB)

    Lrgb = tstack([L, r, g, b])

    return from_range_1(Lrgb)


def Prismatic_to_RGB(Lrgb: ArrayLike) -> NDArrayFloat:
    """
    Convert from *Prismatic* :math:`L\\rho\\gamma\\beta` colourspace array to
    *RGB* colourspace.

    Parameters
    ----------
    Lrgb
        *Prismatic* :math:`L\\rho\\gamma\\beta` colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colourspace array.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Lrgb``   | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Shirley2015a`

    Examples
    --------
    >>> Lrgb = np.array([0.75000000, 0.16666667, 0.33333333, 0.50000000])
    >>> Prismatic_to_RGB(Lrgb)  # doctest: +ELLIPSIS
    array([ 0.25...   ,  0.4999999...,  0.75...  ])
    """

    Lrgb = to_domain_1(Lrgb)

    rgb = Lrgb[..., 1:]
    m = np.max(rgb, axis=-1)

    with sdiv_mode():
        RGB = sdiv(Lrgb[..., 0][..., None], m[..., None])

    RGB = RGB * rgb

    return from_range_1(RGB)
