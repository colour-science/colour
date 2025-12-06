"""
Hexadecimal Notation
====================

Define objects for converting between RGB colour values and hexadecimal
notation.

-   :func:`colour.notation.RGB_to_HEX`
-   :func:`colour.notation.HEX_to_RGB`
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import normalise_maximum

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayStr

from colour.hints import (  # noqa: TC001
    ArrayLike,
    Domain1,
    Range1,
)
from colour.models import eotf_inverse_sRGB, eotf_sRGB
from colour.utilities import (
    as_float_array,
    as_int_array,
    from_range_1,
    to_domain_1,
    usage_warning,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "RGB_to_HEX",
    "HEX_to_RGB",
]


def RGB_to_HEX(RGB: Domain1) -> NDArrayStr:
    """
    Convert from *RGB* colourspace to hexadecimal representation.

    Parameters
    ----------
    RGB
        *RGB* colourspace array with values typically normalised to [0, 1].

    Returns
    -------
    :class:`str` or :class:`numpy.array`
        Hexadecimal representation as a string in the format '#RRGGBB'.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> RGB = np.array([0.66666667, 0.86666667, 1.00000000])
    >>> RGB_to_HEX(RGB)
    '#aaddff'
    """

    RGB = to_domain_1(RGB)

    if np.any(RGB < 0):
        usage_warning(
            '"RGB" array contains negative values, those will be clipped, '
            "unpredictable results may occur!"
        )

        RGB = as_float_array(np.clip(RGB, 0, np.inf))

    if np.any(RGB > 1):
        usage_warning(
            '"RGB" array contains values over 1 and will be normalised, '
            "unpredictable results may occur!"
        )

        RGB = eotf_inverse_sRGB(normalise_maximum(eotf_sRGB(RGB)))

    to_HEX = np.vectorize("{:02x}".format)

    HEX = to_HEX(as_int_array(RGB * 255, dtype=np.uint8)).astype(object)

    return np.asarray("#") + HEX[..., 0] + HEX[..., 1] + HEX[..., 2]


def HEX_to_RGB(HEX: ArrayLike) -> Range1:
    """
    Convert from hexadecimal representation to *RGB* colourspace.

    Parameters
    ----------
    HEX
        Hexadecimal representation as a string in the format '#RRGGBB'.

    Returns
    -------
    :class:`numpy.array`
        *RGB* colourspace array with values typically normalised to [0, 1].

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``RGB``   | 1                     | 1             |
    +-----------+-----------------------+---------------+

    Examples
    --------
    >>> HEX = "#aaddff"
    >>> HEX_to_RGB(HEX)  # doctest: +ELLIPSIS
    array([ 0.6666666...,  0.8666666...,  1.        ])
    """

    HEX = np.char.lstrip(HEX, "#")  # pyright: ignore

    def to_RGB(x: list) -> list:
        """Convert specified hexadecimal representation to *RGB*."""

        l_x = len(x)

        return [
            int(x[i : i + l_x // 3], 16)  # pyright: ignore
            for i in range(0, l_x, l_x // 3)
        ]

    to_RGB_v = np.vectorize(to_RGB, otypes=[np.ndarray])

    RGB = as_float_array(to_RGB_v(HEX).tolist()) / 255

    return from_range_1(RGB)
