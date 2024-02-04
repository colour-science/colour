"""
CSS Color Module Level 3 - Web Colours
======================================

Defines the conversion of colour keywords to *RGB* colourspace:

-   :attr:`colour.notation.keyword_to_RGB_CSSColor3`

References
----------
-   :cite:`W3C2022` : W3C. (2022). CSS Color Module Level 3.
    https://www.w3.org/TR/css-color-3/
"""

from __future__ import annotations

from colour.hints import NDArrayFloat
from colour.notation import CSS_COLOR_3, HEX_to_RGB
from colour.utilities import attest

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "keyword_to_RGB_CSSColor3",
]


def keyword_to_RGB_CSSColor3(keyword: str) -> NDArrayFloat:
    """
    Convert given colour keyword to *RGB* colourspace according to
    *CSS Color Module Level 3* *W3C Recommendation*.

    Parameters
    ----------
    keyword
        Colour keyword.

    Returns
    -------
    :class:`numpy.array`
        *RGB* colourspace array.

    Notes
    -----
    -   All the RGB colors are specified in the *IEC 61966-2-1:1999* *sRGB*
        colourspace.

    Examples
    --------
    >>> keyword_to_RGB_CSSColor3("black")
    array([ 0.,  0.,  0.])
    >>> keyword_to_RGB_CSSColor3("white")
    array([ 1.,  1.,  1.])
    >>> keyword_to_RGB_CSSColor3("aliceblue")  # doctest: +ELLIPSIS
    array([ 0.9411764...,  0.9725490...,  1.        ])
    """

    attest(keyword in CSS_COLOR_3, f'{keyword} is not defined in "CSS Color 3"!')

    return HEX_to_RGB(CSS_COLOR_3[keyword])
