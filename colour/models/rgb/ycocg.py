# -*- coding: utf-8 -*-
"""
YCoCg Colour Encoding
======================

Defines the *YCoCg* colour encoding related transformations:

-   :func:`colour.RGB_to_YCoCg`
-   :func:`colour.YCoCg_to_RGB`

See Also
--------
`YCoCg Colours Encoding Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/ycocg.ipynb>`_

References
----------
-   :cite:`Malvar2003` : Malvar, H., & Sullivan, G. (2003). YCoCg-R: A Color
    Space with RGB Reversibility and Low Dynamic Range. Retrieved from
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/\
Malvar_Sullivan_YCoCg-R_JVT-I014r3-2.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import dot_vector

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Development'

__all__ = [
    'RGB_TO_YCOCG_MATRIX',
    'YCOCG_TO_RGB_MATRIX',
    'RGB_to_YCoCg',
    'YCoCg_to_RGB',
]

RGB_TO_YCOCG_MATRIX = np.array([
    [1 / 4, 1 / 2, 1 / 4],
    [1 / 2, 0, -1 / 2],
    [-1 / 4, 1 / 2, -1 / 4],
])
"""
*R'G'B'* colourspace to *YCoCg* colour encoding matrix.

RGB_TO_YCOCG_MATRIX : array_like, (3, 3)
"""

YCOCG_TO_RGB_MATRIX = np.array([
    [1, 1, -1],
    [1, 0, 1],
    [1, -1, -1],
])
"""
*YCoCg* colour encoding to *R'G'B'* colourspace matrix.

YCOCG_TO_RGB_MATRIX : array_like, (3, 3)
"""


def RGB_to_YCoCg(RGB):
    """
    Converts an array of *R'G'B'* values to the corresponding *YCoCg* colour
    encoding values array.

    Parameters
    ----------
    RGB : array_like
        Input *R'G'B'* array.

    Returns
    -------
    ndarray
        *YCoCg* colour encoding array.

    References
    ----------
    :cite:`Malvar2003`

    Examples
    --------
    >>> RGB_to_YCoCg(np.array([1.0, 1.0, 1.0]))
    array([ 1.,  0.,  0.])
    >>> RGB_to_YCoCg(np.array([0.75, 0.5, 0.5]))
    array([ 0.5625,  0.125 , -0.0625])
    """

    return dot_vector(RGB_TO_YCOCG_MATRIX, RGB)


def YCoCg_to_RGB(YCoCg):
    """
    Converts an array of *YCoCg* colour encoding values to the corresponding
    *R'G'B'* values array.

    Parameters
    ----------
    YCoCg : array_like
        *YCoCg* colour encoding array.

    Returns
    -------
    ndarray
        Output *R'G'B'* array.

    References
    ----------
    :cite:`Malvar2003`

    Examples
    --------
    >>> YCoCg_to_RGB(np.array([1.0, 0.0, 0.0]))
    array([ 1.,  1.,  1.])
    >>> YCoCg_to_RGB(np.array([0.5625, 0.125, -0.0625]))
    array([ 0.75,  0.5 ,  0.5 ])
    """

    return dot_vector(YCOCG_TO_RGB_MATRIX, YCoCg)
