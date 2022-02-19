"""
Sony .spimtx LUT Format Input / Output Utilities
================================================

Defines *Sony* *.spimtx* *LUT* format related input / output utilities objects.

-   :func:`colour.io.read_LUT_SonySPImtx`
-   :func:`colour.io.write_LUT_SonySPImtx`
"""

from __future__ import annotations

import numpy as np

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.hints import Integer
from colour.io.luts.common import path_to_title
from colour.io.luts import LUTOperatorMatrix
from colour.hints import Boolean

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "read_LUT_SonySPImtx",
    "write_LUT_SonySPImtx",
]


def read_LUT_SonySPImtx(path: str) -> LUTOperatorMatrix:
    """
    Read given *Sony* *.spimtx* *LUT* file.

    Parameters
    ----------
    path
        *LUT* path.

    Returns
    -------
    :class:`colour.LUTOperatorMatrix`
        :class:`colour.io.Matrix` class instance.

    Examples
    --------
    >>> import os
    >>> path = os.path.join(
    ...     os.path.dirname(__file__), 'tests', 'resources', 'sony_spimtx',
    ...     'dt.spimtx')
    >>> print(read_LUT_SonySPImtx(path))
    LUTOperatorMatrix - dt
    ----------------------
    <BLANKLINE>
    Matrix     : [[ 0.864274  0.        0.        0.      ]
                  [ 0.        0.864274  0.        0.      ]
                  [ 0.        0.        0.864274  0.      ]
                  [ 0.        0.        0.        1.      ]]
    Offset     : [ 0.  0.  0.  0.]
    """

    matrix = np.loadtxt(path, dtype=DEFAULT_FLOAT_DTYPE)
    matrix = np.reshape(matrix, (3, 4))
    offset = matrix[:, 3] / 65535
    matrix = matrix[:3, :3]

    title = path_to_title(path)

    return LUTOperatorMatrix(matrix, offset, name=title)


def write_LUT_SonySPImtx(
    LUT: LUTOperatorMatrix, path: str, decimals: Integer = 7
) -> Boolean:
    """
    Write given *LUT* to given *Sony* *.spimtx* *LUT* file.

    Parameters
    ----------
    LUT
        :class:`colour.LUTOperatorMatrix` class instance to write at given
        path.
    path
        *LUT* path.
    decimals
        Formatting decimals.

    Returns
    -------
    :class:`bool`
        Definition success.

    Examples
    --------
    >>> matrix = np.array([[ 1.45143932, -0.23651075, -0.21492857],
    ...                    [-0.07655377,  1.1762297 , -0.09967593],
    ...                    [ 0.00831615, -0.00603245,  0.9977163 ]])
    >>> M = LUTOperatorMatrix(matrix)
    >>> write_LUT_SonySPI1D(M, 'My_LUT.spimtx')  # doctest: +SKIP
    """

    matrix, offset = LUT.matrix, LUT.offset
    offset *= 65535

    array = np.hstack(
        [
            np.reshape(matrix, (4, 4))[:3, :3],
            np.transpose(np.array([offset[:3]])),
        ]
    )

    np.savetxt(path, array, fmt=f"%.{decimals}f")

    return True
