"""
Sony .spimtx LUT Format Input / Output Utilities
================================================

Define the *Sony* *.spimtx* *LUT* format related input / output utilities
objects:

-   :func:`colour.io.read_LUT_SonySPImtx`
-   :func:`colour.io.write_LUT_SonySPImtx`
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import DTYPE_FLOAT_DEFAULT

if typing.TYPE_CHECKING:
    from colour.hints import PathLike

from colour.io.luts import LUTOperatorMatrix
from colour.io.luts.common import path_to_title

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "read_LUT_SonySPImtx",
    "write_LUT_SonySPImtx",
]


def read_LUT_SonySPImtx(path: str | PathLike) -> LUTOperatorMatrix:
    """
    Read the specified *Sony* *.spimtx* *LUT* file.

    Parse the *.spimtx* format which contains a 3x4 matrix stored as 12
    values. Extract the 3x3 transformation matrix and offset vector from
    the fourth column (scaled by 65535) to create a
    :class:`colour.LUTOperatorMatrix` instance.

    Parameters
    ----------
    path
        *LUT* file path.

    Returns
    -------
    :class:`colour.LUTOperatorMatrix`
        *LUT* operator matrix instance containing the extracted 3x3 matrix
        and offset vector.

    Examples
    --------
    >>> import os
    >>> path = os.path.join(
    ...     os.path.dirname(__file__),
    ...     "tests",
    ...     "resources",
    ...     "sony_spimtx",
    ...     "dt.spimtx",
    ... )
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

    path = str(path)

    matrix = np.loadtxt(path, dtype=DTYPE_FLOAT_DEFAULT)
    matrix = np.reshape(matrix, (3, 4))
    offset = matrix[:, 3] / 65535
    matrix = matrix[:3, :3]

    title = path_to_title(path)

    return LUTOperatorMatrix(matrix, offset, name=title)


def write_LUT_SonySPImtx(
    LUT: LUTOperatorMatrix,
    path: str | PathLike | typing.IO[typing.Any],
    decimals: int = 7,
) -> bool:
    """
    Write the specified *LUT* to the specified *Sony* *.spimtx* *LUT* file.

    Parameters
    ----------
    LUT
        :class:`LUTOperatorMatrix` class instance to write at the
        specified path.
    path
        *LUT* file path.
    decimals
        Number of decimal places for formatting numeric values.

    Returns
    -------
    :class:`bool`
        Definition success.

    Examples
    --------
    >>> matrix = np.array(
    ...     [
    ...         [1.45143932, -0.23651075, -0.21492857],
    ...         [-0.07655377, 1.1762297, -0.09967593],
    ...         [0.00831615, -0.00603245, 0.9977163],
    ...     ]
    ... )
    >>> M = LUTOperatorMatrix(matrix)
    >>> write_LUT_SonySPImtx(M, "My_LUT.spimtx")  # doctest: +SKIP
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
