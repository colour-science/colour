# -*- coding: utf-8 -*-
"""
Sony .spimtx LUT Format Input / Output Utilities
================================================

Defines *Sony* *.spimtx* *LUT* format related input / output utilities objects.

-   :func:`colour.io.read_LUT_SonySPImtx`
-   :func:`colour.io.write_LUT_SonySPImtx`
"""

import numpy as np

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.io.luts.common import path_to_title
from colour.io.luts import Matrix

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['read_LUT_SonySPImtx', 'write_LUT_SonySPImtx']


def read_LUT_SonySPImtx(path):
    """
    Reads given *Sony* *.spimtx* *LUT* file.

    Parameters
    ----------
    path : unicode
        *LUT* path.

    Returns
    -------
    Matrix
        :class:`colour.io.Matrix` class instance.

    Examples
    --------
    >>> import os
    >>> path = os.path.join(
    ...     os.path.dirname(__file__), 'tests', 'resources', 'sony_spimtx',
    ...     'dt.spimtx')
    >>> print(read_LUT_SonySPImtx(path))
    Matrix - dt
    -----------
    <BLANKLINE>
    Dimensions : (3, 4)
    Matrix     : [[ 0.864274  0.        0.        0.      ]
                  [ 0.        0.864274  0.        0.      ]
                  [ 0.        0.        0.864274  0.      ]]
    Offset     : [ 0.  0.  0.]
    """

    matrix = np.loadtxt(path, dtype=DEFAULT_FLOAT_DTYPE)
    matrix = matrix.reshape(3, 4)

    title = path_to_title(path)

    return Matrix(matrix, name=title)


def write_LUT_SonySPImtx(matrix, path, decimals=7):
    """
    Writes given *LUT* to given *Sony* *.spimtx* *LUT* file.

    Parameters
    ----------
    matrix : Matrix
        :class:`Matrix` class instance to write at given path.
    path : unicode
        *LUT* path.
    decimals : int, optional
        Formatting decimals.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> matrix = np.array([[ 1.45143932, -0.23651075, -0.21492857],
    ...                    [-0.07655377,  1.1762297 , -0.09967593],
    ...                    [ 0.00831615, -0.00603245,  0.9977163 ]])
    >>> M = Matrix(matrix)
    >>> write_LUT_SonySPI1D(M, 'My_LUT.spimtx')  # doctest: +SKIP
    """

    if matrix.matrix.shape == (3, 4):
        matrix = matrix.matrix
    else:
        matrix = np.hstack([matrix.matrix, np.zeros((3, 1))])

    np.savetxt(path, matrix, fmt='%.{0}f'.format(decimals).encode('ascii'))

    return True
