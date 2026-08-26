"""
Regression
==========

Various objects to perform regression:

-   :func:`colour.algebra.least_square_mapping_MoorePenrose`: *Least-squares*
    mapping using *Moore-Penrose* inverse.

References
----------
-   :cite:`Finlayson2015` : Finlayson, G. D., MacKiewicz, M., & Hurlbert, A.
    (2015). Color Correction Using Root-Polynomial Regression. IEEE
    Transactions on Image Processing, 24(5), 1460-1470.
    doi:10.1109/TIP.2015.2405336
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat

from colour.utilities import (
    array_namespace,
    as_float_array,
    xp_atleast_2d,
    xp_matrix_transpose,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "least_square_mapping_MoorePenrose",
]


def least_square_mapping_MoorePenrose(y: ArrayLike, x: ArrayLike) -> NDArrayFloat:
    """
    Compute the *least-squares* mapping from dependent variable :math:`y` to
    independent variable :math:`x` using *Moore-Penrose* inverse.

    Parameters
    ----------
    y
        Dependent and already known :math:`y` variable.
    x
        Independent :math:`x` variable(s) values corresponding with :math:`y`
        variable.

    Returns
    -------
    :class:`numpy.ndarray`
        *Least-squares* mapping matrix.

    References
    ----------
    :cite:`Finlayson2015`

    Examples
    --------
    >>> import numpy as np
    >>> prng = np.random.RandomState(2)
    >>> y = prng.random_sample((24, 3))
    >>> x = y + (prng.random_sample((24, 3)) - 0.5) * 0.5
    >>> least_square_mapping_MoorePenrose(y, x)  # doctest: +ELLIPSIS
    array([[ 1.0526376...,  0.1378078..., -0.2276339...],
           [ 0.0739584...,  1.0293994..., -0.1060115...],
           [ 0.0572550..., -0.2052633...,  1.1015194...]])
    """

    y = as_float_array(y)
    x = as_float_array(x)

    xp = array_namespace(y, x)

    y = xp_atleast_2d(y, xp=xp)
    x = xp_atleast_2d(x, xp=xp)

    return xp.matmul(
        xp_matrix_transpose(x, xp=xp),
        xp.linalg.pinv(xp_matrix_transpose(y, xp=xp)),
    )
