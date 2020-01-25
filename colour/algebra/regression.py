# -*- coding: utf-8 -*-
"""
Regression
==========

Defines various objects to perform regression:

-   :func:`colour.algebra.least_square_mapping_MoorePenrose`: *Least-squares*
    mapping using *Moore-Penrose* inverse.

References
----------
-   :cite:`Finlayson2015` : Finlayson, G. D., MacKiewicz, M., & Hurlbert, A.
    (2015). Color Correction Using Root-Polynomial Regression. IEEE
    Transactions on Image Processing, 24(5), 1460-1470.
    doi:10.1109/TIP.2015.2405336
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['least_square_mapping_MoorePenrose']


def least_square_mapping_MoorePenrose(y, x):
    """
    Computes the *least-squares* mapping from dependent variable :math:`y` to
    independent variable :math:`x` using *Moore-Penrose* inverse.

    Parameters
    ----------
    y : array_like
        Dependent and already known :math:`y` variable.
    x : array_like, optional
        Independent :math:`x` variable(s) values corresponding with :math:`y`
        variable.

    Returns
    -------
    ndarray
        *Least-squares* mapping.

    References
    ----------
    :cite:`Finlayson2015`

    Examples
    --------
    >>> prng = np.random.RandomState(2)
    >>> y = prng.random_sample((24, 3))
    >>> x = y + (prng.random_sample((24, 3)) - 0.5) * 0.5
    >>> least_square_mapping_MoorePenrose(y, x)  # doctest: +ELLIPSIS
    array([[ 1.0526376...,  0.1378078..., -0.2276339...],
           [ 0.0739584...,  1.0293994..., -0.1060115...],
           [ 0.0572550..., -0.2052633...,  1.1015194...]])
    """

    y = np.atleast_2d(y)
    x = np.atleast_2d(x)

    return np.dot(np.transpose(x), np.linalg.pinv(np.transpose(y)))
