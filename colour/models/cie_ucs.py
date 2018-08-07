# -*- coding: utf-8 -*-
"""
CIE 1960 UCS Colourspace
========================

Defines the *CIE 1960 UCS* colourspace transformations:

-   :func:`colour.XYZ_to_UCS`
-   :func:`colour.UCS_to_XYZ`
-   :func:`colour.UCS_to_uv`
-   :func:`colour.UCS_uv_to_xy`
-   :func:`colour.xy_to_UCS_uv`

See Also
--------
`CIE UCS Colourspace Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/cie_ucs.ipynb>`_

References
----------
-   :cite:`Wikipediabr` : Wikipedia. (n.d.). Relation to CIE XYZ. Retrieved
    February 24, 2014, from http://en.wikipedia.org/wiki/\
CIE_1960_color_space#Relation_to_CIE_XYZ
-   :cite:`Wikipediabw` : Wikipedia. (n.d.). CIE 1960 color space. Retrieved
    February 24, 2014, from http://en.wikipedia.org/wiki/CIE_1960_color_space
"""

from __future__ import division, unicode_literals

from colour.utilities import from_range_1, to_domain_1, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'XYZ_to_UCS', 'UCS_to_XYZ', 'UCS_to_uv', 'UCS_uv_to_xy', 'xy_to_UCS_uv'
]


def XYZ_to_UCS(XYZ):
    """
    Converts from *CIE XYZ* tristimulus values to *CIE 1960 UCS* colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.

    Returns
    -------
    ndarray
        *CIE 1960 UCS* colourspace array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``UVW``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    -   :cite:`Wikipediabr`
    -   :cite:`Wikipediabw`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> XYZ_to_UCS(XYZ)  # doctest: +ELLIPSIS
    array([ 0.0469968...,  0.1008    ,  0.1637439...])
    """

    X, Y, Z = tsplit(to_domain_1(XYZ))

    UVW = tstack((2 / 3 * X, Y, 1 / 2 * (-X + 3 * Y + Z)))

    return from_range_1(UVW)


def UCS_to_XYZ(UVW):
    """
    Converts from *CIE 1960 UCS* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    UVW : array_like
        *CIE 1960 UCS* colourspace array.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``UVW``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    -   :cite:`Wikipediabr`
    -   :cite:`Wikipediabw`

    Examples
    --------
    >>> import numpy as np
    >>> UVW = np.array([0.04699689, 0.10080000, 0.16374390])
    >>> UCS_to_XYZ(UVW)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    """

    U, V, W = tsplit(to_domain_1(UVW))

    XYZ = tstack((3 / 2 * U, V, 3 / 2 * U - (3 * V) + (2 * W)))

    return from_range_1(XYZ)


def UCS_to_uv(UVW):
    """
    Returns the *uv* chromaticity coordinates from given *CIE 1960 UCS*
    colourspace array.

    Parameters
    ----------
    UVW : array_like
        *CIE 1960 UCS* colourspace array.

    Returns
    -------
    ndarray
        *uv* chromaticity coordinates.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``UVW``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    -   :cite:`Wikipediabr`

    Examples
    --------
    >>> import numpy as np
    >>> UCS = np.array([0.04699689, 0.10080000, 0.16374390])
    >>> UCS_to_uv(UCS)  # doctest: +ELLIPSIS
    array([ 0.1508530...,  0.3235531...])
    """

    U, V, W = tsplit(to_domain_1(UVW))

    uv = tstack((U / (U + V + W), V / (U + V + W)))

    return uv


def UCS_uv_to_xy(uv):
    """
    Returns the *xy* chromaticity coordinates from given *CIE 1960 UCS*
    colourspace *uv* chromaticity coordinates.

    Parameters
    ----------
    uv : array_like
        *CIE UCS uv* chromaticity coordinates.

    Returns
    -------
    ndarray
        *xy* chromaticity coordinates.

    References
    ----------
    -   :cite:`Wikipediabr`

    Examples
    --------
    >>> import numpy as np
    >>> uv = np.array([0.150853087327666, 0.323553137295440])
    >>> UCS_uv_to_xy(uv)  # doctest: +ELLIPSIS
    array([ 0.2641477...,  0.3777000...])
    """

    u, v = tsplit(uv)

    d = 2 * u - 8 * v + 4
    xy = tstack((3 * u / d, 2 * v / d))

    return xy


def xy_to_UCS_uv(xy):
    """
    Returns the *CIE 1960 UCS* colourspace *uv* chromaticity coordinates from
    given *xy* chromaticity coordinates.

    Parameters
    ----------
    xy : array_like
        *xy* chromaticity coordinates.

    Returns
    -------
    ndarray
        *CIE UCS uv* chromaticity coordinates.

    References
    ----------
    -   :cite:`Wikipediabr`

    Examples
    --------
    >>> import numpy as np
    >>> xy = np.array([0.2641477, 0.37770001])
    >>> xy_to_UCS_uv(xy)  # doctest: +ELLIPSIS
    array([ 0.1508530...,  0.3235531...])
    """

    x, y = tsplit(xy)

    d = 12 * y - 2 * x + 3
    uv = tstack((4 * x / d, 6 * y / d))

    return uv
