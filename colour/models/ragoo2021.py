"""
Ragoo and Farup (2021) Optimised IPT Colourspace
================================================

Defines the *Ragoo and Farup (2021)* *Optimised IPT* colourspace
transformations:

-   :func:`colour.XYZ_to_IPT_Ragoo2021`
-   :func:`colour.IPT_Ragoo2021_to_XYZ`

References
----------
-   :cite:`Ragoo2021` : Ragoo, L., & Farup, I. (2021). Optimising a Euclidean
    Colour Space Transform for Colour Order and Perceptual Uniformity.
    Color and Imaging Conference, 29(1), 282-287.
    doi:10.2352/issn.2169-2629.2021.29.282
"""

from __future__ import annotations

import numpy as np
from functools import partial

from colour.algebra import spow
from colour.models import Iab_to_XYZ, XYZ_to_Iab
from colour.hints import ArrayLike, NDArrayFloat

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MATRIX_IPT_XYZ_TO_LMS",
    "MATRIX_IPT_LMS_TO_XYZ",
    "MATRIX_IPT_LMS_P_TO_IPT",
    "MATRIX_IPT_IPT_TO_LMS_P",
    "XYZ_to_IPT_Ragoo2021",
    "IPT_Ragoo2021_to_XYZ",
]

MATRIX_IPT_XYZ_TO_LMS: NDArrayFloat = np.array(
    [
        [0.4321, 0.6906, -0.0930],
        [-0.1793, 1.1458, 0.0226],
        [0.0631, 0.1532, 0.7226],
    ]
)
"""*CIE XYZ* tristimulus values to normalised cone responses matrix."""

MATRIX_IPT_LMS_TO_XYZ: NDArrayFloat = np.linalg.inv(MATRIX_IPT_XYZ_TO_LMS)
"""Normalised cone responses to *CIE XYZ* tristimulus values matrix."""

MATRIX_IPT_LMS_P_TO_IPT: NDArrayFloat = np.array(
    [
        [0.3037, 0.6688, 0.0276],
        [3.9247, -4.7339, 0.8093],
        [1.5932, -0.5205, -1.0727],
    ]
)
"""
Normalised non-linear cone responses to *Ragoo and Farup (2021)*
*Optimised IPT* colourspace matrix.
"""

MATRIX_IPT_IPT_TO_LMS_P: NDArrayFloat = np.linalg.inv(MATRIX_IPT_LMS_P_TO_IPT)
"""
*Ragoo and Farup (2021)* *Optimised IPT* colourspace to normalised
non-linear cone responses matrix.
"""


def XYZ_to_IPT_Ragoo2021(XYZ: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to
    *Ragoo and Farup (2021)* *Optimised IPT* colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *Ragoo and Farup (2021)* *Optimised IPT* colourspace array.

    Notes
    -----
    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``IPT``    | ``I`` : [0, 1]        | ``I`` : [0, 1]  |
    |            |                       |                 |
    |            | ``P`` : [-1, 1]       | ``P`` : [-1, 1] |
    |            |                       |                 |
    |            | ``T`` : [-1, 1]       | ``T`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    -   Input *CIE XYZ* tristimulus values must be adapted to
        *CIE Standard Illuminant D Series* *D65*.

    References
    ----------
    :cite:`Ragoo2021`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_IPT_Ragoo2021(XYZ)  # doctest: +ELLIPSIS
    array([ 0.4224824...,  0.2910514...,  0.2041066...])
    """

    return XYZ_to_Iab(
        XYZ,
        partial(spow, p=0.4071),
        MATRIX_IPT_XYZ_TO_LMS,
        MATRIX_IPT_LMS_P_TO_IPT,
    )


def IPT_Ragoo2021_to_XYZ(IPT: ArrayLike) -> NDArrayFloat:
    """
    Convert from *Ragoo and Farup (2021)* *Optimised IPT* colourspace to
    *CIE XYZ* tristimulus values.

    Parameters
    ----------
    IPT
        *Ragoo and Farup (2021)* *Optimised IPT* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``IPT``    | ``I`` : [0, 1]        | ``I`` : [0, 1]  |
    |            |                       |                 |
    |            | ``P`` : [-1, 1]       | ``P`` : [-1, 1] |
    |            |                       |                 |
    |            | ``T`` : [-1, 1]       | ``T`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Ragoo2021`

    Examples
    --------
    >>> IPT = np.array([0.42248243, 0.2910514, 0.20410663])
    >>> IPT_Ragoo2021_to_XYZ(IPT)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    return Iab_to_XYZ(
        IPT,
        partial(spow, p=1 / 0.4071),
        MATRIX_IPT_IPT_TO_LMS_P,
        MATRIX_IPT_LMS_TO_XYZ,
    )
