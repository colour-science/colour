"""
:math:`IC_AC_B` Colourspace
===========================

Defines the :math:`IC_AC_B` colourspace transformations:

-   :func:`colour.XYZ_to_ICaCb`
-   :func:`colour.ICaCb_to_XYZ`

References
----------
-   :cite:`Frohlich2017` : Fröhlich, J. (2017). Encoding high dynamic range
    and wide color gamut imagery. doi:10.18419/OPUS-9664
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, NDArrayFloat
from colour.models import Iab_to_XYZ, XYZ_to_Iab
from colour.models.rgb.transfer_functions import (
    eotf_inverse_ST2084,
    eotf_ST2084,
)
from colour.utilities import domain_range_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MATRIX_ICACB_XYZ_TO_LMS",
    "MATRIX_ICACB_LMS_TO_XYZ",
    "MATRIX_ICACB_XYZ_TO_LMS_2",
    "MATRIX_ICACB_LMS_TO_XYZ_2",
    "XYZ_to_ICaCb",
    "ICaCb_to_XYZ",
]

MATRIX_ICACB_XYZ_TO_LMS: NDArrayFloat = np.array(
    [
        [0.37613, 0.70431, -0.05675],
        [-0.21649, 1.14744, 0.05356],
        [0.02567, 0.16713, 0.74235],
    ]
)
"""*CIE XYZ* tristimulus values to normalised cone responses matrix."""

MATRIX_ICACB_LMS_TO_XYZ: NDArrayFloat = np.linalg.inv(MATRIX_ICACB_XYZ_TO_LMS)
"""Normalised cone responses to *CIE XYZ* tristimulus values matrix."""

MATRIX_ICACB_XYZ_TO_LMS_2: NDArrayFloat = np.array(
    [
        [0.4949, 0.5037, 0.0015],
        [4.2854, -4.5462, 0.2609],
        [0.3605, 1.1499, -1.5105],
    ]
)
"""Normalised non-linear cone responses to :math:`IC_AC_B` colourspace matrix."""

MATRIX_ICACB_LMS_TO_XYZ_2: NDArrayFloat = np.linalg.inv(
    MATRIX_ICACB_XYZ_TO_LMS_2
)
""":math:`IC_AC_B` to normalised non-linear cone responses colourspace matrix."""


def XYZ_to_ICaCb(XYZ: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to :math:`IC_AC_B` colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        :math:`IC_AC_B` colourspace array.

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
    | ``ICaCb``  | ``I`` : [0, 1]        | ``I`` : [0, 1]  |
    |            |                       |                 |
    |            | ``Ca`` : [-1, 1]      | ``Ca``: [-1, 1] |
    |            |                       |                 |
    |            | ``Cb`` : [-1, 1]      | ``Cb``: [-1, 1] |
    +------------+-----------------------+-----------------+

    -   Input *CIE XYZ* tristimulus values must be adapted to
        *CIE Standard Illuminant D Series* *D65*.

    References
    ----------
    :cite:`Frohlich2017`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_ICaCb(XYZ)
    array([ 0.06875297,  0.05753352,  0.02081548])
    """

    def LMS_to_LMS_p_callable(LMS: ArrayLike) -> NDArrayFloat:
        """
        Callable applying the forward non-linearity to the :math:`LMS`
        colourspace array.
        """

        with domain_range_scale("ignore"):
            return eotf_inverse_ST2084(LMS)

    return XYZ_to_Iab(
        XYZ,
        LMS_to_LMS_p_callable,
        MATRIX_ICACB_XYZ_TO_LMS,
        MATRIX_ICACB_XYZ_TO_LMS_2,
    )


def ICaCb_to_XYZ(ICaCb: ArrayLike) -> NDArrayFloat:
    """
    Convert from :math:`IC_AC_B` tristimulus values to *CIE XYZ* colourspace.

    Parameters
    ----------
    ICaCb
        :math:`IC_AC_B` tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* colourspace array.

    Notes
    -----
    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``ICaCb``  | ``I`` : [0, 1]        | ``I`` : [0, 1]  |
    |            |                       |                 |
    |            | ``Ca`` : [-1, 1]      | ``Ca``: [-1, 1] |
    |            |                       |                 |
    |            | ``Cb`` : [-1, 1]      | ``Cb``: [-1, 1] |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Frohlich2017`

    Examples
    --------
    >>> XYZ = np.array([0.06875297, 0.05753352, 0.02081548])
    >>> ICaCb_to_XYZ(XYZ)
    array([ 0.20654008,  0.12197225,  0.05136951])
    """

    def LMS_p_to_LMS_callable(LMS_p: ArrayLike) -> NDArrayFloat:
        """
        Callable applying the reverse non-linearity to the :math:`LMS_p`
        colourspace array.
        """

        with domain_range_scale("ignore"):
            return eotf_ST2084(LMS_p)

    return Iab_to_XYZ(
        ICaCb,
        LMS_p_to_LMS_callable,
        MATRIX_ICACB_LMS_TO_XYZ_2,
        MATRIX_ICACB_LMS_TO_XYZ,
    )
