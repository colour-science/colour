"""
sUCS Colourspace
================

Define the *sUCS* colourspace transformations:

-   :func:`colour.XYZ_to_sUCS`
-   :func:`colour.sUCS_to_XYZ`

The sUCS (Simple Uniform Colour Space) is designed for simplicity and perceptual
uniformity. This implementation is based on the work by Li & Luo (2024).

References
----------
-   :cite:`Li2024` : Li, M., & Luo, M. R. (2024). Simple color appearance model
    (sCAM) based on simple uniform color space (sUCS). Optics Express, 32(3),
    3100-3122. doi:10.1364/OE.510196
"""

from __future__ import annotations

from typing import TYPE_CHECKING  # Added for TC001

import numpy as np

from colour.algebra import spow, vecmul
from colour.utilities import (
    from_range_1,
    to_domain_1,
    tsplit,
    tstack,
)

if TYPE_CHECKING:  # Added for TC001
    from colour.hints import ArrayLike, NDArrayFloat


__author__ = "UltraMo114(Molin Li), Colour Developers"
__copyright__ = "Copyright 2024 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MATRIX_SUCS_XYZ_TO_LMS",
    "MATRIX_SUCS_LMS_TO_XYZ",
    "MATRIX_SUCS_LMS_P_TO_IAB",
    "MATRIX_SUCS_IAB_TO_LMS_P",
    "XYZ_to_sUCS",
    "sUCS_to_XYZ",
]

MATRIX_SUCS_XYZ_TO_LMS: NDArrayFloat = np.array(
    [
        [0.4002, 0.7075, -0.0807],
        [-0.2280, 1.1500, 0.0612],
        [0.0000, 0.0000, 0.9184],
    ]
)
"""
*CIE XYZ* tristimulus values (D65-adapted, Y=1 for white) to LMS-like cone
responses matrix for sUCS.
"""

MATRIX_SUCS_LMS_TO_XYZ: NDArrayFloat = np.linalg.inv(MATRIX_SUCS_XYZ_TO_LMS)
"""
LMS-like cone responses to *CIE XYZ* tristimulus values
(D65-adapted, Y=1 for white) matrix for sUCS.
"""

MATRIX_SUCS_LMS_P_TO_IAB: NDArrayFloat = np.array(
    [
        [200.0 / 3.05, 100.0 / 3.05, 5.0 / 3.05],  # Approx: [65.57, 32.78, 1.64]
        [430.0, -470.0, 40.0],
        [49.0, 49.0, -98.0],
    ]
)
"""
Non-linear LMS-like responses (`LMS_p`) to intermediate `I_S A_I B_I`
colourspace matrix for sUCS. `I_S` is the final lightness-like component.
`A_I` and `B_I` are intermediate chromatic components.
"""

MATRIX_SUCS_IAB_TO_LMS_P: NDArrayFloat = np.linalg.inv(MATRIX_SUCS_LMS_P_TO_IAB)
"""
Intermediate `I_S A_I B_I` colourspace to non-linear LMS-like responses
(`LMS_p`) matrix for sUCS.
"""


def XYZ_to_sUCS(XYZ: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to *sUCS* colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values, assumed to be adapted to
        *CIE Standard Illuminant D65* and in domain [0, 1] (where D65 white
        Y is 1.0).

    Returns
    -------
    :class:`numpy.ndarray`
        *sUCS* colourspace array as `[I_S, A_S, B_S]`.

    Notes
    -----
    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+=================+
    | ``sUCS``   | ``I_S`` : [0, ~100]   | ``I_S`` : [0, ~1] |
    |            | ``A_S`` : [-X, +X]    | ``A_S`` : [-Y, +Y]|
    |            | ``B_S`` : [-X, +X]    | ``B_S`` : [-Y, +Y]|
    +------------+-----------------------+-----------------+
    (Note: Exact range for A_S, B_S depends on C_S scaling. I_S is typically
    0-100 in reference scale. If domain_range_scale("1") is active, I_S would
    be scaled by 1/100, A_S/B_S might also be scaled if their reference range
    is known and large.)

    References
    ----------
    :cite:`Li2024`

    Examples
    --------
    >>> XYZ_d65_sample = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_sUCS(XYZ_d65_sample)  # doctest: +ELLIPSIS
    array([ 42.62923...,  37.75997...,  14.42227...])
    >>> XYZ_d65_white = np.array([0.95047, 1.00000, 1.08883])  # D65 Y=1
    >>> XYZ_to_sUCS(XYZ_d65_white)  # doctest: +ELLIPSIS
    array([  9.99992575e+01,   2.79134110e-02,  -9.03996769e-04])
    """
    XYZ_arr = to_domain_1(XYZ)

    # Step 1: Convert D65-adapted XYZ (Y=1 for white) to LMS cone responses
    # MATRIX_SUCS_XYZ_TO_LMS expects XYZ where D65 white Y=1.
    LMS = vecmul(MATRIX_SUCS_XYZ_TO_LMS, XYZ_arr)

    # Step 2: Apply nonlinear transformation to LMS values (LMS_p)
    LMS_p = spow(LMS, 0.43)

    # Step 3: Transform non-linear LMS_p to I_S A_I B_I coordinates
    # I_S is the final lightness-like component. A_I, B_I are intermediate.
    IAB_intermediate = vecmul(MATRIX_SUCS_LMS_P_TO_IAB, LMS_p)
    I_S, A_I, B_I = tsplit(IAB_intermediate)

    # Step 4: Calculate intermediate chroma C_I from A_I, B_I
    C_I = np.sqrt(A_I**2 + B_I**2)

    # Step 5: Apply logarithmic compression to C_I
    # C_I >= 0, so 1 + 0.0447 * C_I >= 1
    C_S = np.log(1 + 0.0447 * C_I) / 0.0252

    # Step 6: Calculate final A_S, B_S using the ratio of C_S to C_I
    # Handle C_I = 0 case to avoid division by zero.
    ratio = np.zeros_like(C_I)
    non_zero_C_I = C_I != 0
    # C_I[non_zero_C_I] is not zero due to the mask. (Shortened E501 comment)
    ratio[non_zero_C_I] = C_S[non_zero_C_I] / C_I[non_zero_C_I]

    A_S = ratio * A_I
    B_S = ratio * B_I
    Iab = tstack([I_S, A_S, B_S])
    return from_range_1(Iab)


def sUCS_to_XYZ(sUCS: ArrayLike) -> NDArrayFloat:
    """
    Convert from *sUCS* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    sUCS
        *sUCS* colourspace array as `[I_S, A_S, B_S]`. `I_S` is assumed to be
        in its reference range [0, ~100].

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values, adapted to *CIE Standard Illuminant D65*
        and in domain [0, 1] (where D65 white Y is 1.0).

    Notes
    -----
    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+=================+
    | ``sUCS``   | ``I_S`` : [0, ~100]   | ``I_S`` : [0, ~1] |
    |            | ``A_S`` : [-X, +X]    | ``A_S`` : [-Y, +Y]|
    |            | ``B_S`` : [-X, +X]    | ``B_S`` : [-Y, +Y]|
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Li2024`

    Examples
    --------
    >>> sUCS_sample = np.array([35.65885236, 22.10004031, 9.01985036])
    >>> sUCS_to_XYZ(sUCS_sample)  # doctest: +ELLIPSIS
    array([ 0.11319...,  0.08469...,  0.05609...])

    >>> # Round trip for D65 white
    >>> sUCS_white_input = np.array([99.9992575, 0.0279134, -0.0009040])
    >>> sUCS_to_XYZ(sUCS_white_input)  # doctest: +ELLIPSIS
    array([ 0.95047,  1.     ,  1.08883])
    """
    I_S, A_S, B_S = tsplit(to_domain_1(sUCS))

    # Step 1: Calculate final chroma magnitude C_S from A_S, B_S
    C_S = np.sqrt(A_S**2 + B_S**2)

    # Step 2: Reverse logarithmic compression to get intermediate chroma C_I
    C_I = (np.exp(0.0252 * C_S) - 1) / 0.0447
    C_I = np.maximum(0, C_I)  # Ensure C_I is non-negative

    # Step 3: Calculate intermediate A_I, B_I using the ratio of C_I to C_S
    reverse_ratio = np.zeros_like(C_S)
    non_zero_C_S = C_S != 0
    reverse_ratio[non_zero_C_S] = C_I[non_zero_C_S] / C_S[non_zero_C_S]

    A_I = reverse_ratio * A_S
    B_I = reverse_ratio * B_S

    # Step 4: Form intermediate IAB array (I_S, A_I, B_I)
    IAB_intermediate = tstack([I_S, A_I, B_I])

    # Step 5: Transform IAB_intermediate to non-linear LMS_p
    LMS_p = vecmul(MATRIX_SUCS_IAB_TO_LMS_P, IAB_intermediate)

    # Step 6: Reverse non-linear transformation from LMS_p to LMS
    LMS = spow(LMS_p, 1.0 / 0.43)

    # Step 7: Convert LMS to D65-adapted XYZ (Y=1 for D65 white)
    XYZ_d65_scaled = vecmul(MATRIX_SUCS_LMS_TO_XYZ, LMS)

    return from_range_1(XYZ_d65_scaled)
