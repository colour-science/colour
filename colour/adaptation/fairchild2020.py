"""
Von Kries 2020 (vK20) Chromatic Adaptation Model
================================================

Defines the *Von Kries 2020* (*vK20*) chromatic adaptation model objects:

-   :attr:`colour.adaptation.CONDITIONS_DEGREE_OF_ADAPTATION_VK20`
-   :func:`colour.adaptation.matrix_chromatic_adaptation_vk20`
-   :func:`colour.adaptation.chromatic_adaptation_vK20`

References
----------
-   :cite:`Fairchild2020` : Fairchild, M. D. (2020). Von Kries 2020: Evolution
    of degree of chromatic adaptation. Color and Imaging Conference, 28(1),
    252-257. doi:10.2352/issn.2169-2629.2020.28.40
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from colour.adaptation import CHROMATIC_ADAPTATION_TRANSFORMS
from colour.algebra import matrix_dot, sdiv, sdiv_mode, vector_dot
from colour.hints import ArrayLike, Literal, NDArrayFloat
from colour.utilities import (
    CanonicalMapping,
    as_float_array,
    from_range_1,
    row_as_diagonal,
    to_domain_1,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Coefficients_DegreeOfAdaptation_vK20",
    "CONDITIONS_DEGREE_OF_ADAPTATION_VK20",
    "TVS_XYZ_R_VK20",
    "matrix_chromatic_adaptation_vk20",
    "chromatic_adaptation_vK20",
]


class Coefficients_DegreeOfAdaptation_vK20(
    namedtuple("Coefficients_DegreeOfAdaptation_vK20", ("D_n", "D_r", "D_p"))
):
    """
    *Von Kries 2020* (*vK20*) degree of adaptation coefficients.

    Parameters
    ----------
    D_n
        Degree of adaptation for the adapting illuminant.
    D_r
        Degree of adaptation for the reference illuminant.
    D_p
        Degree of adaptation for the previous illuminant.

    References
    ----------
    :cite:`Fairchild2020`
    """


CONDITIONS_DEGREE_OF_ADAPTATION_VK20: CanonicalMapping = CanonicalMapping(
    {
        "Fairchild": Coefficients_DegreeOfAdaptation_vK20(0.7, 0.3, 0),
        "Hands": Coefficients_DegreeOfAdaptation_vK20(0.95, 0.05, 0),
        "No Hands": Coefficients_DegreeOfAdaptation_vK20(0.85, 0.15, 0),
        "Ordinal 1st": Coefficients_DegreeOfAdaptation_vK20(0.9, 0.1, 0),
        "Ordinal 2nd": Coefficients_DegreeOfAdaptation_vK20(0.8, 0.1, 0.1),
        "Reversibility Trial 1st": Coefficients_DegreeOfAdaptation_vK20(0.7, 0.3, 0.1),
        "Reversibility Trial 2nd": Coefficients_DegreeOfAdaptation_vK20(0.6, 0.3, 0.1),
        "Ma et al.": Coefficients_DegreeOfAdaptation_vK20(1 / 3, 1 / 3, 1 / 3),
        "Hunt & Winter": Coefficients_DegreeOfAdaptation_vK20(0.6, 0.2, 0.2),
        "Hurvich & Jameson": Coefficients_DegreeOfAdaptation_vK20(0.7, 0.3, 0),
        "Simple von Kries": Coefficients_DegreeOfAdaptation_vK20(1, 0, 0),
    }
)
CONDITIONS_DEGREE_OF_ADAPTATION_VK20.__doc__ = """
Conditions for the *Von Kries 2020* (*vK20*) degree of adaptation coefficients.

References
----------
:cite:`Fairchild2020`
"""

TVS_XYZ_R_VK20 = np.array([0.97941176, 1.00000000, 1.73235294])
"""
*Von Kries 2020* (*vK20*) reference illuminant (taken to be
u' = 0.185, v' = 0.425, approximately 15000K, sky blue).

References
----------
:cite:`Fairchild2020`
"""


def matrix_chromatic_adaptation_vk20(
    XYZ_p: ArrayLike,
    XYZ_n: ArrayLike,
    XYZ_r: ArrayLike = TVS_XYZ_R_VK20,
    transform: Literal[
        "Bianco 2010",
        "Bianco PC 2010",
        "Bradford",
        "CAT02 Brill 2008",
        "CAT02",
        "CAT16",
        "CMCCAT2000",
        "CMCCAT97",
        "Fairchild",
        "Sharp",
        "Von Kries",
        "XYZ Scaling",
    ]
    | str = "CAT02",
    coefficients: Coefficients_DegreeOfAdaptation_vK20 = (
        CONDITIONS_DEGREE_OF_ADAPTATION_VK20["Fairchild"]
    ),
) -> NDArrayFloat:
    """
    Compute the *chromatic adaptation* matrix from previous viewing conditions
    to adapting viewing conditions using *Von Kries 2020* (*vK20*) method.

    Parameters
    ----------
    XYZ_p
        Previous viewing conditions *CIE XYZ* tristimulus values of whitepoint.
    XYZ_n
        Adapting viewing conditions *CIE XYZ* tristimulus values of whitepoint.
    XYZ_r
        Reference viewing conditions *CIE XYZ* tristimulus values of
        whitepoint.
    transform
        Chromatic adaptation transform.
    coefficients
        *vK20* degree of adaptation coefficients.

    Returns
    -------
    :class:`numpy.ndarray`
        Chromatic adaptation matrix :math:`M_{cat}`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_p``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_n``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_r``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2020`

    Examples
    --------
    >>> XYZ_p = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> XYZ_n = np.array([0.96429568, 1.00000000, 0.82510460])
    >>> matrix_chromatic_adaptation_vk20(XYZ_p, XYZ_n)
    ... # doctest: +ELLIPSIS
    array([[  1.0279139...e+00,   2.9137117...e-02,  -2.2794068...e-02],
           [  2.0702840...e-02,   9.9005316...e-01,  -9.2143464...e-03],
           [ -6.3758553...e-04,  -1.1577319...e-03,   9.1296320...e-01]])

    Using *Bradford* transform:

    >>> XYZ_p = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> XYZ_n = np.array([0.96429568, 1.00000000, 0.82510460])
    >>> transform = "Bradford"
    >>> matrix_chromatic_adaptation_vk20(XYZ_p, XYZ_n, transform=transform)
    ... # doctest: +ELLIPSIS
    array([[ 1.0367230...,  0.0195580..., -0.0219321...],
           [ 0.0276321...,  0.9822296..., -0.0082419...],
           [-0.0029508...,  0.0040690...,  0.9102430...]])
    """

    XYZ_n = as_float_array(XYZ_n)
    XYZ_r = as_float_array(XYZ_r)
    XYZ_p = as_float_array(XYZ_p)

    transform = validate_method(
        transform,
        tuple(CHROMATIC_ADAPTATION_TRANSFORMS),
        '"{0}" chromatic adaptation transform is invalid, it must be one of {1}!',
    )

    M = CHROMATIC_ADAPTATION_TRANSFORMS[transform]

    D_n, D_r, D_p = coefficients

    LMS_n = vector_dot(M, XYZ_n)
    LMS_r = vector_dot(M, XYZ_r)
    LMS_p = vector_dot(M, XYZ_p)

    with sdiv_mode():
        D = row_as_diagonal(sdiv(1, (D_n * LMS_n + D_r * LMS_r + D_p * LMS_p)))

    M_CAT = matrix_dot(np.linalg.inv(M), D)
    M_CAT = matrix_dot(M_CAT, M)

    return M_CAT


def chromatic_adaptation_vK20(
    XYZ: ArrayLike,
    XYZ_p: ArrayLike,
    XYZ_n: ArrayLike,
    XYZ_r: ArrayLike = TVS_XYZ_R_VK20,
    transform: Literal[
        "Bianco 2010",
        "Bianco PC 2010",
        "Bradford",
        "CAT02 Brill 2008",
        "CAT02",
        "CAT16",
        "CMCCAT2000",
        "CMCCAT97",
        "Fairchild",
        "Sharp",
        "Von Kries",
        "XYZ Scaling",
    ]
    | str = "CAT02",
    coefficients: Coefficients_DegreeOfAdaptation_vK20 = (
        CONDITIONS_DEGREE_OF_ADAPTATION_VK20["Fairchild"]
    ),
) -> NDArrayFloat:
    """
    Adapt given stimulus from previous viewing conditions to adapting viewing
    conditions using *Von Kries 2020* (*vK20*) method.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of stimulus to adapt.
    XYZ_p
        Previous viewing conditions *CIE XYZ* tristimulus values of whitepoint.
    XYZ_n
        Adapting viewing conditions *CIE XYZ* tristimulus values of whitepoint.
    XYZ_r
        Reference viewing conditions *CIE XYZ* tristimulus values of
        whitepoint.
    transform
        Chromatic adaptation transform.
    coefficients
        *vK20* degree of adaptation coefficients.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ_c* tristimulus values of the stimulus corresponding colour.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_p``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_n``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_r``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_a``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2020`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_p = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> XYZ_n = np.array([0.96429568, 1.00000000, 0.82510460])
    >>> chromatic_adaptation_vK20(XYZ, XYZ_p, XYZ_n)
    ... # doctest: +ELLIPSIS
    array([ 0.2146884...,  0.1245616...,  0.0466255...])

    Using *Bradford* transform:

    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_p = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> XYZ_n = np.array([0.96429568, 1.00000000, 0.82510460])
    >>> transform = "Bradford"
    >>> chromatic_adaptation_vK20(XYZ, XYZ_p, XYZ_n, transform=transform)
    ... # doctest: +ELLIPSIS
    array([ 0.2153837...,  0.1250885...,  0.0466455...])
    """

    XYZ = to_domain_1(XYZ)
    XYZ_p = to_domain_1(XYZ_p)
    XYZ_n = to_domain_1(XYZ_n)
    XYZ_r = to_domain_1(XYZ_r)

    M_CAT = matrix_chromatic_adaptation_vk20(
        XYZ_p, XYZ_n, XYZ_r, transform, coefficients
    )
    XYZ_a = vector_dot(M_CAT, XYZ)

    return from_range_1(XYZ_a)
