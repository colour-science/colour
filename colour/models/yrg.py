"""
Yrg Colourspace - Kirk (2019)
=============================

Define the *Kirk (2019)* *Yrg* colourspace:

-   :func:`colour.models.LMS_to_Yrg`
-   :func:`colour.models.Yrg_to_LMS`
-   :func:`colour.XYZ_to_Yrg`
-   :func:`colour.Yrg_to_XYZ`

References
----------
-   :cite:`Kirk2019` : Kirk, R. A. (2019). Chromaticity coordinates for graphic
    arts based on CIE 2006 LMS with even spacing of Munsell colours. Color and
    Imaging Conference, 27(1), 215-219. doi:10.2352/issn.2169-2629.2019.27.38
"""

from __future__ import annotations

import numpy as np

from colour.algebra import sdiv, sdiv_mode, vecmul
from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import (
    from_range_1,
    to_domain_1,
    tsplit,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "LMS_to_Yrg",
    "Yrg_to_LMS",
    "XYZ_to_Yrg",
    "Yrg_to_XYZ",
]

MATRIX_XYZ_TO_LMS_KIRK2019: NDArrayFloat = np.array(
    [
        [0.257085, 0.859943, -0.031061],
        [-0.394427, 1.175800, 0.106423],
        [0.064856, -0.076250, 0.559067],
    ]
)
"""
*Kirk (2019)* matrix converting from *CIE XYZ* tristimulus values to *LMS*
colourspace.
"""

MATRIX_LMS_TO_XYZ_KIRK2019: NDArrayFloat = np.linalg.inv(MATRIX_XYZ_TO_LMS_KIRK2019)
"""
*Kirk (2019)* matrix converting from *LMS* colourspace to *CIE XYZ* tristimulus
values.
"""


def LMS_to_Yrg(LMS: ArrayLike) -> NDArrayFloat:
    """
    Convert from *LMS* colourspace to *Kirk (2019)* *Yrg* colourspace.

    Parameters
    ----------
    LMS
        *LMS* colourspace values.

    Returns
    -------
    :class:`numpy.ndarray`
        *Kirk (2019)* *Yrg* :math:`Yrg` luminance, redness, and greenness.

    Notes
    -----
    +------------+-----------------------+----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**  |
    +============+=======================+================+
    | ``LMS``    | [0, 100]              | [0, 1]         |
    +------------+-----------------------+----------------+

    +------------+-----------------------+----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**  |
    +============+=======================+================+
    | ``Yrg``    | ``Y`` : [0, 1]        | ``Y`` : [0, 1] |
    |            |                       |                |
    |            | ``r`` : [0, 1]        | ``r`` : [0, 1] |
    |            |                       |                |
    |            | ``g`` : [0, 1]        | ``g`` : [0, 1] |
    +------------+-----------------------+----------------+

    References
    ----------
    :cite:`Kirk2019`

    Examples
    --------
    >>> import numpy as np
    >>> LMS = np.array([0.15639195, 0.06741689, 0.03281398])
    >>> LMS_to_Yrg(LMS)  # doctest: +ELLIPSIS
    array([ 0.1313780...,  0.4903764...,  0.3777739...])
    """

    L, M, S = tsplit(to_domain_1(LMS))

    Y = 0.68990272 * L + 0.34832189 * M

    a = L + M + S

    with sdiv_mode():
        l = sdiv(L, a)  # noqa: E741
        m = sdiv(M, a)

    r = 1.0671 * l - 0.6873 * m + 0.02062
    g = -0.0362 * l + 1.7182 * m - 0.05155

    Yrg = tstack([Y, r, g])

    return from_range_1(Yrg)


def Yrg_to_LMS(Yrg: ArrayLike) -> NDArrayFloat:
    """
    Convert from *Kirk (2019)* *Yrg* colourspace to *LMS* colourspace.

    Parameters
    ----------
    Yrg
        *Kirk (2019)* *Yrg* :math:`Yrg` luminance, redness, and greenness.

    Returns
    -------
    :class:`numpy.ndarray`
        *LMS* colourspace values.

    Notes
    -----
    +------------+-----------------------+----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**  |
    +============+=======================+================+
    | ``Yrg``    | ``Y`` : [0, 1]        | ``Y`` : [0, 1] |
    |            |                       |                |
    |            | ``r`` : [0, 1]        | ``r`` : [0, 1] |
    |            |                       |                |
    |            | ``g`` : [0, 1]        | ``g`` : [0, 1] |
    +------------+-----------------------+----------------+

    +------------+-----------------------+----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**  |
    +============+=======================+================+
    | ``LMS``    | [0, 100]              | [0, 1]         |
    +------------+-----------------------+----------------+

    References
    ----------
    :cite:`Kirk2019`

    Examples
    --------
    >>> import numpy as np
    >>> Yrg = np.array([0.13137801, 0.49037644, 0.37777391])
    >>> Yrg_to_LMS(Yrg)  # doctest: +ELLIPSIS
    array([ 0.1563929...,  0.0674150...,  0.0328213...])
    """

    Y, r, g = tsplit(to_domain_1(Yrg))

    l = 0.95 * r + 0.38 * g  # noqa: E741
    m = 0.02 * r + 0.59 * g + 0.03
    a = Y / (0.68990272 * l + 0.34832189 * m)
    L = l * a
    M = m * a
    S = (1 - l - m) * a

    LMS = tstack([L, M, S])

    return from_range_1(LMS)


def XYZ_to_Yrg(XYZ: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to *Kirk (2019)* *Yrg*
    colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values values.

    Returns
    -------
    :class:`numpy.ndarray`
        *Kirk (2019)* *Yrg* :math:`Yrg` luminance, redness, and greenness.

    Notes
    -----
    +------------+-----------------------+----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**  |
    +============+=======================+================+
    | ``XYZ``    | [0, 100]              | [0, 1]         |
    +------------+-----------------------+----------------+

    +------------+-----------------------+----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**  |
    +============+=======================+================+
    | ``Yrg``    | ``Y`` : [0, 1]        | ``Y`` : [0, 1] |
    |            |                       |                |
    |            | ``r`` : [0, 1]        | ``r`` : [0, 1] |
    |            |                       |                |
    |            | ``g`` : [0, 1]        | ``g`` : [0, 1] |
    +------------+-----------------------+----------------+

    References
    ----------
    :cite:`Kirk2019`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_Yrg(XYZ)  # doctest: +ELLIPSIS
    array([ 0.1313780...,  0.4903764...,  0.3777738...])
    """

    return LMS_to_Yrg(vecmul(MATRIX_XYZ_TO_LMS_KIRK2019, XYZ))


def Yrg_to_XYZ(Yrg: ArrayLike) -> NDArrayFloat:
    """
    Convert from *Kirk (2019)* *Yrg* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Yrg
        *Kirk (2019)* *Yrg* :math:`Yrg` luminance, redness, and greenness.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values values.

    Notes
    -----
    +------------+-----------------------+----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**  |
    +============+=======================+================+
    | ``Yrg``    | ``Y`` : [0, 1]        | ``Y`` : [0, 1] |
    |            |                       |                |
    |            | ``r`` : [0, 1]        | ``r`` : [0, 1] |
    |            |                       |                |
    |            | ``g`` : [0, 1]        | ``g`` : [0, 1] |
    +------------+-----------------------+----------------+

    +------------+-----------------------+----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**  |
    +============+=======================+================+
    | ``XYZ``    | [0, 100]              | [0, 1]         |
    +------------+-----------------------+----------------+

    References
    ----------
    :cite:`Kirk2019`

    Examples
    --------
    >>> import numpy as np
    >>> Yrg = np.array([0.13137801, 0.49037645, 0.37777388])
    >>> Yrg_to_XYZ(Yrg)  # doctest: +ELLIPSIS
    array([ 0.2065468...,  0.1219717...,  0.0513819...])
    """

    return vecmul(MATRIX_LMS_TO_XYZ_KIRK2019, Yrg_to_LMS(Yrg))
