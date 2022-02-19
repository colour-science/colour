"""
Optical Society of America Uniform Colour Scales (OSA UCS)
==========================================================

Defines the *OSA UCS* colourspace:

-   :func:`colour.XYZ_to_OSA_UCS`
-   :func:`colour.OSA_UCS_to_XYZ`

References
----------
-   :cite:`Cao2013` : Cao, R., Trussell, H. J., & Shamey, R. (2013). Comparison
    of the performance of inverse transformation methods from OSA-UCS to
    CIEXYZ. Journal of the Optical Society of America A, 30(8), 1508.
    doi:10.1364/JOSAA.30.001508
-   :cite:`Moroney2003` : Moroney, N. (2003). A Radial Sampling of the OSA
    Uniform Color Scales. Color and Imaging Conference, 2003(1), 175-180.
    ISSN:2166-9635
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fmin

from colour.algebra import spow, vector_dot
from colour.hints import ArrayLike, Dict, FloatingOrNDArray, NDArray, Optional
from colour.models import XYZ_to_xyY
from colour.utilities import (
    as_float,
    as_float_array,
    domain_range_scale,
    from_range_100,
    to_domain_100,
    tsplit,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "XYZ_to_OSA_UCS",
    "OSA_UCS_to_XYZ",
]

MATRIX_XYZ_TO_RGB_OSA_UCS: NDArray = np.array(
    [
        [0.799, 0.4194, -0.1648],
        [-0.4493, 1.3265, 0.0927],
        [-0.1149, 0.3394, 0.717],
    ]
)
"""
*OSA UCS* matrix converting from *CIE XYZ* tristimulus values to *RGB*
colourspace.
"""


def XYZ_to_OSA_UCS(XYZ: ArrayLike) -> NDArray:
    """
    Convert from *CIE XYZ* tristimulus values under the
    *CIE 1964 10 Degree Standard Observer* to *OSA UCS* colourspace.

    The lightness axis, *L* is usually in range [-9, 5] and centered around
    middle gray (Munsell N/6). The yellow-blue axis, *j* is usually in range
    [-15, 15]. The red-green axis, *g* is usually in range [-20, 15].

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values under the
        *CIE 1964 10 Degree Standard Observer*.

    Returns
    -------
    :class:`numpy.ndarray`
        *OSA UCS* :math:`Ljg` lightness, jaune (yellowness), and greenness.

    Notes
    -----
    +------------+-----------------------+--------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**      |
    +============+=======================+====================+
    | ``XYZ``    | [0, 100]              | [0, 1]             |
    +------------+-----------------------+--------------------+

    +------------+-----------------------+--------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**      |
    +============+=======================+====================+
    | ``Ljg``    | ``L`` : [-100, 100]   | ``L`` : [-1, 1]    |
    |            |                       |                    |
    |            | ``j`` : [-100, 100]   | ``j`` : [-1, 1]    |
    |            |                       |                    |
    |            | ``g`` : [-100, 100]   | ``g`` : [-1, 1]    |
    +------------+-----------------------+--------------------+

    -   *OSA UCS* uses the *CIE 1964 10 Degree Standard Observer*.

    References
    ----------
    :cite:`Cao2013`, :cite:`Moroney2003`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
    >>> XYZ_to_OSA_UCS(XYZ)  # doctest: +ELLIPSIS
    array([-3.0049979...,  2.9971369..., -9.6678423...])
    """

    XYZ = to_domain_100(XYZ)
    x, y, Y = tsplit(XYZ_to_xyY(XYZ))

    Y_0 = Y * (
        4.4934 * x**2
        + 4.3034 * y**2
        - 4.276 * x * y
        - 1.3744 * x
        - 2.5643 * y
        + 1.8103
    )

    o_3 = 1 / 3
    Y_0_es = spow(Y_0, o_3) - 2 / 3
    # Gracefully handles Y_0 < 30.
    Y_0_s = Y_0 - 30
    Lambda = 5.9 * (Y_0_es + 0.042 * spow(Y_0_s, o_3))

    RGB = vector_dot(MATRIX_XYZ_TO_RGB_OSA_UCS, XYZ)
    RGB_3 = spow(RGB, 1 / 3)

    C = Lambda / (5.9 * Y_0_es)
    L = (Lambda - 14.4) / spow(2, 1 / 2)
    j = C * np.dot(RGB_3, np.array([1.7, 8, -9.7]))
    g = C * np.dot(RGB_3, np.array([-13.7, 17.7, -4]))

    Ljg = tstack([L, j, g])

    return from_range_100(Ljg)


def OSA_UCS_to_XYZ(
    Ljg: ArrayLike, optimisation_kwargs: Optional[Dict] = None
) -> NDArray:
    """
    Convert from *OSA UCS* colourspace to *CIE XYZ* tristimulus values under
    the *CIE 1964 10 Degree Standard Observer*.

    Parameters
    ----------
    Ljg
        *OSA UCS* :math:`Ljg` lightness, jaune (yellowness), and greenness.
    optimisation_kwargs
        Parameters for :func:`scipy.optimize.fmin` definition.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values under the
        *CIE 1964 10 Degree Standard Observer*.

    Warnings
    --------
    There is no analytical inverse transformation from *OSA UCS* to :math:`Ljg`
    lightness, jaune (yellowness), and greenness to *CIE XYZ* tristimulus
    values, the current implementation relies on optimization using
    :func:`scipy.optimize.fmin` definition and thus has reduced precision and
    poor performance.

    Notes
    -----
    +------------+-----------------------+--------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**      |
    +============+=======================+====================+
    | ``Ljg``    | ``L`` : [-100, 100]   | ``L`` : [-1, 1]    |
    |            |                       |                    |
    |            | ``j`` : [-100, 100]   | ``j`` : [-1, 1]    |
    |            |                       |                    |
    |            | ``g`` : [-100, 100]   | ``g`` : [-1, 1]    |
    +------------+-----------------------+--------------------+

    +------------+-----------------------+--------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**      |
    +============+=======================+====================+
    | ``XYZ``    | [0, 100]              | [0, 1]             |
    +------------+-----------------------+--------------------+

    -   *OSA UCS* uses the *CIE 1964 10 Degree Standard Observer*.

    References
    ----------
    :cite:`Cao2013`, :cite:`Moroney2003`

    Examples
    --------
    >>> import numpy as np
    >>> Ljg = np.array([-3.00499790, 2.99713697, -9.66784231])
    >>> OSA_UCS_to_XYZ(Ljg)  # doctest: +ELLIPSIS
    array([ 20.6540240...,  12.1972369...,   5.1369372...])
    """

    Ljg = to_domain_100(Ljg)
    shape = Ljg.shape
    Ljg = np.atleast_1d(Ljg.reshape([-1, 3]))

    optimisation_settings = {"disp": False}
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    def error_function(XYZ: ArrayLike, Ljg: ArrayLike) -> FloatingOrNDArray:
        """Error function."""

        # Error must be computed in "reference" domain and range.
        with domain_range_scale("ignore"):
            error = np.linalg.norm(XYZ_to_OSA_UCS(XYZ) - as_float_array(Ljg))

        return as_float(error)

    x_0 = np.array([30, 30, 30])
    XYZ = as_float_array(
        [
            fmin(error_function, x_0, (Ljg_i,), **optimisation_settings)
            for Ljg_i in as_float_array(Ljg)
        ]
    )

    return from_range_100(np.reshape(XYZ, shape))
