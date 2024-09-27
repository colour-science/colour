"""
Recommendation ITU-R BT.1886
============================

Define the *Recommendation ITU-R BT.1886* electro-optical transfer function
(EOTF) and its inverse:

-   :func:`colour.models.eotf_inverse_BT1886`
-   :func:`colour.models.eotf_BT1886`

References
----------
-   :cite:`InternationalTelecommunicationUnion2011h` : International
    Telecommunication Union. (2011). Recommendation ITU-R BT.1886 - Reference
    electro-optical transfer function for flat panel displays used in HDTV
    studio production BT Series Broadcasting service.
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.1886-0-201103-I!!PDF-E.pdf
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow
from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import as_float, from_range_1, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "eotf_inverse_BT1886",
    "eotf_BT1886",
]


def eotf_inverse_BT1886(L: ArrayLike, L_B: float = 0, L_W: float = 1) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.1886* inverse electro-optical transfer
    function (EOTF).

    Parameters
    ----------
    L
        Screen luminance in :math:`cd/m^2`.
    L_B
        Screen luminance for black.
    L_W
        Screen luminance for white.

    Returns
    -------
    :class:`numpy.ndarray`
        Input video signal level (normalised, black at :math:`V = 0`, to white
        at :math:`V = 1`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2011h`

    Examples
    --------
    >>> eotf_inverse_BT1886(0.11699185725296059)  # doctest: +ELLIPSIS
    0.4090077...
    """

    L = to_domain_1(L)

    gamma = 2.40
    gamma_d = 1 / gamma

    n = L_W**gamma_d - L_B**gamma_d
    a = n**gamma
    b = L_B**gamma_d / n

    V = spow(L / a, gamma_d) - b

    return as_float(from_range_1(V))


def eotf_BT1886(V: ArrayLike, L_B: float = 0, L_W: float = 1) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.1886* electro-optical transfer function
    (EOTF).

    Parameters
    ----------
    V
        Input video signal level (normalised, black at :math:`V = 0`, to white
        at :math:`V = 1`. For content mastered per
        *Recommendation ITU-R BT.709*, 10-bit digital code values :math:`D` map
        into values of :math:`V` per the following equation:
        :math:`V = (D-64)/876`
    L_B
        Screen luminance for black.
    L_W
        Screen luminance for white.

    Returns
    -------
    :class:`numpy.ndarray`
        Screen luminance in :math:`cd/m^2`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2011h`

    Examples
    --------
    >>> eotf_BT1886(0.409007728864150)  # doctest: +ELLIPSIS
    0.1169918...
    """

    V = to_domain_1(V)

    gamma = 2.40
    gamma_d = 1 / gamma

    n = L_W**gamma_d - L_B**gamma_d
    a = n**gamma
    b = L_B**gamma_d / n
    L = a * spow(np.maximum(V + b, 0), gamma)

    return as_float(from_range_1(L))
