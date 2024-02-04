"""
Recommendation ITU-R BT.1361
============================

Defines the *Recommendation ITU-R BT.1361* opto-electrical transfer function
(OETF) and its inverse:

-   :func:`colour.models.oetf_BT1361`
-   :func:`colour.models.oetf_inverse_BT1361`

References
----------
-   :cite:`InternationalTelecommunicationUnion1998` : International
    Telecommunication Union. (1998). Recommendation ITU-R BT.1361 - Worldwide
    unified colorimetry and related characteristics of future television and
    imaging systems (pp. 1-32). https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.1361-0-199802-W!!PDF-E.pdf
"""

import numpy as np

from colour.algebra import spow
from colour.models.rgb.transfer_functions import oetf_BT709, oetf_inverse_BT709
from colour.utilities import (
    as_float,
    domain_range_scale,
    from_range_1,
    to_domain_1,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "oetf_BT1361",
    "oetf_inverse_BT1361",
]


def oetf_BT1361(L):
    """
    Define *Recommendation ITU-R BT.1361* extended color gamut system
    opto-electronic transfer function (OETF).

    Parameters
    ----------
    L
        Scene *Luminance* :math:`L`.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding non-linear primary signal :math:`E'`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+-------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**     |
    +============+=======================+===================+
    | ``E_p'``   | [0, 1]                | [0, 1]            |
    +------------+-----------------------+-------------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion1998`

    Examples
    --------
    >>> oetf_BT1361(0.18)  # doctest: +ELLIPSIS
    0.4090077288641...
    >>> oetf_BT1361(-0.25)  # doctest: +ELLIPSIS
    -0.25
    >>> oetf_BT1361(1.33)  # doctest: +ELLIPSIS
    1.1504846663972...
    """

    L = to_domain_1(L)

    with domain_range_scale("ignore"):
        E_p = np.where(
            L >= 0,
            oetf_BT709(L),
            np.where(
                L <= -0.0045,
                # L in [-0.25, -0.0045] range
                -(1.099 * spow(-4 * L, 0.45) - 0.099) / 4,
                # L in [-0.0045, 0] range
                4.500 * L,
            ),
        )

    return as_float(from_range_1(E_p))


def oetf_inverse_BT1361(E_p):
    """
    Define *Recommendation ITU-R BT.1361* extended color gamut system inverse
    opto-electronic transfer functions (OETF).

    Parameters
    ----------
    E_p
        Non-linear primary signal :math:`E'`.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding scene *Luminance* :math:`L`.

    Notes
    -----
    +------------+-----------------------+-------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**     |
    +============+=======================+===================+
    | ``E_p``    | [0, 1]                | [0, 1]            |
    +------------+-----------------------+-------------------+

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+


    References
    ----------
    :cite:`InternationalTelecommunicationUnion1998`

    Examples
    --------
    >>> oetf_inverse_BT1361(0.4090077288641)  # doctest: +ELLIPSIS
    0.1799999...
    >>> oetf_inverse_BT1361(-0.25)  # doctest: +ELLIPSIS
    -0.25
    >>> oetf_inverse_BT1361(1.1504846663972)  # doctest: +ELLIPSIS
    1.3299999...
    """

    E_p = to_domain_1(E_p)

    with domain_range_scale("ignore"):
        L = np.where(
            E_p >= 0,
            oetf_inverse_BT709(E_p),
            np.where(
                E_p <= 4.500 * -0.0045,
                # L in [-0.25, -0.0045] range
                -spow((-4 * E_p + 0.099) / 1.099, 1 / 0.45) / 4,
                # L in [-0.0045, 0] range
                E_p / 4.500,
            ),
        )

    return as_float(from_range_1(L))
