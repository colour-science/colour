"""
SMPTE ST 2084:2014
==================

Defines the *SMPTE ST 2084:2014* electro-optical transfer function (EOTF) and
its inverse:

-   :func:`colour.models.eotf_inverse_ST2084`
-   :func:`colour.models.eotf_ST2084`

References
----------
-   :cite:`Miller2014a` : Miller, S. (2014). A Perceptual EOTF for Extended
    Dynamic Range Imagery (pp. 1-17).
    https://www.smpte.org/sites/default/files/\
2014-05-06-EOTF-Miller-1-2-handout.pdf
-   :cite:`SocietyofMotionPictureandTelevisionEngineers2014a` : Society of
    Motion Picture and Television Engineers. (2014). SMPTE ST 2084:2014 -
    Dynamic Range Electro-Optical Transfer Function of Mastering Reference
    Displays (pp. 1-14). doi:10.5594/SMPTE.ST2084.2014
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow
from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import Structure, as_float, as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANTS_ST2084",
    "eotf_inverse_ST2084",
    "eotf_ST2084",
]

CONSTANTS_ST2084: Structure = Structure(
    m_1=2610 / 4096 * (1 / 4),
    m_2=2523 / 4096 * 128,
    c_1=3424 / 4096,
    c_2=2413 / 4096 * 32,
    c_3=2392 / 4096 * 32,
)
"""
Constants for *SMPTE ST 2084:2014* inverse electro-optical transfer function
(EOTF) and electro-optical transfer function (EOTF).
"""


def eotf_inverse_ST2084(
    C: ArrayLike,
    L_p: float = 10000,
    constants: Structure = CONSTANTS_ST2084,
) -> NDArrayFloat:
    """
    Define *SMPTE ST 2084:2014* optimised perceptual inverse electro-optical
    transfer function (EOTF).

    Parameters
    ----------
    C
        Target optical output :math:`C` in :math:`cd/m^2` of the ideal
        reference display.
    L_p
        System peak luminance :math:`cd/m^2`, this parameter should stay at its
        default :math:`10000 cd/m^2` value for practical applications. It is
        exposed so that the definition can be used as a fitting function.
    constants
        *SMPTE ST 2084:2014* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Color value abbreviated as :math:`N`, that is directly proportional to
        the encoded signal representation, and which is not directly
        proportional to the optical output of a display device.

    Warnings
    --------
    *SMPTE ST 2084:2014* is an absolute transfer function.

    Notes
    -----
    -   *SMPTE ST 2084:2014* is an absolute transfer function, thus the
        domain and range values for the *Reference* and *1* scales are only
        indicative that the data is not affected by scale transformations.
        The effective domain of *SMPTE ST 2084:2014* inverse electro-optical
        transfer function (EOTF) is [0.0001, 10000].

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``C``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``N``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Miller2014a`,
    :cite:`SocietyofMotionPictureandTelevisionEngineers2014a`

    Examples
    --------
    >>> eotf_inverse_ST2084(100)  # doctest: +ELLIPSIS
    0.5080784...
    """

    C = as_float_array(C)

    c_1 = constants.c_1
    c_2 = constants.c_2
    c_3 = constants.c_3
    m_1 = constants.m_1
    m_2 = constants.m_2

    Y_p = spow(C / L_p, m_1)

    N = spow((c_1 + c_2 * Y_p) / (c_3 * Y_p + 1), m_2)

    return as_float(N)


def eotf_ST2084(
    N: ArrayLike,
    L_p: float = 10000,
    constants: Structure = CONSTANTS_ST2084,
) -> NDArrayFloat:
    """
    Define *SMPTE ST 2084:2014* optimised perceptual electro-optical transfer
    function (EOTF).

    This perceptual quantizer (PQ) has been modeled by Dolby Laboratories
    using *Barten (1999)* contrast sensitivity function.

    Parameters
    ----------
    N
        Color value abbreviated as :math:`N`, that is directly proportional to
        the encoded signal representation, and which is not directly
        proportional to the optical output of a display device.
    L_p
        System peak luminance :math:`cd/m^2`, this parameter should stay at its
        default :math:`10000 cd/m^2` value for practical applications. It is
        exposed so that the definition can be used as a fitting function.
    constants
        *SMPTE ST 2084:2014* constants.

    Returns
    -------
    :class:`numpy.ndarray`
          Target optical output :math:`C` in :math:`cd/m^2` of the ideal
          reference display.

    Warnings
    --------
    *SMPTE ST 2084:2014* is an absolute transfer function.

    Notes
    -----
    -   *SMPTE ST 2084:2014* is an absolute transfer function, thus the
        domain and range values for the *Reference* and *1* scales are only
        indicative that the data is not affected by scale transformations.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``N``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``C``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Miller2014a`,
    :cite:`SocietyofMotionPictureandTelevisionEngineers2014a`

    Examples
    --------
    >>> eotf_ST2084(0.508078421517399)  # doctest: +ELLIPSIS
    100.0000000...
    """

    N = as_float_array(N)

    c_1 = constants.c_1
    c_2 = constants.c_2
    c_3 = constants.c_3
    m_1 = constants.m_1
    m_2 = constants.m_2

    m_1_d = 1 / m_1
    m_2_d = 1 / m_2

    V_p = spow(N, m_2_d)
    n = np.maximum(0, V_p - c_1)
    L = spow((n / (c_2 - c_3 * V_p)), m_1_d)
    C = L_p * L

    return as_float(C)
