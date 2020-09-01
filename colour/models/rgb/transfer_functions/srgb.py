# -*- coding: utf-8 -*-
"""
sRGB
====

Defines the *sRGB* electro-optical transfer function (EOTF / EOCF) and its
inverse:

-   :func:`colour.models.eotf_inverse_sRGB`
-   :func:`colour.models.eotf_sRGB`

References
----------
-   :cite:`InternationalElectrotechnicalCommission1999a` : International
    Electrotechnical Commission. (1999). IEC 61966-2-1:1999 - Multimedia
    systems and equipment - Colour measurement and management - Part 2-1:
    Colour management - Default RGB colour space - sRGB (p. 51).
    https://webstore.iec.ch/publication/6169
-   :cite:`InternationalTelecommunicationUnion2015i` : International
    Telecommunication Union. (2015). Recommendation ITU-R BT.709-6 - Parameter
    values for the HDTV standards for production and international programme
    exchange BT Series Broadcasting service (pp. 1-32).
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.709-6-201506-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import colour.ndarray as np

from colour.algebra import spow
from colour.utilities import (as_float, domain_range_scale, from_range_1,
                              to_domain_1)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['eotf_inverse_sRGB', 'eotf_sRGB']


def eotf_inverse_sRGB(L):
    """
    Defines the *IEC 61966-2-1:1999* *sRGB* inverse electro-optical transfer
    function (EOTF / EOCF).

    Parameters
    ----------
    L : numeric or array_like
        *Luminance* :math:`L` of the image.

    Returns
    -------
    numeric or ndarray
        Corresponding electrical signal :math:`V`.

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
    :cite:`InternationalElectrotechnicalCommission1999a`,
    :cite:`InternationalTelecommunicationUnion2015i`

    Examples
    --------
    >>> eotf_inverse_sRGB(0.18)  # doctest: +ELLIPSIS
    0.4613561...
    """

    L = to_domain_1(L)

    V = np.where(L <= 0.0031308, L * 12.92, 1.055 * spow(L, 1 / 2.4) - 0.055)

    return as_float(from_range_1(V))


def eotf_sRGB(V):
    """
    Defines the *IEC 61966-2-1:1999* *sRGB* electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    V : numeric or array_like
        Electrical signal :math:`V`.

    Returns
    -------
    numeric or ndarray
        Corresponding *luminance* :math:`L` of the image.

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
    :cite:`InternationalElectrotechnicalCommission1999a`,
    :cite:`InternationalTelecommunicationUnion2015i`

    Examples
    --------
    >>> eotf_sRGB(0.461356129500442)  # doctest: +ELLIPSIS
    0.1...
    """

    V = to_domain_1(V)

    with domain_range_scale('ignore'):
        L = np.where(
            V <= eotf_inverse_sRGB(0.0031308),
            V / 12.92,
            spow((V + 0.055) / 1.055, 2.4),
        )

    return as_float(from_range_1(L))
