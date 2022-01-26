# -*- coding: utf-8 -*-
"""
ITU-R BT.709-6
==============

Defines the *ITU-R BT.709-6* opto-electrical transfer function (OETF)
and its inverse:

-   :func:`colour.models.oetf_BT709`
-   :func:`colour.models.oetf_inverse_BT709`

References
----------
-   :cite:`InternationalTelecommunicationUnion2015i` : International
    Telecommunication Union. (2015). Recommendation ITU-R BT.709-6 - Parameter
    values for the HDTV standards for production and international programme
    exchange BT Series Broadcasting service (pp. 1-32).
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.709-6-201506-I!!PDF-E.pdf
"""

from __future__ import annotations

from colour.hints import FloatingOrArrayLike, FloatingOrNDArray
from colour.models.rgb.transfer_functions import oetf_BT601, oetf_inverse_BT601

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'oetf_BT709',
    'oetf_inverse_BT709',
]


def oetf_BT709(L: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines *Recommendation ITU-R BT.709-6* opto-electronic transfer function
    (OETF).

    Parameters
    ----------
    L
        *Luminance* :math:`L` of the image.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
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
    :cite:`InternationalTelecommunicationUnion2015i`

    Examples
    --------
    >>> oetf_BT709(0.18)  # doctest: +ELLIPSIS
    0.4090077...
    """

    return oetf_BT601(L)


def oetf_inverse_BT709(V: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines *Recommendation ITU-R BT.709-6* inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    V
        Electrical signal :math:`V`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
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
    :cite:`InternationalTelecommunicationUnion2015i`

    Examples
    --------
    >>> oetf_inverse_BT709(0.409007728864150)  # doctest: +ELLIPSIS
    0.1...
    """

    return oetf_inverse_BT601(V)
