"""
ITU-T H.273 video transfer functions
====================================

Contains several transfer functions that are defined in ITU-T H.273
(:cite:`ITU2021`) but don't belong in another specification or standard.

References
----------
-   :cite:`ITU2021` : International Telecommunication Union. (2021). Recommendation
    ITU-T H.273 - Coding-independent code points for video signal type identification.
    https://www.itu.int/rec/T-REC-H.273-202107-I/en
"""

import numpy as np

from colour.utilities import (
    as_float,
    from_range_1,
    to_domain_1,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "oetf_linear",
    "oetf_log",
    "oetf_log_sqrt",
]


def oetf_linear(Lc):
    """
    Linear opto-electronic transfer function (OETF) defined in ITU-T H.273.

    Parameters
    ----------
    Lc
        Scene *Luminance* :math:`Lc`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding electrical signal :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Lc``     | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ITU2021`

    Examples
    --------
    >>> oetf_linear(0.18)  # doctest: +ELLIPSIS
    0.17999999...
    """

    Lc = to_domain_1(Lc)

    V = Lc

    return as_float(from_range_1(V))


def oetf_log(Lc):
    """
    Define the opto-electronic transfer functions (OETF) for Logarithmic
    encoding (100:1 range) defined in ITU-T H.273.

    Parameters
    ----------
    Lc
        Scene *Luminance* :math:`Lc`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding electrical signal :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Lc``     | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ITU2021`

    Examples
    --------
    >>> oetf_log(0.18)  # doctest: +ELLIPSIS
    0.6276362525516...
    """

    Lc = to_domain_1(Lc)

    V = np.where(
        Lc >= 0.01,
        # Lc in [0.01, 1] range
        1.0 + np.log10(Lc) / 2.0,
        # Lc in [0, 0.01] range
        0.0,
    )

    return as_float(from_range_1(V))


def oetf_log_sqrt(Lc):
    """
    Define the opto-electronic transfer functions (OETF) for Logarithmic
    encoding (100*Sqrt(10):1 range) defined in ITU-T H.273.

    Parameters
    ----------
    Lc
        Scene *Luminance* :math:`Lc`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding electrical signal :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Lc``     | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ITU2021`

    Examples
    --------
    >>> oetf_log_sqrt(0.18)  # doctest: +ELLIPSIS
    0.702109002041...
    """

    Lc = to_domain_1(Lc)

    V = np.where(
        Lc >= np.sqrt(10) / 1000,
        # Lc in [0.01, 1] range
        1.0 + np.log10(Lc) / 2.5,
        # Lc in [0, 0.01] range
        0.0,
    )

    return as_float(from_range_1(V))
