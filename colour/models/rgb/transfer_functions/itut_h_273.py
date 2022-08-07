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

from colour.algebra import spow
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
    "oetf_inverse_linear",
    "oetf_inverse_log",
    "oetf_inverse_log_sqrt",
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


def oetf_inverse_linear(V):
    """
    Linear inverse-opto-electronic transfer function (OETF) defined in ITU-T
    H.273.

    Parameters
    ----------
    V
        Electrical signal :math:`V`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding scene *Luminance* :math:`Lc`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Lc``     | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ITU2021`

    Examples
    --------
    >>> oetf_inverse_linear(0.18)  # doctest: +ELLIPSIS
    0.17999999...
    """

    V = to_domain_1(V)

    Lc = V

    return as_float(from_range_1(Lc))


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
    >>> oetf_log(0.01)  # doctest: +ELLIPSIS
    0.0
    >>> oetf_log(0.001)  # doctest: +ELLIPSIS
    0.0
    >>> oetf_log(1.0)  # doctest: +ELLIPSIS
    1.0
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


def oetf_inverse_log(V):
    """
    Define the inverse-opto-electronic transfer functions (OETF) for
    Logarithmic encoding (100:1 range) defined in ITU-T H.273.

    Parameters
    ----------
    V
        Electrical signal :math:`V`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding scene *Luminance* :math:`Lc`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Lc``     | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ITU2021`

    Examples
    --------
    >>> oetf_inverse_log(0.6276362525516)  # doctest: +ELLIPSIS
    0.17999999...
    >>> oetf_inverse_log(0.0)  # doctest: +ELLIPSIS
    0.01
    >>> oetf_inverse_log(1.0)  # doctest: +ELLIPSIS
    1.0
    """

    V = to_domain_1(V)

    Lc = np.where(
        V >= 0.0,
        # Lc in [0.01, 1] range
        spow(10, (V-1.0)*2.0),
        # Lc in [0, 0.01] range
        0.01,
    )

    return as_float(from_range_1(Lc))


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
    >>> oetf_log_sqrt(0.003162277660168)  # doctest: +ELLIPSIS
    0.0
    >>> oetf_log_sqrt(0.0001)  # doctest: +ELLIPSIS
    0.0
    >>> oetf_log_sqrt(1.0)  # doctest: +ELLIPSIS
    1.0
    """

    Lc = to_domain_1(Lc)

    V = np.where(
        Lc >= np.sqrt(10) / 1000,
        # Lc in [sqrt(10)/1000, 1] range
        1.0 + np.log10(Lc) / 2.5,
        # Lc in [0, sqrt(10)/1000] range
        0.0,
    )

    return as_float(from_range_1(V))


def oetf_inverse_log_sqrt(V):
    """
    Define the inverse-opto-electronic transfer functions (OETF) for
    Logarithmic encoding (100*Sqrt(10):1 range) defined in ITU-T H.273.

    Parameters
    ----------
    V
        Electrical signal :math:`V`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding scene *Luminance* :math:`Lc`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Lc``     | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ITU2021`

    Examples
    --------
    >>> oetf_inverse_log_sqrt(0.702109002041)  # doctest: +ELLIPSIS
    0.1799999999...
    >>> oetf_inverse_log_sqrt(0.0)  # doctest: +ELLIPSIS
    0.00316227766...
    >>> oetf_inverse_log_sqrt(1.0)  # doctest: +ELLIPSIS
    1.0
    """

    V = to_domain_1(V)

    Lc = np.where(
        V >= 0.0,
        # Lc in [sqrt(10)/1000, 1] range
        spow(10, (V-1.0)*2.5),
        # Lc in [0, sqrt(10)/1000] range
        np.sqrt(10) / 1000,
    )

    return as_float(from_range_1(Lc))
