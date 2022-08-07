"""
SMPTE ST 428-1 (2019)
=====================

Defines *SMPTE ST 428-1 (2019)* inverse electro-optical transfer function (EOTF).

Note that this function uses the definition from :cite:`ITU2021` since SMPTE ST
428-1 is not publicly accessible.

References
----------
-   :cite:`ITU2021` : International Telecommunication Union. (2021).
    Recommendation ITU-T H.273 - Coding-independent code points for video
    signal type identification.
    https://www.itu.int/rec/T-REC-H.273-202107-I/en
"""

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
    "eotf_ST428_1",
    "eotf_inverse_ST428_1",
]


def eotf_ST428_1(V):
    """
    Define the *SMPTE ST 428-1 (2019)* electro-optical transfer function (EOTF).

    Parameters
    ----------
    V
        Electrical signal :math:`V`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding output display *Luminance* :math:`Lo` of the image.

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
    | ``Lo``     | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    -   :cite:`ITU2021`

    Examples
    --------
    >>> eotf_ST428_1(0.5000483377172)  # doctest: +ELLIPSIS
    0.179999999...
    """

    V = to_domain_1(V)

    Lo = 52.37 * spow(V, 2.6) / 48

    return as_float(from_range_1(Lo))


def eotf_inverse_ST428_1(Lo):
    """
    Define the *SMPTE ST 428-1 (2019)* inverse electro-optical transfer function (EOTF).

    Parameters
    ----------
    Lo
        Output display *Luminance* :math:`Lo` of the image.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding electrical signal :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Lo``     | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    -   :cite:`ITU2021`

    Examples
    --------
    >>> eotf_inverse_ST428_1(0.18)  # doctest: +ELLIPSIS
    0.5000483377172...
    """

    Lo = to_domain_1(Lo)

    V = spow(48 * Lo / 52.37, 1 / 2.6)

    return as_float(from_range_1(V))
