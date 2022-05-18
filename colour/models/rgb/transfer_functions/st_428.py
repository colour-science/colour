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
    | ``L``      | [0, 1]                | [0, 1]        |
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
