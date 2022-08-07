"""
IEC 61966-2
===========

Define transfer functions from *IEC 61966-2* (sRGB, sYCC, xvYCC).

Note that this function uses the definition from :cite:`ITU2021` since IEC
61966-2 is not publicly accessible.

References
----------
-   :cite:`ITU2021` : International Telecommunication Union. (2021).
    Recommendation ITU-T H.273 - Coding-independent code points for video
    signal type identification.
    https://www.itu.int/rec/T-REC-H.273-202107-I/en
"""

import numpy as np

from colour.models.rgb.transfer_functions.srgb import eotf_inverse_sRGB
from colour.models.rgb.transfer_functions.srgb import eotf_sRGB

from colour.utilities import (
    as_float,
    from_range_1,
    to_domain_1,
)


def oetf_iec_61966_2_unbounded(Lc):
    """
    Define the unbounded opto-electronic transfer functions (OETF) for *IEC 61966-2*
    family of transfer functions (*2-1 sRGB*, *2-1 sYCC*, *2-4 xvYCC*).

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
    Usage in :cite:`ITU2021` is as follows:

    - For IEC 61966-2-1 sRGB (MatrixCoefficients=0), function is only defined
      for Lc in [0-1] range.

    - For IEC 61966-2-1 sYCC (MatrixCoefficients=5) and IEC 61966-2-4 xvYCC,
      function is defined for any real-valued Lc.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Lc``     | [-inf, inf]           | [-inf, inf]   |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [-inf, inf]           | [-inf, inf]   |
    +------------+-----------------------+---------------+

    References
    ----------
    -   :cite:`ITU2021`

    Examples
    --------
    >>> oetf_iec_61966_2_unbounded(0.18)  # doctest: +ELLIPSIS
    0.4613561295004...
    >>> oetf_iec_61966_2_unbounded(-0.18)  # doctest: +ELLIPSIS
    -0.4613561295004...
    """

    Lc = to_domain_1(Lc)

    V = np.where(
        Lc >= 0,
        eotf_inverse_sRGB(Lc),
        -eotf_inverse_sRGB(-Lc),
    )

    return as_float(from_range_1(V))


def oetf_inverse_iec_61966_2_unbounded(V):
    """
    Define the unbounded inverse-opto-electronic transfer functions (OETF) for
    *IEC 61966-2* family of transfer functions (*2-1 sRGB*, *2-1 sYCC*, *2-4
    xvYCC*).

    Parameters
    ----------
    V
        Electrical signal :math:`V`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding scene luminance :math:`Lc`.

    Notes
    -----
    Usage in :cite:`ITU2021` is as follows:

    - For IEC 61966-2-1 sRGB (MatrixCoefficients=0), function is only defined
      for Lc in [0-1] range.

    - For IEC 61966-2-1 sYCC (MatrixCoefficients=5) and IEC 61966-2-4 xvYCC,
      function is defined for any real-valued Lc.

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [-inf, inf]           | [-inf, inf]   |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Lc``     | [-inf, inf]           | [-inf, inf]   |
    +------------+-----------------------+---------------+

    References
    ----------
    -   :cite:`ITU2021`

    Examples
    --------
    >>> oetf_inverse_iec_61966_2_unbounded(0.461356129500)  # doctest: +ELLIPSIS
    0.1799999999...
    >>> oetf_inverse_iec_61966_2_unbounded(-0.461356129500)  # doctest: +ELLIPSIS
    -0.1799999999...
    """

    V = to_domain_1(V)

    Lc = np.where(
        V >= 0,
        eotf_sRGB(V),
        -eotf_sRGB(-V),
    )

    return as_float(from_range_1(Lc))
