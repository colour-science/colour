"""
ITU-R BT.1361
=============

Define transfer functions from *ITU-R BT.1361*.

References
----------
-   :cite:`InternationalTelecommunicationUnion1998` : International
    Telecommunication Union. (1998). Recommendation ITU-R BT.1361 -
    Worldwide unified colorimetry and related characteristics of
    future television and imaging systems
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.1361-0-199802-W!!PDF-E.pdf
"""
import numpy as np

from colour.algebra import spow
from colour.utilities import (
    as_float_array,
    as_float,
)


L_LINEAR_THRESHOLD_POSITIVE = 0.018
L_LINEAR_THRESHOLD_NEGATIVE = -0.0045


def oetf_BT1361_extended(L):
    """
    Define the opto-electronic transfer functions (OETF) for *ITU-R BT.1361* extended
    color gamut.

    Parameters
    ----------
    L
        Scene *Luminance* :math:`L`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding non-linear primary signal :math:`E'`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [-0.25, 1.33]         | [-0.25, 1.33] |
    +------------+-----------------------+---------------+

    +------------+-----------------------+-------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**     |
    +============+=======================+===================+
    | ``Ep'``    | [-0.25, 1.152...]     | [-0.25, 1.152...] |
    +------------+-----------------------+-------------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion1998`

    Examples
    --------
    >>> oetf_BT1361_extended(0.18)  # doctest: +ELLIPSIS
    0.4090077288641...
    >>> oetf_BT1361_extended(-0.25)  # doctest: +ELLIPSIS
    -0.25
    >>> oetf_BT1361_extended(1.33)  # doctest: +ELLIPSIS
    1.1504846663972...
    """

    L = as_float_array(L)

    Ep = np.where(
        L >= 0,
        np.where(
            L >= L_LINEAR_THRESHOLD_POSITIVE,
            # L in [0.018, 1.33] range
            1.099 * spow(L, 0.45) - 0.099,
            # L in [0, 0.018] range
            4.500 * L,
        ),
        np.where(
            L <= L_LINEAR_THRESHOLD_NEGATIVE,
            # L in [-0.25, -0.0045] range
            -(1.099 * spow(-4 * L, 0.45) - 0.099)
            / 4,
            # L in [-0.0045, 0] range
            4.500 * L,
        ),
    )

    return as_float(Ep)
