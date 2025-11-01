"""
Xiaomi Mi-Log Profile Log Encoding
==================================

Define the *Xiaomi Mi-Log Profile* log encoding.

-   :func:`colour.models.log_encoding_MiLog`
-   :func:`colour.models.log_decoding_MiLog`

References
----------
-   :cite:`Zhang2024` : Xiaomi Inc. (2024). Xiaomi Log Profile White Paper.
    December 2024.
"""

from __future__ import annotations

from colour.hints import (  # noqa: TC001
    Domain1,
    Range1,
)
from colour.utilities import Structure, optional

from .apple_log_profile import (
    log_decoding_AppleLogProfile,
    log_encoding_AppleLogProfile,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANTS_MI_LOG",
    "log_encoding_MiLog",
    "log_decoding_MiLog",
]

CONSTANTS_MI_LOG: Structure = Structure(
    R_0=-0.09023729,
    R_t=0.01974185,
    sigma=18.10531998,  # 'c' in whitepaper, 'sigma' for Apple compatibility
    beta=0.01384578,
    gamma=0.09271529,
    delta=0.67291850,
)
"""*Xiaomi Mi-Log Profile* constants."""


def log_encoding_MiLog(
    R: Domain1,
    constants: Structure | None = None,
) -> Range1:
    """
    Apply the *Xiaomi Mi-Log Profile* log encoding opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    R
        Linear reflection data :math:`R`.
    constants
        *Xiaomi Mi-Log Profile* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Logarithmically encoded value :math:`P`.

    References
    ----------
    :cite:`Zhang2024`

    Notes
    -----
    -   The scene reflection signal :math:`R` captured by the camera is
        represented using a floating point encoding. The :math:`R` value
        of 0.18 corresponds to the signal produced by an 18% reflectance
        reference gray chart.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``R``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``P``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_encoding_MiLog(0.18)  # doctest: +ELLIPSIS
    0.4534596...
    """

    return log_encoding_AppleLogProfile(R, optional(constants, CONSTANTS_MI_LOG))


def log_decoding_MiLog(
    P: Domain1,
    constants: Structure | None = None,
) -> Range1:
    """
    Apply the *Xiaomi Mi-Log Profile* log decoding inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    P
        Logarithmically encoded value :math:`P`.
    constants
        *Xiaomi Mi-Log Profile* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear reflection data :math:`R`.

    References
    ----------
    :cite:`Zhang2024`

    Notes
    -----
    -   The captured pixel :math:`P` value uses floating point encoding
        normalized to the [0, 1] range.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``P``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``R``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_decoding_MiLog(0.45345968)  # doctest: +ELLIPSIS
    0.1800000...
    """

    return log_decoding_AppleLogProfile(P, optional(constants, CONSTANTS_MI_LOG))
