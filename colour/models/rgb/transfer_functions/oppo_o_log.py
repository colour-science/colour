"""
OPPO O-Log Profile Log Encoding
================================

Define the *OPPO O-Log Profile* log encoding.

-   :func:`colour.models.log_encoding_OPPOOLog`
-   :func:`colour.models.log_decoding_OPPOOLog`

References
----------
-   :cite:`OPPO2025` : OPPO. (2025). OPPO O-Log Profile White Paper. May 2025.
"""

from __future__ import annotations

import numpy as np

from colour.hints import (  # noqa: TC001
    Domain1,
    Range1,
)
from colour.utilities import Structure, as_float, from_range_1, optional, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANTS_OPPO_O_LOG",
    "log_encoding_OPPOOLog",
    "log_decoding_OPPOOLog",
]

CONSTANTS_OPPO_O_LOG: Structure = Structure(
    gamma=0.139,
    beta=0.019,
    delta=0.614,
)
"""*OPPO O-Log Profile* constants."""


def log_encoding_OPPOOLog(
    R: Domain1,
    constants: Structure | None = None,
) -> Range1:
    """
    Apply the *OPPO O-Log Profile* log encoding opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    R
        Linear reflection data :math:`R`.
    constants
        *OPPO O-Log Profile* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Logarithmically encoded value :math:`P`.

    References
    ----------
    :cite:`OPPO2025`

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
    >>> log_encoding_OPPOOLog(0.18)  # doctest: +ELLIPSIS
    np.float64(0.3895913...)
    """

    R = to_domain_1(R)
    constants = optional(constants, CONSTANTS_OPPO_O_LOG)

    gamma = constants.gamma
    beta = constants.beta
    delta = constants.delta

    P = gamma * np.log(R + beta) + delta

    return as_float(from_range_1(P))


def log_decoding_OPPOOLog(
    P: Domain1,
    constants: Structure | None = None,
) -> Range1:
    """
    Apply the *OPPO O-Log Profile* log decoding inverse opto-electronic
    transfer function (OETF).

    Parameters
    ----------
    P
        Logarithmically encoded value :math:`P`.
    constants
        *OPPO O-Log Profile* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear reflection data :math:`R`.

    References
    ----------
    :cite:`OPPO2025`

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
    >>> log_decoding_OPPOOLog(0.38959139)  # doctest: +ELLIPSIS
    np.float64(0.1800000...)
    """

    P = to_domain_1(P)
    constants = optional(constants, CONSTANTS_OPPO_O_LOG)

    gamma = constants.gamma
    beta = constants.beta
    delta = constants.delta

    R = np.exp((P - delta) / gamma) - beta

    return as_float(from_range_1(R))
