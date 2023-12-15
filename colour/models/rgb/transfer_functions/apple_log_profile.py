"""
Apple Log Profile Log Encoding
==============================

Defines the *Apple Log Profile* log encoding:

-   :func:`colour.models.log_encoding_AppleLogProfile`
-   :func:`colour.models.log_decoding_AppleLogProfile`

References
----------
-   :cite:`AppleInc.2023` : Apple Inc. (2023). Apple Log Profile White Paper.
    https://developer.apple.com/download/all/?q=Apple%20log%20profile
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import Structure, as_float, from_range_1, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANTS_APPLE_LOG_PROFILE",
    "log_encoding_AppleLogProfile",
    "log_decoding_AppleLogProfile",
]

CONSTANTS_APPLE_LOG_PROFILE: Structure = Structure(
    R_0=-0.05641088,
    R_t=0.01,
    sigma=47.28711236,
    beta=0.00964052,
    gamma=0.08550479,
    delta=0.69336945,
)
"""*Apple Log Profile* constants."""


def log_encoding_AppleLogProfile(
    R: ArrayLike,
    constants: Structure = CONSTANTS_APPLE_LOG_PROFILE,
) -> NDArrayFloat:
    """
    Define the *Apple Log Profile* log encoding curve.

    Parameters
    ----------
    R
        Linear reflection data :math`R`.
    constants
        *Apple Log Profile* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        *Apple Log Profile* captured pixel value :math:`P`

    References
    ----------
    :cite:`AppleInc.2023`

    Notes
    -----
    -   The scene reflection signal :math:`R` captured by the camera is
        represented using a floating point encoding. The :math:`R` value of
        0.18 corresponds to the signal produced by an 18% reflectance;
        reference gray chart.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``R``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``P``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_encoding_AppleLogProfile(0.18)  # doctest: +ELLIPSIS
    0.4882724...
    """

    R = to_domain_1(R)

    R_0 = constants.R_0
    R_t = constants.R_t
    sigma = constants.sigma
    beta = constants.beta
    gamma = constants.gamma
    delta = constants.delta

    P = np.select(
        [
            R >= R_t,  # noqa: SIM300
            np.logical_and(R_0 <= R, R < R_t),  # noqa: SIM300
            R < R_0,
        ],
        [
            gamma * np.log2(R + beta) + delta,
            sigma * (R - R_0) ** 2,
            0,
        ],
    )

    return as_float(from_range_1(P))


def log_decoding_AppleLogProfile(
    P: ArrayLike,
    constants: Structure = CONSTANTS_APPLE_LOG_PROFILE,
) -> NDArrayFloat:
    """
    Define the *Apple Log Profile* log decoding curve.

    Parameters
    ----------
    P
        *Apple Log Profile* captured pixel value :math:`P`
    constants
        *Apple Log Profile* constants.

    Returns
    -------
    :class:`numpy.ndarray`
         Linear reflection data :math`R`.


    References
    ----------
    :cite:`AppleInc.2023`

    Notes
    -----
    -   The captured pixel :math:`P` value is using a floating point encoding
        normalized to the [0, 1] range.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``P``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``R``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_decoding_AppleLogProfile(0.48827245852686763)  # doctest: +ELLIPSIS
    0.1800000...
    """

    P = to_domain_1(P)

    R_0 = constants.R_0
    R_t = constants.R_t
    sigma = constants.sigma
    beta = constants.beta
    gamma = constants.gamma
    delta = constants.delta

    P_t = sigma * (R_t - R_0) ** 2

    R = np.select(
        [
            P >= P_t,  # noqa: SIM300
            np.logical_and(0 <= P, P < P_t),  # noqa: SIM300
            P < 0,
        ],
        [
            2 ** ((P - delta) / gamma) - beta,
            np.sqrt(P / sigma) + R_0,
            R_0,
        ],
    )

    return as_float(from_range_1(R))
