"""
Pivoted Log Encoding
====================

Define the *Pivoted Log* encoding.

-   :func:`colour.models.log_encoding_PivotedLog`
-   :func:`colour.models.log_decoding_PivotedLog`

References
----------
-   :cite:`SonyImageworks2012a` : Sony Imageworks. (2012). make.py. Retrieved
    November 27, 2014, from
    https://github.com/imageworks/OpenColorIO-Configs/blob/master/\
nuke-default/make.py
"""

from __future__ import annotations

import numpy as np

from colour.hints import (  # noqa: TC001
    Domain1,
    Range1,
)
from colour.utilities import as_float, from_range_1, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "log_encoding_PivotedLog",
    "log_decoding_PivotedLog",
]


def log_encoding_PivotedLog(
    x: Domain1,
    log_reference: float = 445,
    linear_reference: float = 0.18,
    negative_gamma: float = 0.6,
    density_per_code_value: float = 0.002,
) -> Range1:
    """
    Apply the *Josh Pines* style *Pivoted Log* log encoding
    opto-electronic transfer function (OETF).

    Parameters
    ----------
    x
        Linear data :math:`x`.
    log_reference
        Log reference that defines the pivot point in code values where
        the logarithmic encoding is centred. Typical value is 445.
    linear_reference
        Linear reference that establishes the relationship between linear
        scene-referred values and the logarithmic code values. Typical
        value is 0.18, representing 18% grey.
    negative_gamma
        Negative gamma that controls the slope and curvature of the
        logarithmic portion of the encoding curve. Lower values produce
        steeper curves with more contrast in the shadows.
    density_per_code_value
        Density per code value that determines the logarithmic step size
        and affects the overall contrast and dynamic range of the encoded
        values.

    Returns
    -------
    :class:`numpy.ndarray`
        Logarithmically encoded data :math:`y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_encoding_PivotedLog(0.18)  # doctest: +ELLIPSIS
    0.4349951...
    """

    x = to_domain_1(x)

    y = (
        log_reference
        + np.log10(x / linear_reference) / (density_per_code_value / negative_gamma)
    ) / 1023

    return as_float(from_range_1(y))


def log_decoding_PivotedLog(
    y: Domain1,
    log_reference: float = 445,
    linear_reference: float = 0.18,
    negative_gamma: float = 0.6,
    density_per_code_value: float = 0.002,
) -> Range1:
    """
    Apply the *Josh Pines* style *Pivoted Log* log decoding inverse
    opto-electronic transfer function (OETF).

    Parameters
    ----------
    y
        Logarithmically encoded data :math:`y`.
    log_reference
        Log reference that defines the pivot point in code values where
        the logarithmic encoding is centred. Typical value is 445.
    linear_reference
        Linear reference that establishes the relationship between linear
        scene-referred values and the logarithmic code values. Typical
        value is 0.18, representing 18% grey.
    negative_gamma
        Negative gamma that controls the slope and curvature of the
        logarithmic portion of the encoding curve. Lower values produce
        steeper curves with more contrast in the shadows.
    density_per_code_value
        Density per code value that determines the logarithmic step size
        and affects the overall contrast and dynamic range of the encoded
        values.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_decoding_PivotedLog(0.434995112414467)  # doctest: +ELLIPSIS
    0.1...
    """

    y = to_domain_1(y)

    x = (
        10 ** ((y * 1023 - log_reference) * (density_per_code_value / negative_gamma))
        * linear_reference
    )

    return as_float(from_range_1(x))
