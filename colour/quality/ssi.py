"""
Academy Spectral Similarity Index (SSI)
=======================================

Defines the *Academy Spectral Similarity Index* (SSI) computation objects:

-   :func:`colour.spectral_similarity_index`

References
----------
-   :cite:`TheAcademyofMotionPictureArtsandSciences2019` : The Academy of
    Motion Picture Arts and Sciences. (2019). Academy Spectral Similarity Index
    (SSI): Overview (pp. 1-7).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage.filters import convolve1d

from colour.algebra import LinearInterpolator
from colour.colorimetry import SpectralDistribution, SpectralShape, reshape_sd
from colour.hints import NDArray, Optional
from colour.utilities import zeros

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SPECTRAL_SHAPE_SSI",
    "spectral_similarity_index",
]

SPECTRAL_SHAPE_SSI: SpectralShape = SpectralShape(375, 675, 1)
"""*Academy Spectral Similarity Index* (SSI) spectral shape."""

_SPECTRAL_SHAPE_SSI_LARGE: SpectralShape = SpectralShape(380, 670, 10)

_MATRIX_INTEGRATION: Optional[NDArray] = None


def spectral_similarity_index(
    sd_test: SpectralDistribution, sd_reference: SpectralDistribution
) -> NDArray:
    """
    Return the *Academy Spectral Similarity Index* (SSI) of given test
    spectral distribution with given reference spectral distribution.

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    sd_reference
        Reference spectral distribution.

    Returns
    -------
    :class:`numpy.ndarray`
        *Academy Spectral Similarity Index* (SSI).

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2019`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd_test = SDS_ILLUMINANTS['C']
    >>> sd_reference = SDS_ILLUMINANTS['D65']
    >>> spectral_similarity_index(sd_test, sd_reference)
    94.0
    """

    global _MATRIX_INTEGRATION

    if _MATRIX_INTEGRATION is None:
        _MATRIX_INTEGRATION = zeros(
            (
                len(_SPECTRAL_SHAPE_SSI_LARGE.range()),
                len(SPECTRAL_SHAPE_SSI.range()),
            )
        )

        weights = np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5])

        for i in range(_MATRIX_INTEGRATION.shape[0]):
            _MATRIX_INTEGRATION[i, (10 * i) : (10 * i + 11)] = weights

    settings = {
        "interpolator": LinearInterpolator,
        "extrapolator_kwargs": {"left": 0, "right": 0},
    }

    sd_test = reshape_sd(sd_test, SPECTRAL_SHAPE_SSI, "Align", **settings)
    sd_reference = reshape_sd(
        sd_reference, SPECTRAL_SHAPE_SSI, "Align", **settings
    )

    test_i = np.dot(_MATRIX_INTEGRATION, sd_test.values)
    reference_i = np.dot(_MATRIX_INTEGRATION, sd_reference.values)

    test_i /= np.sum(test_i)
    reference_i /= np.sum(reference_i)

    d_i = test_i - reference_i
    dr_i = d_i / (reference_i + np.mean(reference_i))
    wdr_i = dr_i * [
        12 / 45,
        22 / 45,
        32 / 45,
        40 / 45,
        44 / 45,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        11 / 15,
        3 / 15,
    ]
    c_wdr_i = convolve1d(np.hstack([0, wdr_i, 0]), [0.22, 0.56, 0.22])
    m_v = np.sum(c_wdr_i**2)

    SSI = np.around(100 - 32 * np.sqrt(m_v))

    return SSI
