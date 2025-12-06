"""
Academy Spectral Similarity Index (SSI)
========================================

Define the *Academy Spectral Similarity Index* (SSI) computation objects.

-   :func:`colour.spectral_similarity_index`

References
----------
-   :cite:`TheAcademyofMotionPictureArtsandSciences2020a` : The Academy of
    Motion Picture Arts and Sciences. (2020). Academy Spectral Similarity
    Index (SSI): Overview (pp. 1-7). Retrieved June 5, 2023, from
    https://www.oscars.org/sites/oscars/files/ssi_overview_2020-09-16.pdf
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import LinearInterpolator, sdiv, sdiv_mode
from colour.colorimetry import (
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    reshape_msds,
    reshape_sd,
)

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayFloat


from colour.utilities import required, zeros

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
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

_MATRIX_INTEGRATION: NDArrayFloat | None = None


@required("SciPy")
def spectral_similarity_index(
    sd_test: SpectralDistribution | MultiSpectralDistributions,
    sd_reference: SpectralDistribution | MultiSpectralDistributions,
    round_result: bool = True,
) -> NDArrayFloat:
    """
    Compute the *Academy Spectral Similarity Index* (SSI) of the specified
    test spectral distribution or multi-spectral distributions with the
    specified reference spectral distribution or multi-spectral distributions.

    Parameters
    ----------
    sd_test
        Test spectral distribution or multi-spectral distributions.
    sd_reference
        Reference spectral distribution or multi-spectral distributions.
    round_result
        Whether to round the result/output. This is particularly useful when
        using SSI in an optimisation routine. Default is *True*.

    Returns
    -------
    :class:`numpy.ndarray`
        *Academy Spectral Similarity Index* (SSI). When both inputs are
        :class:`colour.SpectralDistribution` objects, returns a scalar.
        When either input is a :class:`colour.MultiSpectralDistributions`
        object, returns an array with one SSI value per spectral distribution.

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2020a`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd_test = SDS_ILLUMINANTS["C"]
    >>> sd_reference = SDS_ILLUMINANTS["D65"]
    >>> spectral_similarity_index(sd_test, sd_reference)
    94.0

    Computing SSI for multi-spectral distributions:

    >>> from colour.colorimetry import sd_single_led, sds_and_msds_to_msds
    >>> sd_led_1 = sd_single_led(520, half_spectral_width=45)
    >>> sd_led_2 = sd_single_led(540, half_spectral_width=55)
    >>> sd_led_3 = sd_single_led(560, half_spectral_width=50)
    >>> msds = sds_and_msds_to_msds([sd_led_1, sd_led_2, sd_led_3])
    >>> sd_reference = sd_single_led(535, half_spectral_width=48)
    >>> spectral_similarity_index(msds, sd_reference)
    array([ 52.,  82.,  18.])
    """

    from scipy.ndimage import convolve1d  # noqa: PLC0415

    global _MATRIX_INTEGRATION  # noqa: PLW0603

    if _MATRIX_INTEGRATION is None:
        _MATRIX_INTEGRATION = zeros(
            (
                len(_SPECTRAL_SHAPE_SSI_LARGE.wavelengths),
                len(SPECTRAL_SHAPE_SSI.wavelengths),
            )
        )

        weights = np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5])

        for i in range(_MATRIX_INTEGRATION.shape[0]):
            _MATRIX_INTEGRATION[i, (10 * i) : (10 * i + 11)] = weights

    settings = {
        "interpolator": LinearInterpolator,
        "extrapolator_kwargs": {"left": 0, "right": 0},
    }

    sd_test = (
        reshape_msds(sd_test, SPECTRAL_SHAPE_SSI, "Align", copy=False, **settings)
        if isinstance(sd_test, MultiSpectralDistributions)
        else reshape_sd(sd_test, SPECTRAL_SHAPE_SSI, "Align", copy=False, **settings)
    )
    sd_reference = (
        reshape_msds(sd_reference, SPECTRAL_SHAPE_SSI, "Align", copy=False, **settings)
        if isinstance(sd_reference, MultiSpectralDistributions)
        else reshape_sd(
            sd_reference, SPECTRAL_SHAPE_SSI, "Align", copy=False, **settings
        )
    )

    test_i = np.dot(_MATRIX_INTEGRATION, sd_test.values)
    reference_i = np.dot(_MATRIX_INTEGRATION, sd_reference.values)

    if test_i.ndim == 1 and reference_i.ndim == 2:
        test_i = np.tile(test_i[:, np.newaxis], (1, reference_i.shape[1]))
    elif test_i.ndim == 2 and reference_i.ndim == 1:
        reference_i = np.tile(reference_i[:, np.newaxis], (1, test_i.shape[1]))

    with sdiv_mode():
        test_i = sdiv(test_i, np.sum(test_i, axis=0, keepdims=True))
        reference_i = sdiv(reference_i, np.sum(reference_i, axis=0, keepdims=True))
        dr_i = sdiv(test_i - reference_i, reference_i + 1 / 30)

    weights = np.array(
        [
            4 / 15,
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
    )

    if dr_i.ndim == 2:
        weights = weights[:, np.newaxis]

    wdr_i = dr_i * weights
    c_wdr_i = convolve1d(wdr_i, [0.22, 0.56, 0.22], axis=0, mode="constant", cval=0)
    m_v = np.sum(np.square(c_wdr_i), axis=0)

    SSI = 100 - 32 * np.sqrt(m_v)

    return np.around(SSI) if round_result else SSI
