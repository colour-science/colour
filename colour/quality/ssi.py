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
    from colour.hints import ModuleType, NDArrayFloat, ProtocolArrayNamespace

from colour.utilities import (
    array_namespace,
    usage_warning,
    xp_as_float_array,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SPECTRAL_SHAPE_SSI",
    "matrix_integration_SSI",
    "spectral_similarity_index",
]

SPECTRAL_SHAPE_SSI: SpectralShape = SpectralShape(375, 675, 1)
"""*Academy Spectral Similarity Index* (SSI) spectral shape."""

_SPECTRAL_SHAPE_SSI_LARGE: SpectralShape = SpectralShape(380, 670, 10)

_MATRIX_INTEGRATION: NDArrayFloat | None = None

_CONVOLUTION_KERNEL_SSI: tuple[float, float, float] = (0.22, 0.56, 0.22)
"""*SSI* three-tap smoothing kernel."""


def _convolve_wdr_i(wdr_i: NDArrayFloat) -> NDArrayFloat:
    """Convolve weighted spectral differences along their spectral axis."""

    xp = array_namespace(wdr_i)
    padding = xp.zeros_like(wdr_i[:1])
    left, centre, right = _CONVOLUTION_KERNEL_SSI

    return (
        left * xp.concat([padding, wdr_i[:-1]], axis=0)
        + centre * wdr_i
        + right * xp.concat([wdr_i[1:], padding], axis=0)
    )


def _is_gradient_tracked(a: object) -> bool:
    """Return whether the specified backend array tracks differentiation."""

    if bool(getattr(a, "requires_grad", False)):
        return True

    namespace = getattr(a, "__array_namespace__", None)
    if namespace is None or namespace().__name__ != "jax.numpy":
        return False

    from jax.core import Tracer  # noqa: PLC0415

    return isinstance(a, Tracer)


def _round_SSI(SSI: NDArrayFloat, round_result: bool) -> NDArrayFloat:
    """Round the *SSI* while warning about unusable gradients."""

    if not round_result:
        return SSI

    if _is_gradient_tracked(SSI):
        usage_warning(
            '"round_result=True" produces zero gradients almost everywhere; '
            "set it to False when using SSI with automatic differentiation."
        )

    return array_namespace(SSI).round(SSI)


def matrix_integration_SSI(
    *, xp: ProtocolArrayNamespace | ModuleType = np
) -> NDArrayFloat:
    """
    Build the *SSI* sparse integration matrix in the specified namespace.

    The matrix maps the 1 nm *SSI* working shape to the 10 nm reference
    bands by convolving each band with a unit-area triangular kernel. It
    is cached at module level via :data:`_MATRIX_INTEGRATION`; callers
    promote it to the per-input backend at use time via
    :func:`xp_as_float_array`.
    """

    n_rows = len(_SPECTRAL_SHAPE_SSI_LARGE.wavelengths)
    n_cols = len(SPECTRAL_SHAPE_SSI.wavelengths)
    weights = xp_as_float_array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5], xp=xp)

    return xp.concat(
        [
            xp_reshape(
                xp.concat(
                    [
                        xp_as_float_array(xp.zeros(10 * i), xp=xp, like=weights),
                        weights,
                        xp_as_float_array(
                            xp.zeros(max(0, n_cols - 10 * i - 11)),
                            xp=xp,
                            like=weights,
                        ),
                    ]
                )[:n_cols],
                (1, -1),
                xp=xp,
            )
            for i in range(n_rows)
        ],
        axis=0,
    )


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
        Whether to round the result to the nearest integer. Disable rounding to
        retain a continuous objective and useful gradients for optimisation.
        Default is *True* for the standard reported *SSI* value.

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
    np.float64(94.0)

    Computing SSI for multi-spectral distributions:

    >>> from colour.colorimetry import sd_single_led, sds_and_msds_to_msds
    >>> sd_led_1 = sd_single_led(520, half_spectral_width=45)
    >>> sd_led_2 = sd_single_led(540, half_spectral_width=55)
    >>> sd_led_3 = sd_single_led(560, half_spectral_width=50)
    >>> msds = sds_and_msds_to_msds([sd_led_1, sd_led_2, sd_led_3])
    >>> sd_reference = sd_single_led(535, half_spectral_width=48)
    >>> spectral_similarity_index(msds, sd_reference)
    array([52., 82., 18.])
    """

    global _MATRIX_INTEGRATION  # noqa: PLW0603

    if _MATRIX_INTEGRATION is None:
        _MATRIX_INTEGRATION = matrix_integration_SSI()

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

    xp = array_namespace(sd_test.values, sd_reference.values)

    sd_test_values = xp_as_float_array(sd_test.values, xp=xp)
    sd_reference_values = xp_as_float_array(
        sd_reference.values, xp=xp, like=sd_test_values
    )
    matrix = xp_as_float_array(_MATRIX_INTEGRATION, xp=xp, like=sd_test_values)

    test_i = xp.matmul(matrix, sd_test_values)
    reference_i = xp.matmul(matrix, sd_reference_values)

    if test_i.ndim == 1 and reference_i.ndim == 2:
        test_i = xp.tile(test_i[:, None], (1, reference_i.shape[1]))
    elif test_i.ndim == 2 and reference_i.ndim == 1:
        reference_i = xp.tile(reference_i[:, None], (1, test_i.shape[1]))

    with sdiv_mode():
        test_i = sdiv(test_i, xp.sum(test_i, axis=0, keepdims=True))
        reference_i = sdiv(reference_i, xp.sum(reference_i, axis=0, keepdims=True))
        dr_i = sdiv(test_i - reference_i, reference_i + 1 / 30)

    weights = xp_as_float_array(
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
        ],
        xp=xp,
        like=sd_test_values,
    )

    if dr_i.ndim == 2:
        weights = weights[:, None]

    wdr_i = dr_i * weights
    c_wdr_i = _convolve_wdr_i(wdr_i)
    m_v = xp.sum(xp.square(c_wdr_i), axis=0)

    SSI = 100 - 32 * xp.sqrt(m_v)

    return _round_SSI(SSI, round_result)
