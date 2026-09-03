"""
Tristimulus Values
==================

Define objects for computing CIE tristimulus values from spectral data.

This module provides comprehensive functionality for converting spectral
distributions to CIE XYZ tristimulus values using various integration and
summation methods. The default implementation follows the *ASTM E308-15*
standard practice.

-   :attr:`colour.SPECTRAL_SHAPE_ASTME308`
-   :func:`colour.colorimetry.handle_spectral_arguments`
-   :func:`colour.colorimetry.tristimulus_weighting_factors_ASTME2022`
-   :func:`colour.colorimetry.sd_to_XYZ_integration`
-   :func:`colour.colorimetry.\
sd_to_XYZ_tristimulus_weighting_factors_ASTME308`
-   :func:`colour.colorimetry.sd_to_XYZ_ASTME308`
-   :attr:`colour.SD_TO_XYZ_METHODS`
-   :func:`colour.sd_to_XYZ`
-   :func:`colour.colorimetry.msds_to_XYZ_integration`
-   :func:`colour.colorimetry.\
msds_to_XYZ_tristimulus_weighting_factors_ASTME308`
-   :func:`colour.colorimetry.msds_to_XYZ_ASTME308`
-   :attr:`colour.MSDS_TO_XYZ_METHODS`
-   :func:`colour.msds_to_XYZ`
-   :func:`colour.wavelength_to_XYZ`

References
----------
-   :cite:`ASTMInternational2011a` : ASTM International. (2011). ASTM E2022-11
    - Standard Practice for Calculation of Weighting Factors for Tristimulus
    Integration (pp. 1-10). doi:10.1520/E2022-11
-   :cite:`ASTMInternational2015b` : ASTM International. (2015). ASTM E308-15 -
    Standard Practice for Computing the Colors of Objects by Using the CIE
    System (pp. 1-47). doi:10.1520/E0308-15
-   :cite:`InternationalOrganizationforStandardization2024` : International
    Organization for Standardization. (2024). INTERNATIONAL STANDARD ISO
    18314-4 - Analytical colorimetry Part 4: Metamerism index for pairs of
    samples for change of illuminant. https://www.iso.org/standard/85116.html
-   :cite:`Wyszecki2000bf` : Wyszecki, Günther, & Stiles, W. S. (2000).
    Integration Replaced by Summation. In Color Science: Concepts and Methods,
    Quantitative Data and Formulae (pp. 158-163). Wiley. ISBN:978-0-471-39918-6
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import lagrange_coefficients, sdiv, sdiv_mode
from colour.colorimetry import (
    SPECTRAL_SHAPE_DEFAULT,
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    reshape_msds,
    reshape_sd,
)

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        Literal,
        NDArrayFloat,
        Range1,
        Range100,
        Tuple,
    )

from colour.hints import Real, cast
from colour.utilities import (
    CACHE_REGISTRY,
    CanonicalMapping,
    array_namespace,
    as_float_array,
    as_int_scalar,
    as_ndarray,
    attest,
    filter_kwargs,
    from_range_100,
    get_domain_range_scale,
    int_digest,
    is_caching_enabled,
    optional,
    runtime_warning,
    validate_method,
    xp_as_array,
    xp_as_float_array,
    xp_matrix_transpose,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SPECTRAL_SHAPE_ASTME308",
    "handle_spectral_arguments",
    "lagrange_coefficients_ASTME2022",
    "tristimulus_weighting_factors_ASTME2022",
    "adjust_tristimulus_weighting_factors_ASTME308",
    "tristimulus_weighting_factors_integration",
    "sd_to_XYZ_integration",
    "sd_to_XYZ_tristimulus_weighting_factors_ASTME308",
    "sd_to_XYZ_ASTME308",
    "SD_TO_XYZ_METHODS",
    "sd_to_XYZ",
    "msds_to_XYZ_integration",
    "msds_to_XYZ_tristimulus_weighting_factors_ASTME308",
    "msds_to_XYZ_ASTME308",
    "MSDS_TO_XYZ_METHODS",
    "msds_to_XYZ",
    "wavelength_to_XYZ",
]

SPECTRAL_SHAPE_ASTME308: SpectralShape = SPECTRAL_SHAPE_DEFAULT
SPECTRAL_SHAPE_ASTME308.__doc__ = """
Define the spectral shape for *ASTM E308-15* practice with wavelength range
from 360 to 780 nm at 1 nm intervals: (360, 780, 1).

References
----------
:cite:`ASTMInternational2015b`
"""

_CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS"
)

_CACHE_TRISTIMULUS_WEIGHTING_FACTORS: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_TRISTIMULUS_WEIGHTING_FACTORS"
)

_CACHE_SD_TO_XYZ: dict = CACHE_REGISTRY.register_cache(f"{__name__}._CACHE_SD_TO_XYZ")


def _is_gradient_tracked(a: Any) -> bool:
    """Return whether the specified backend array tracks differentiation."""

    if bool(getattr(a, "requires_grad", False)):
        return True

    namespace = getattr(a, "__array_namespace__", None)
    if namespace is None or namespace().__name__ != "jax.numpy":
        return False

    import jax  # noqa: PLC0415

    return isinstance(a, jax.core.Tracer)


def handle_spectral_arguments(
    cmfs: MultiSpectralDistributions | None = None,
    illuminant: SpectralDistribution | None = None,
    cmfs_default: str = "CIE 1931 2 Degree Standard Observer",
    illuminant_default: str = "D65",
    shape_default: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    issue_runtime_warnings: bool = True,
) -> Tuple[MultiSpectralDistributions, SpectralDistribution]:
    """
    Handle spectral arguments for various *Colour* definitions that perform
    spectral computations.

    -   If ``cmfs`` is not specified, select one according to
        ``cmfs_default``. The returned colour matching functions adopt the
        spectral shape specified by ``shape_default``.
    -   If ``illuminant`` is not specified, select one according to
        ``illuminant_default``. The returned illuminant adopts the spectral
        shape of the returned colour matching functions.
    -   If ``illuminant`` is specified, align the returned illuminant's
        spectral shape to that of the returned colour matching functions.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to
        *CIE Standard Illuminant D65*.
    cmfs_default
        Default colour matching functions to use if ``cmfs`` is not
        specified.
    illuminant_default
        Default illuminant to use if ``illuminant`` is not specified.
    shape_default
        Default spectral shape to align the final colour matching functions
        and illuminant.
    issue_runtime_warnings
        Whether to issue runtime warnings.

    Returns
    -------
    :class:`tuple`
        Colour matching functions and illuminant.

    Examples
    --------
    >>> cmfs, illuminant = handle_spectral_arguments()
    >>> cmfs.name, cmfs.shape, illuminant.name, illuminant.shape
    ('CIE 1931 2 Degree Standard Observer', SpectralShape(360.0, 780.0, 1.0), \
'D65', SpectralShape(360.0, 780.0, 1.0))
    >>> cmfs, illuminant = handle_spectral_arguments(
    ...     shape_default=SpectralShape(400, 700, 20)
    ... )
    >>> cmfs.name, cmfs.shape, illuminant.name, illuminant.shape
    ('CIE 1931 2 Degree Standard Observer', \
SpectralShape(400.0, 700.0, 20.0), 'D65', SpectralShape(400.0, 700.0, 20.0))
    """

    from colour import MSDS_CMFS, SDS_ILLUMINANTS  # noqa: PLC0415

    cmfs = optional(
        cmfs, reshape_msds(MSDS_CMFS[cmfs_default], shape_default, copy=False)
    )
    illuminant = optional(
        illuminant,
        reshape_sd(SDS_ILLUMINANTS[illuminant_default], cmfs.shape, copy=False),
    )

    if illuminant.shape != cmfs.shape:
        issue_runtime_warnings and runtime_warning(
            f'Aligning "{illuminant.name}" illuminant shape to "{cmfs.name}" '
            f"colour matching functions shape."
        )

        illuminant = reshape_sd(illuminant, cmfs.shape, copy=False)

    return cmfs, illuminant


def lagrange_coefficients_ASTME2022(
    interval: int = 10,
    interval_type: Literal["Boundary", "Inner"] | str = "Inner",
) -> NDArrayFloat:
    """
    Compute *Lagrange Coefficients* for the specified interval size using
    practice *ASTM E2022-11* method.

    Parameters
    ----------
    interval
        Interval size in nm.
    interval_type
        If the interval is an *inner* interval, *Lagrange Coefficients* are
        computed for degree 4. Degree 3 is used for a *boundary* interval.

    Returns
    -------
    :class:`numpy.ndarray`
        *Lagrange Coefficients*.

    References
    ----------
    :cite:`ASTMInternational2011a`

    Examples
    --------
    >>> lagrange_coefficients_ASTME2022(10, "inner")
    ... # doctest: +ELLIPSIS
    array([[-0.028...,  0.940...,  0.104..., -0.016...],
           [-0.048...,  0.864...,  0.216..., -0.032...],
           [-0.059...,  0.773...,  0.331..., -0.045...],
           [-0.064...,  0.672...,  0.448..., -0.056...],
           [-0.062...,  0.562...,  0.562..., -0.062...],
           [-0.056...,  0.448...,  0.672..., -0.064...],
           [-0.045...,  0.331...,  0.773..., -0.059...],
           [-0.032...,  0.216...,  0.864..., -0.048...],
           [-0.016...,  0.104...,  0.940..., -0.028...]])
    >>> lagrange_coefficients_ASTME2022(10, "boundary")
    ... # doctest: +ELLIPSIS
    array([[ 0.85...,  0.19..., -0.04...],
           [ 0.72...,  0.36..., -0.08...],
           [ 0.59...,  0.51..., -0.10...],
           [ 0.48...,  0.64..., -0.12...],
           [ 0.37...,  0.75..., -0.12...],
           [ 0.28...,  0.84..., -0.12...],
           [ 0.19...,  0.91..., -0.10...],
           [ 0.12...,  0.96..., -0.08...],
           [ 0.05...,  0.99..., -0.04...]])
    """

    global _CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS  # noqa: PLW0602

    interval_type = validate_method(
        interval_type,
        ("Boundary", "Inner"),
        '"{0}" interval type is invalid, it must be one of {1}!',
    )

    hash_key = hash((interval, interval_type))

    if is_caching_enabled() and hash_key in _CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS:
        # Defensive copy so caller mutations don't poison the cache.
        return _CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS[hash_key].copy()

    r_n = np.linspace(1 / interval, 1 - (1 / interval), interval - 1)
    d = 3
    if interval_type == "inner":
        r_n += 1
        d = 4

    lica = as_float_array([lagrange_coefficients(r, d) for r in r_n])

    if is_caching_enabled():
        _CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS[hash_key] = lica.copy()

    return lica


def tristimulus_weighting_factors_ASTME2022(
    cmfs: MultiSpectralDistributions,
    illuminant: SpectralDistribution,
    shape: SpectralShape,
    k: Real | None = None,
) -> NDArrayFloat:
    """
    Compute a table of tristimulus weighting factors for the specified colour
    matching functions and illuminant using practise *ASTM E2022-11* method.

    The computed table of tristimulus weighting factors should be used with
    spectral data that has been corrected for spectral bandpass dependence.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions.
    illuminant
        Illuminant spectral distribution.
    shape
        Shape used to build the table, only the interval is needed.
    k
        Normalisation constant :math:`k`. For reflecting or transmitting
        object colours, :math:`k` is chosen so that :math:`Y = 100` for
        objects for which the spectral reflectance factor
        :math:`R(\\lambda)` of the object colour or the spectral
        transmittance factor :math:`\\tau(\\lambda)` of the object is equal
        to unity for all wavelengths. For self-luminous objects and
        illuminants, the constants :math:`k` is usually chosen on the
        grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be
        numerically equal to the absolute value of a photometric quantity,
        the constant, :math:`k`, must be put equal to the numerical value
        of :math:`K_m`, the maximum spectral luminous efficacy (which is
        equal to 683 :math:`lm\\cdot W^{-1}`) and
        :math:`\\Phi_\\lambda(\\lambda)` must be the spectral concentration
        of the radiometric quantity corresponding to the photometric
        quantity required.

    Returns
    -------
    :class:`numpy.ndarray`
        Tristimulus weighting factors table.

    Raises
    ------
    ValueError
        If the colour matching functions or illuminant intervals are not
        equal to 1 nm.

    Notes
    -----
    -   Input colour matching functions and illuminant intervals are
        expected to be equal to 1 nm. If the illuminant data is not
        available at 1 nm interval, it needs to be interpolated using *CIE*
        recommendations: The method developed by *Sprague (1880)* should be
        used for interpolating functions having a uniformly spaced
        independent variable and a *Cubic Spline* method for non-uniformly
        spaced independent variable.

    References
    ----------
    :cite:`ASTMInternational2011a`

    Examples
    --------
    >>> from colour import (
    ...     MSDS_CMFS,
    ...     SpectralDistribution,
    ...     SpectralShape,
    ...     sd_CIE_standard_illuminant_A,
    ... )
    >>> from colour.utilities import numpy_print_options
    >>> cmfs = MSDS_CMFS["CIE 1964 10 Degree Standard Observer"]
    >>> A = sd_CIE_standard_illuminant_A(cmfs.shape)
    >>> with numpy_print_options(suppress=True):
    ...     tristimulus_weighting_factors_ASTME2022(
    ...         cmfs, A, SpectralShape(360, 830, 20)
    ...     )
    ... # doctest: +ELLIPSIS
    array([[-0.0002981..., -0.0000317..., -0.0013301...],
           [-0.0087155..., -0.0008915..., -0.0407436...],
           [ 0.0599679...,  0.0050203...,  0.2565018...],
           [ 0.7734225...,  0.0779839...,  3.6965732...],
           [ 1.9000905...,  0.3037005...,  9.7554195...],
           [ 1.9707727...,  0.8552809..., 11.4867325...],
           [ 0.7183623...,  2.1457000...,  6.7845806...],
           [ 0.0426667...,  4.8985328...,  2.3208000...],
           [ 1.5223302...,  9.6471138...,  0.7430671...],
           [ 5.6770329..., 14.4609708...,  0.1958194...],
           [12.4451744..., 17.4742541...,  0.0051827...],
           [20.5535772..., 17.5838219..., -0.0026512...],
           [25.3315384..., 14.8957035...,  0.       ...],
           [21.5711570..., 10.0796619...,  0.       ...],
           [12.1785817...,  5.0680655...,  0.       ...],
           [ 4.6675746...,  1.8303239...,  0.       ...],
           [ 1.3236117...,  0.5129694...,  0.       ...],
           [ 0.3175325...,  0.1230084...,  0.       ...],
           [ 0.0746341...,  0.0290243...,  0.       ...],
           [ 0.0182990...,  0.0071606...,  0.       ...],
           [ 0.0047942...,  0.0018888...,  0.       ...],
           [ 0.0013293...,  0.0005277...,  0.       ...],
           [ 0.0004254...,  0.0001704...,  0.       ...],
           [ 0.0000962...,  0.0000389...,  0.       ...]])
    """

    if cmfs.shape.interval != 1:
        error = f'"{cmfs}" shape "interval" must be 1!'

        raise ValueError(error)

    if illuminant.shape.interval != 1:
        error = f'"{illuminant}" shape "interval" must be 1!'

        raise ValueError(error)

    global _CACHE_TRISTIMULUS_WEIGHTING_FACTORS  # noqa: PLW0602

    hash_key = hash(
        (
            cmfs,
            illuminant,
            shape,
            k,
            get_domain_range_scale(),
            type(illuminant.values).__module__,
        )
    )

    if is_caching_enabled() and hash_key in _CACHE_TRISTIMULUS_WEIGHTING_FACTORS:
        # Defensive copy in the cached array's own namespace so caller
        # mutations do not poison the cache and a backend entry is not
        # downgraded to *NumPy* on read.
        cached = _CACHE_TRISTIMULUS_WEIGHTING_FACTORS[hash_key]
        return array_namespace(cached).asarray(cached, copy=True)

    interval_i = int(shape.interval)

    xp = array_namespace(cmfs.values, illuminant.values)

    # The colour matching functions and illuminant values are promoted to a
    # single namespace so that the products below dispatch for a backend
    # ``cmfs`` or ``illuminant``.
    Y = xp_as_float_array(cmfs.values, xp=xp)
    S = xp_as_float_array(illuminant.values, xp=xp, like=Y)

    W = S[::interval_i, None] * Y[::interval_i, :]

    # First and last measurement intervals *Lagrange Coefficients*. The
    # coefficients are host *NumPy* constants and are promoted to the input
    # namespace so the products with the backend ``S`` and ``Y`` values
    # dispatch correctly.
    c_c = xp_as_float_array(
        lagrange_coefficients_ASTME2022(interval_i, "boundary"), xp=xp, like=W
    )
    # Intermediate measurement intervals *Lagrange Coefficients*.
    c_b = xp_as_float_array(
        lagrange_coefficients_ASTME2022(interval_i, "inner"), xp=xp, like=W
    )

    # Total wavelengths count.
    w_c = len(Y)
    # Measurement interval interpolated values count.
    r_c = c_b.shape[0]
    # Last interval first interpolated wavelength.
    w_lif = w_c - (w_c - 1) % interval_i - 1 - r_c

    # Intervals count.
    i_c = W.shape[0]
    i_cm = i_c - 1

    # Only apply Lagrange interpolation when interval > 1
    if r_c > 0:
        # First interval: W[:3, :] += sum over h of c_c[h, g] * S[h+1] * Y[h+1, :]
        first_interval = xp.sum(
            c_c[:, :, None] * (S[1 : r_c + 1, None, None] * Y[1 : r_c + 1, None, :]),
            axis=0,
        )
        W = xp.concat([W[:3, :] + first_interval, W[3:, :]], axis=0)

        # Last interval: W[i_cm-2:i_cm+1, :] += contributions with reversed c_c.
        # ``xp.flip`` is used rather than a ``[::-1]`` slice: backends such as
        # *PyTorch* reject a negative slice step.
        last_interval = xp.sum(
            xp.flip(c_c, axis=0)[:, :, None]
            * (S[w_lif : w_lif + r_c, None, None] * Y[w_lif : w_lif + r_c, None, :]),
            axis=0,
        )
        W = xp.concat(
            [
                W[: i_cm - 2, :],
                W[i_cm - 2 : i_cm + 1, :] + xp.flip(last_interval, axis=0),
            ],
            axis=0,
        )

        # Intermediate intervals: accumulate c_b contributions
        for h in range(i_c - 3):
            w_indices = (r_c + 1) * (h + 1) + 1 + xp.arange(r_c)
            contrib = xp.sum(
                c_b[:, :, None] * (S[w_indices, None, None] * Y[w_indices, None, :]),
                axis=0,
            )
            W = xp.concat([W[:h, :], W[h : h + 4, :] + contrib, W[h + 4 :, :]], axis=0)

        # Extrapolation of potential incomplete interval
        extrap_start = as_int_scalar(w_c - ((w_c - 1) % interval_i))
        if extrap_start < w_c:
            extrap_contrib = xp.sum(
                S[extrap_start:w_c, None] * Y[extrap_start:w_c, :], axis=0
            )
            W = xp.concat(
                [W[:i_cm, :], W[i_cm : i_cm + 1, :] + extrap_contrib, W[i_cm + 1 :, :]],
                axis=0,
            )

    with sdiv_mode():
        W = W * optional(k, sdiv(100, xp.sum(W, axis=0)[1]))

    if is_caching_enabled():
        _CACHE_TRISTIMULUS_WEIGHTING_FACTORS[hash_key] = (
            W.copy() if hasattr(W, "copy") else np.asarray(W).copy()
        )

    return W


def adjust_tristimulus_weighting_factors_ASTME308(
    W: ArrayLike, shape_r: SpectralShape, shape_t: SpectralShape
) -> NDArrayFloat:
    """
    Adjust the specified table of tristimulus weighting factors to account for a
    shorter wavelength range of the test spectral shape compared to the
    reference spectral shape using practice *ASTM E308-15* method.

    The adjustment redistributes weights at wavelengths for which data are
    not available by adding them to the weights at the shortest and longest
    wavelengths for which spectral data are available.

    Parameters
    ----------
    W
        Tristimulus weighting factors table.
    shape_r
        Reference spectral shape.
    shape_t
        Test spectral shape.

    Returns
    -------
    :class:`numpy.ndarray`
        Adjusted tristimulus weighting factors.

    References
    ----------
    :cite:`ASTMInternational2015b`

    Examples
    --------
    >>> from colour import (
    ...     MSDS_CMFS,
    ...     SpectralDistribution,
    ...     SpectralShape,
    ...     sd_CIE_standard_illuminant_A,
    ... )
    >>> from colour.utilities import numpy_print_options
    >>> cmfs = MSDS_CMFS["CIE 1964 10 Degree Standard Observer"]
    >>> A = sd_CIE_standard_illuminant_A(cmfs.shape)
    >>> W = tristimulus_weighting_factors_ASTME2022(
    ...     cmfs, A, SpectralShape(360, 830, 20)
    ... )
    >>> with numpy_print_options(suppress=True):
    ...     adjust_tristimulus_weighting_factors_ASTME308(
    ...         W, SpectralShape(360, 830, 20), SpectralShape(400, 700, 20)
    ...     )
    ... # doctest: +ELLIPSIS
    array([[ 0.0509543...,  0.0040971...,  0.2144280...],
           [ 0.7734225...,  0.0779839...,  3.6965732...],
           [ 1.9000905...,  0.3037005...,  9.7554195...],
           [ 1.9707727...,  0.8552809..., 11.4867325...],
           [ 0.7183623...,  2.1457000...,  6.7845806...],
           [ 0.0426667...,  4.8985328...,  2.3208000...],
           [ 1.5223302...,  9.6471138...,  0.7430671...],
           [ 5.6770329..., 14.4609708...,  0.1958194...],
           [12.4451744..., 17.4742541...,  0.0051827...],
           [20.5535772..., 17.5838219..., -0.0026512...],
           [25.3315384..., 14.8957035...,  0.       ...],
           [21.5711570..., 10.0796619...,  0.       ...],
           [12.1785817...,  5.0680655...,  0.       ...],
           [ 4.6675746...,  1.8303239...,  0.       ...],
           [ 1.3236117...,  0.5129694...,  0.       ...],
           [ 0.4171109...,  0.1618194...,  0.       ...]])
    """

    W = as_float_array(W)

    xp = array_namespace(W)

    start_index = int((shape_t.start - shape_r.start) / shape_r.interval)
    end_index = int((shape_r.end - shape_t.end) / shape_r.interval)

    # Compute sums of trimmed portions
    first_summation = xp.sum(W[:start_index], axis=0) if start_index > 0 else 0
    last_summation = xp.sum(W[-end_index:], axis=0) if end_index > 0 else 0

    # Get the result slice
    end_slice = -end_index if end_index > 0 else None
    W_slice = W[start_index:end_slice, ...]

    # Build adjustment array using row index broadcasting
    n = W_slice.shape[0]
    row_indices = xp.arange(n)
    adjustment = (row_indices == 0)[:, None] * first_summation + (row_indices == n - 1)[
        :, None
    ] * last_summation

    return W_slice + adjustment


def tristimulus_weighting_factors_integration(
    cmfs: MultiSpectralDistributions,
    illuminant: SpectralDistribution,
    shape: SpectralShape,
    k: Real | None = None,
) -> NDArrayFloat:
    """
    Compute the tristimulus weighting factors using the integration method.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions.
    illuminant
        Illuminant spectral distribution.
    shape
        Shape used to build the table, only the interval is needed.
    k
        Normalisation constant :math:`k`. For reflecting or transmitting
        object colours, :math:`k` is chosen so that :math:`Y = 100` for
        objects for which the spectral reflectance factor
        :math:`R(\\lambda)` of the object colour or the spectral
        transmittance factor :math:`\\tau(\\lambda)` of the object is equal
        to unity for all wavelengths. For self-luminous objects and
        illuminants, the constants :math:`k` is usually chosen on the
        grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be
        numerically equal to the absolute value of a photometric quantity,
        the constant, :math:`k`, must be put equal to the numerical value
        of :math:`K_m`, the maximum spectral luminous efficacy (which is
        equal to 683 :math:`lm\\cdot W^{-1}`) and
        :math:`\\Phi_\\lambda(\\lambda)` must be the spectral concentration
        of the radiometric quantity corresponding to the photometric
        quantity required.

    Returns
    -------
    :class:`numpy.ndarray`
        Tristimulus weighting factors table.

    References
    ----------
    :cite:`InternationalOrganizationforStandardization2024`

    Examples
    --------
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
    >>> from colour.utilities import numpy_print_options
    >>> cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    >>> illuminant = SDS_ILLUMINANTS["D65"]
    >>> shape = SpectralShape(400, 700, 20)
    >>> with numpy_print_options(suppress=True):
    ...     tristimulus_weighting_factors_integration(cmfs, illuminant, shape)
    ... # doctest: +ELLIPSIS
    array([[ 0.2236344...,  0.0061886...,  1.0603492...],
           [ 2.3710169...,  0.0705764..., 11.3910441...],
           [ 6.8970661...,  0.4554741..., 34.5974171...],
           [ 6.4697757...,  1.3348918..., 37.1366907...],
           [ 2.0937001...,  3.0433520..., 17.7966720...],
           [ 0.1011896...,  6.6702558...,  5.6170575...],
           [ 1.2520537..., 14.0502316...,  1.5484936...],
           [ 5.7256290..., 18.8094011...,  0.4002419...],
           [11.2268302..., 18.7900691...,  0.0736495...],
           [16.5750210..., 15.7374968...,  0.0298469...],
           [18.0544399..., 10.7252415...,  0.0135977...],
           [14.1509323...,  6.3099138...,  0.0031466...],
           [ 7.0795828...,  2.7660794...,  0.0003161...],
           [ 2.4979248...,  0.9240352...,  0.       ...],
           [ 0.6914277...,  0.2513207...,  0.       ...],
           [ 0.1536100...,  0.0554714...,  0.       ...]])
    """

    if cmfs.shape != shape:
        runtime_warning(f'Aligning "{cmfs.name}" cmfs shape to "{shape}".')
        cmfs = reshape_msds(cmfs, shape, copy=False)

    if illuminant.shape != shape:
        runtime_warning(f'Aligning "{illuminant.name}" illuminant shape to "{shape}".')
        illuminant = reshape_sd(illuminant, shape, copy=False)

    XYZ_b = cmfs.values
    S = illuminant.values

    xp = array_namespace(XYZ_b, S)

    XYZ_b = xp_as_float_array(XYZ_b, xp=xp)
    S = xp_as_float_array(S, xp=xp)

    d_w = cmfs.shape.interval

    # normalisation constant k from Y = 100 of perfect diffuser
    with sdiv_mode():
        k = cast("Real", optional(k, sdiv(100, (xp.sum(XYZ_b[..., 1] * S) * d_w))))

    # --- weights matrix A (DIN EN ISO 18314-4, eq. 7-8) ---
    # A[i, :] = k * S(λ_i) * [x̄(λ_i), ȳ(λ_i), z̄(λ_i)] * Δλ
    return k * S[..., None] * XYZ_b * d_w  # shape: (n, 3)


def sd_to_XYZ_integration(
    sd: ArrayLike | SpectralDistribution | MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions | None = None,
    illuminant: SpectralDistribution | None = None,
    k: Real | None = None,
    shape: SpectralShape | None = None,
) -> Range100:
    """
    Convert the specified spectral distribution to *CIE XYZ* tristimulus
    values using the specified colour matching functions and illuminant
    using the classical integration method.

    The spectral distribution can be either a
    :class:`colour.SpectralDistribution` class instance or an `ArrayLike` in
    which case the ``shape`` must be passed.

    Parameters
    ----------
    sd
        Spectral distribution, if an `ArrayLike` the wavelengths are
        expected to be in the last axis, e.g., for a spectral array with
        77 bins, ``sd`` shape could be (77, ) or (1, 77).
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to *CIE Illuminant E*.
    k
        Normalisation constant :math:`k`. For reflecting or transmitting
        object colours, :math:`k` is chosen so that :math:`Y = 100` for
        objects for which the spectral reflectance factor
        :math:`R(\\lambda)` of the object colour or the spectral
        transmittance factor :math:`\\tau(\\lambda)` of the object is equal
        to unity for all wavelengths. For self-luminous objects and
        illuminants, the constants :math:`k` is usually chosen on the
        grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be
        numerically equal to the absolute value of a photometric quantity,
        the constant, :math:`k`, must be put equal to the numerical value
        of :math:`K_m`, the maximum spectral luminous efficacy (which is
        equal to 683 :math:`lm\\cdot W^{-1}`) and
        :math:`\\Phi_\\lambda(\\lambda)` must be the spectral concentration
        of the radiometric quantity corresponding to the photometric
        quantity required.
    shape
        Spectral shape that ``sd``, ``cmfs`` and ``illuminant`` will be
        aligned to if passed.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | 100                   | 1             |
    +-----------+-----------------------+---------------+

    -   When :math:`k` is set to a value other than *None*, the computed
        *CIE XYZ* tristimulus values are assumed to be absolute and are thus
        converted from percentages by a final division by 100.
    -   The code path using the `ArrayLike` spectral distribution produces
        results different to the code path using a
        :class:`colour.SpectralDistribution` class instance: the former
        favours execution speed by aligning the colour matching functions
        and illuminant to the specified spectral shape while the latter
        favours precision by aligning the spectral distribution to the
        colour matching functions.

    References
    ----------
    :cite:`Wyszecki2000bf`

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
    >>> cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    >>> illuminant = SDS_ILLUMINANTS["D65"]
    >>> shape = SpectralShape(400, 700, 20)
    >>> data = np.array(
    ...     [
    ...         0.0641,
    ...         0.0645,
    ...         0.0562,
    ...         0.0537,
    ...         0.0559,
    ...         0.0651,
    ...         0.0705,
    ...         0.0772,
    ...         0.0870,
    ...         0.1128,
    ...         0.1360,
    ...         0.1511,
    ...         0.1688,
    ...         0.1996,
    ...         0.2397,
    ...         0.2852,
    ...     ]
    ... )
    >>> sd = SpectralDistribution(data, shape)
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant)
    ... # doctest: +ELLIPSIS
    array([10.8404805...,  9.6838697...,  6.2115722...])
    >>> sd_to_XYZ_integration(data, cmfs, illuminant, shape=shape)
    ... # doctest: +ELLIPSIS
    array([10.8993917...,  9.6986145...,  6.2540301...])

    The default CMFS are the *CIE 1931 2 Degree Standard Observer*, and the
    default illuminant is *CIE Illuminant E*:

    >>> sd_to_XYZ_integration(sd)
    ... # doctest: +ELLIPSIS
    array([11.7786939...,  9.9583972...,  5.7371816...])
    """

    as_percentage = k is not None

    # NOTE: The "illuminant" argument is reshaped by the
    # `handle_spectral_arguments` definition, but, in this case, it is not
    # desirable as we want to reshape it according to the final "shape" which,
    # if not directly passed, is only available after the subsequent if/else
    # block thus we are carefully avoiding to unpack over it.
    if illuminant is None:
        cmfs, illuminant = handle_spectral_arguments(
            cmfs, illuminant, illuminant_default="E"
        )
    else:
        cmfs, _illuminant = handle_spectral_arguments(
            cmfs, illuminant, illuminant_default="E"
        )

    if isinstance(sd, (SpectralDistribution, MultiSpectralDistributions)):
        if shape is not None:
            cmfs = reshape_msds(cmfs, shape, copy=False)
            illuminant = reshape_sd(illuminant, shape, copy=False)

        shape = cmfs.shape

        if sd.shape != shape:
            runtime_warning(f'Aligning "{sd.name}" spectral data shape to "{shape}".')

            sd = (
                reshape_sd(sd, shape, copy=False)
                if isinstance(sd, SpectralDistribution)
                else reshape_msds(sd, shape, copy=False)
            )

        R = as_float_array(sd.values)
    else:
        attest(
            shape is not None,
            "A spectral shape must be explicitly passed with a spectral data array!",
        )

        shape = cast("SpectralShape", shape)

        R = as_float_array(sd)
        wl_c = len(shape.wavelengths)
        wl_c_r = R.shape[-1]

        attest(
            wl_c_r == wl_c,
            f"Spectral data array with {wl_c_r} wavelengths is not compatible "
            f"with spectral shape with {wl_c} wavelengths!",
        )

        if cmfs.shape != shape:
            runtime_warning(f'Aligning "{cmfs.name}" cmfs shape to "{shape}".')
            cmfs = reshape_msds(cmfs, shape, copy=False)

    if illuminant.shape != shape:
        runtime_warning(f'Aligning "{illuminant.name}" illuminant shape to "{shape}".')
        illuminant = reshape_sd(illuminant, shape, copy=False)

    A = tristimulus_weighting_factors_integration(cmfs, illuminant, shape, k)

    xp = array_namespace(R, A)

    if isinstance(sd, MultiSpectralDistributions):
        R = xp_matrix_transpose(R, xp=xp)

    R = xp_as_float_array(R, xp=xp)
    A = xp_as_float_array(A, xp=xp, like=R)

    XYZ = from_range_100(xp.matmul(R, A))

    if as_percentage:
        XYZ /= 100

    return XYZ


def sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
    sd: SpectralDistribution,
    cmfs: MultiSpectralDistributions | None = None,
    illuminant: SpectralDistribution | None = None,
    k: Real | None = None,
) -> Range100:
    """
    Convert the specified spectral distribution to *CIE XYZ* tristimulus
    values using the specified colour matching functions and illuminant
    with a table of tristimulus weighting factors according to the
    *ASTM E308-15* method.

    Parameters
    ----------
    sd
        Spectral distribution.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to *CIE Illuminant E*.
    k
        Normalisation constant :math:`k`. For reflecting or transmitting
        object colours, :math:`k` is chosen so that :math:`Y = 100` for
        objects for which the spectral reflectance factor
        :math:`R(\\lambda)` of the object colour or the spectral
        transmittance factor :math:`\\tau(\\lambda)` of the object is equal
        to unity for all wavelengths. For self-luminous objects and
        illuminants, the constants :math:`k` is usually chosen on the
        grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be
        numerically equal to the absolute value of a photometric quantity,
        the constant, :math:`k`, must be put equal to the numerical value
        of :math:`K_m`, the maximum spectral luminous efficacy (which is
        equal to 683 :math:`lm\\cdot W^{-1}`) and
        :math:`\\Phi_\\lambda(\\lambda)` must be the spectral concentration
        of the radiometric quantity corresponding to the photometric
        quantity required.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | 100                   | 1             |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational2015b`

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
    >>> cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    >>> illuminant = SDS_ILLUMINANTS["D65"]
    >>> shape = SpectralShape(400, 700, 20)
    >>> data = np.array(
    ...     [
    ...         0.0641,
    ...         0.0645,
    ...         0.0562,
    ...         0.0537,
    ...         0.0559,
    ...         0.0651,
    ...         0.0705,
    ...         0.0772,
    ...         0.0870,
    ...         0.1128,
    ...         0.1360,
    ...         0.1511,
    ...         0.1688,
    ...         0.1996,
    ...         0.2397,
    ...         0.2852,
    ...     ]
    ... )
    >>> sd = SpectralDistribution(data, shape)
    >>> sd_to_XYZ_tristimulus_weighting_factors_ASTME308(
    ...     sd, cmfs, illuminant
    ... )  # doctest: +ELLIPSIS
    array([10.8405832...,  9.6844909...,  6.2155622...])

    The default CMFS are the *CIE 1931 2 Degree Standard Observer*, and the
    default illuminant is *CIE Illuminant E*:

    >>> sd_to_XYZ_tristimulus_weighting_factors_ASTME308(sd)
    ... # doctest: +ELLIPSIS
    array([11.7786111...,  9.9589055...,  5.7403205...])
    """

    cmfs, illuminant = handle_spectral_arguments(
        cmfs,
        illuminant,
        "CIE 1931 2 Degree Standard Observer",
        "E",
        SPECTRAL_SHAPE_ASTME308,
    )

    if cmfs.shape.interval != 1:
        runtime_warning(f'Interpolating "{cmfs.name}" cmfs to 1nm interval.')
        cmfs = reshape_msds(
            cmfs,
            SpectralShape(cmfs.shape.start, cmfs.shape.end, 1),
            "Interpolate",
            copy=False,
        )

    if illuminant.shape != cmfs.shape:
        runtime_warning(
            f'Aligning "{illuminant.name}" illuminant shape to "{cmfs.name}" '
            f"colour matching functions shape."
        )
        illuminant = reshape_sd(illuminant, cmfs.shape, copy=False)

    if sd.shape.boundaries != cmfs.shape.boundaries:
        runtime_warning(
            f'Trimming "{sd.name}" spectral distribution boundaries using '
            f'"{cmfs.name}" colour matching functions shape.'
        )
        sd = reshape_sd(sd, cmfs.shape, "Trim", copy=False)

    W = tristimulus_weighting_factors_ASTME2022(
        cmfs,
        illuminant,
        SpectralShape(cmfs.shape.start, cmfs.shape.end, sd.shape.interval),
        k,
    )
    start_w = cmfs.shape.start
    end_w = cmfs.shape.start + sd.shape.interval * (W.shape[0] - 1)
    W = adjust_tristimulus_weighting_factors_ASTME308(
        W, SpectralShape(start_w, end_w, sd.shape.interval), sd.shape
    )

    R = as_float_array(sd.values)

    xp = array_namespace(R)
    W = xp_as_float_array(W, xp=xp, like=R)

    XYZ = xp.sum(W * R[..., None], axis=0)

    return from_range_100(XYZ)


def sd_to_XYZ_ASTME308(
    sd: SpectralDistribution,
    cmfs: MultiSpectralDistributions | None = None,
    illuminant: SpectralDistribution | None = None,
    use_practice_range: bool = True,
    mi_5nm_omission_method: bool = True,
    mi_20nm_interpolation_method: bool = True,
    k: Real | None = None,
) -> Range100:
    """
    Convert the specified spectral distribution to *CIE XYZ* tristimulus values
    using the specified colour matching functions and illuminant according to
    practice *ASTM E308-15* method.

    Parameters
    ----------
    sd
        Spectral distribution.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to *CIE Illuminant E*.
    use_practice_range
        Practice *ASTM E308-15* working wavelengths range is [360, 780],
        if *True* this argument will trim the colour matching functions
        appropriately.
    mi_5nm_omission_method
        5 nm measurement intervals spectral distribution conversion to
        tristimulus values will use a 5 nm version of the colour matching
        functions instead of a table of tristimulus weighting factors.
    mi_20nm_interpolation_method
        20 nm measurement intervals spectral distribution conversion to
        tristimulus values will use a dedicated interpolation method instead
        of a table of tristimulus weighting factors.
    k
        Normalisation constant :math:`k`. For reflecting or transmitting
        object colours, :math:`k` is chosen so that :math:`Y = 100` for
        objects for which the spectral reflectance factor
        :math:`R(\\lambda)` of the object colour or the spectral
        transmittance factor :math:`\\tau(\\lambda)` of the object is equal
        to unity for all wavelengths. For self-luminous objects and
        illuminants, the constants :math:`k` is usually chosen on the
        grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be
        numerically equal to the absolute value of a photometric quantity,
        the constant, :math:`k`, must be put equal to the numerical value
        of :math:`K_m`, the maximum spectral luminous efficacy (which is
        equal to 683 :math:`lm\\cdot W^{-1}`) and
        :math:`\\Phi_\\lambda(\\lambda)` must be the spectral concentration
        of the radiometric quantity corresponding to the photometric
        quantity required.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | 100                   | 1             |
    +-----------+-----------------------+---------------+

    -   When :math:`k` is set to a value other than *None*, the computed
        *CIE XYZ* tristimulus values are assumed to be absolute and are thus
        converted from percentages by a final division by 100.

    References
    ----------
    :cite:`ASTMInternational2015b`

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
    >>> cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    >>> illuminant = SDS_ILLUMINANTS["D65"]
    >>> shape = SpectralShape(400, 700, 20)
    >>> data = np.array(
    ...     [
    ...         0.0641,
    ...         0.0645,
    ...         0.0562,
    ...         0.0537,
    ...         0.0559,
    ...         0.0651,
    ...         0.0705,
    ...         0.0772,
    ...         0.0870,
    ...         0.1128,
    ...         0.1360,
    ...         0.1511,
    ...         0.1688,
    ...         0.1996,
    ...         0.2397,
    ...         0.2852,
    ...     ]
    ... )
    >>> sd = SpectralDistribution(data, shape)
    >>> sd_to_XYZ_ASTME308(sd, cmfs, illuminant)
    ... # doctest: +ELLIPSIS
    array([10.8401953...,  9.6841740...,  6.2158913...])

    The default CMFS are the *CIE 1931 2 Degree Standard Observer*, and the
    default illuminant is *CIE Illuminant E*:

    >>> sd_to_XYZ_ASTME308(sd)
    ... # doctest: +ELLIPSIS
    array([11.7781589...,  9.9585580...,  5.7408602...])
    """

    as_percentage = k is not None

    cmfs, illuminant = handle_spectral_arguments(
        cmfs,
        illuminant,
        "CIE 1931 2 Degree Standard Observer",
        "E",
        SPECTRAL_SHAPE_ASTME308,
    )

    if sd.shape.interval not in (1, 5, 10, 20):
        error = (
            "Tristimulus values conversion from spectral data according to "
            'practise "ASTM E308-15" should be performed on spectral data '
            "with measurement interval of 1, 5, 10 or 20nm!"
        )

        raise ValueError(error)

    if sd.shape.interval in (10, 20) and (
        sd.shape.start % 10 != 0 or sd.shape.end % 10 != 0
    ):
        runtime_warning(
            f'"{sd.name}" spectral distribution shape does not start at a '
            f'tenth and will be aligned to "{cmfs.name}" colour matching '
            'functions shape! Note that practise "ASTM E308-15" does not '
            "define a behaviour in this case."
        )

        sd = reshape_sd(sd, cmfs.shape, copy=False)

    if use_practice_range:
        cmfs = reshape_msds(cmfs, SPECTRAL_SHAPE_ASTME308, "Trim", copy=False)

    method = sd_to_XYZ_tristimulus_weighting_factors_ASTME308
    if sd.shape.interval == 1:
        method = sd_to_XYZ_integration
    elif sd.shape.interval == 5 and mi_5nm_omission_method:
        if cmfs.shape.interval != 5:
            cmfs = reshape_msds(
                cmfs,
                SpectralShape(cmfs.shape.start, cmfs.shape.end, 5),
                "Interpolate",
                copy=False,
            )
        method = sd_to_XYZ_integration
    elif sd.shape.interval == 20 and mi_20nm_interpolation_method:
        sd = sd.copy()
        if sd.shape.boundaries != cmfs.shape.boundaries:
            runtime_warning(
                f'Trimming "{sd.name}" spectral distribution shape to '
                f'"{cmfs.name}" colour matching functions shape.'
            )
            sd = reshape_sd(sd, cmfs.shape, "Trim", copy=False)

        # Extrapolation of additional 20nm padding intervals.
        sd = reshape_sd(
            sd,
            SpectralShape(sd.shape.start - 20, sd.shape.end + 20, 10),
            copy=False,
        )

        # ASTM E308-15 prescribes Lagrange interpolants for the four padded
        # endpoints and every odd-indexed wavelength; the interpolants are
        # applied via in-place index assignment, which requires a mutable
        # array. The spectral values are materialised to *NumPy* and
        # converted back to the original namespace afterwards, supporting
        # immutable backends (e.g. *JAX*) as in :class:`colour.continuous.Signal`.
        R = as_float_array(sd.values)
        xp = array_namespace(R)

        values = np.array(as_ndarray(R))
        for i in range(2):
            values[i] = 3 * values[i + 2] - 3 * values[i + 4] + values[i + 6]
            i_e = values.shape[0] - 1 - i
            values[i_e] = values[i_e - 6] - 3 * values[i_e - 4] + 3 * values[i_e - 2]

        # Interpolating every odd numbered values.
        for i in range(3, values.shape[0] - 3, 2):
            values[i] = (
                -0.0625 * values[i - 3]
                + 0.5625 * values[i - 1]
                + 0.5625 * values[i + 1]
                - 0.0625 * values[i + 3]
            )

        sd.values = xp_as_array(values, xp=xp, like=R)

        # Discarding the additional 20nm padding intervals.
        sd = reshape_sd(
            sd,
            SpectralShape(sd.shape.start + 20, sd.shape.end - 20, 10),
            "Trim",
            copy=False,
        )

    XYZ = method(sd, cmfs, illuminant, k=k)

    if as_percentage and method is not sd_to_XYZ_integration:
        XYZ /= 100

    return XYZ


SD_TO_XYZ_METHODS = CanonicalMapping(
    {"ASTM E308": sd_to_XYZ_ASTME308, "Integration": sd_to_XYZ_integration}
)
SD_TO_XYZ_METHODS.__doc__ = """
Supported spectral distribution to *CIE XYZ* tristimulus values conversion
methods.

References
----------
:cite:`ASTMInternational2011a`, :cite:`ASTMInternational2015b`,
:cite:`Wyszecki2000bf`

Aliases:

-   'astm2015': 'ASTM E308'
"""
SD_TO_XYZ_METHODS["astm2015"] = SD_TO_XYZ_METHODS["ASTM E308"]


def sd_to_XYZ(
    sd: ArrayLike | SpectralDistribution | MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions | None = None,
    illuminant: SpectralDistribution | None = None,
    k: Real | None = None,
    method: Literal["ASTM E308", "Integration"] | str = "ASTM E308",
    **kwargs: Any,
) -> Range100:
    """
    Convert specified spectral distribution to *CIE XYZ* tristimulus values using
    specified colour matching functions, illuminant and method.

    If ``method`` is *Integration*, the spectral distribution can be either a
    :class:`colour.SpectralDistribution` class instance or an `ArrayLike` in
    which case the ``shape`` must be passed.

    Parameters
    ----------
    sd
        Spectral distribution, if an `ArrayLike` and ``method`` is
        *Integration* the wavelengths are expected to be in the last axis, e.g.,
        for a spectral array with 77 bins, ``sd`` shape could be (77, ) or
        (1, 77).
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to *CIE Illuminant E*.
    k
        Normalisation constant :math:`k`. For reflecting or transmitting
        object colours, :math:`k` is chosen so that :math:`Y = 100` for
        objects for which the spectral reflectance factor
        :math:`R(\\lambda)` of the object colour or the spectral
        transmittance factor :math:`\\tau(\\lambda)` of the object is equal
        to unity for all wavelengths. For self-luminous objects and
        illuminants, the constants :math:`k` is usually chosen on the
        grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be
        numerically equal to the absolute value of a photometric quantity,
        the constant, :math:`k`, must be put equal to the numerical value
        of :math:`K_m`, the maximum spectral luminous efficacy (which is
        equal to 683 :math:`lm\\cdot W^{-1}`) and
        :math:`\\Phi_\\lambda(\\lambda)` must be the spectral concentration
        of the radiometric quantity corresponding to the photometric
        quantity required.
    method
        Computation method.

    Other Parameters
    ----------------
    mi_5nm_omission_method
        {:func:`colour.colorimetry.sd_to_XYZ_ASTME308`},
        5 nm measurement intervals spectral distribution conversion to
        tristimulus values will use a 5 nm version of the colour matching
        functions instead of a table of tristimulus weighting factors.
    mi_20nm_interpolation_method
        {:func:`colour.colorimetry.sd_to_XYZ_ASTME308`},
        20 nm measurement intervals spectral distribution conversion to
        tristimulus values will use a dedicated interpolation method instead
        of a table of tristimulus weighting factors.
    shape
        {:func:`colour.colorimetry.sd_to_XYZ_integration`},
        Spectral shape that ``sd``, ``cmfs`` and ``illuminant`` will be
        aligned to if passed.
    use_practice_range
        {:func:`colour.colorimetry.sd_to_XYZ_ASTME308`},
        Practise *ASTM E308-15* working wavelengths range is [360, 780],
        if *True* this argument will trim the colour matching functions
        appropriately.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | 100                   | 1             |
    +-----------+-----------------------+---------------+

    -   When :math:`k` is set to a value other than *None*, the computed
        *CIE XYZ* tristimulus values are assumed to be absolute and are thus
        converted from percentages by a final division by 100.
    -   The code path using the `ArrayLike` spectral distribution produces
        results different to the code path using a
        :class:`colour.SpectralDistribution` class instance: the former
        favours execution speed by aligning the colour matching functions and
        illuminant to the specified spectral shape while the latter favours
        precision by aligning the spectral distribution to the colour matching
        functions.

    References
    ----------
    :cite:`ASTMInternational2011a`, :cite:`ASTMInternational2015b`,
    :cite:`Wyszecki2000bf`

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
    >>> cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    >>> illuminant = SDS_ILLUMINANTS["D65"]
    >>> shape = SpectralShape(400, 700, 20)
    >>> data = np.array(
    ...     [
    ...         0.0641,
    ...         0.0645,
    ...         0.0562,
    ...         0.0537,
    ...         0.0559,
    ...         0.0651,
    ...         0.0705,
    ...         0.0772,
    ...         0.0870,
    ...         0.1128,
    ...         0.1360,
    ...         0.1511,
    ...         0.1688,
    ...         0.1996,
    ...         0.2397,
    ...         0.2852,
    ...     ]
    ... )
    >>> sd = SpectralDistribution(data, shape)
    >>> sd_to_XYZ(sd, cmfs, illuminant)
    ... # doctest: +ELLIPSIS
    array([10.8401953...,  9.6841740...,  6.2158913...])
    >>> sd_to_XYZ(sd, cmfs, illuminant, use_practice_range=False)
    ... # doctest: +ELLIPSIS
    array([10.8402774...,  9.6841967...,  6.2158838...])
    >>> sd_to_XYZ(sd, cmfs, illuminant, method="Integration")
    ... # doctest: +ELLIPSIS
    array([10.8404805...,  9.6838697...,  6.2115722...])
    >>> sd_to_XYZ(data, cmfs, illuminant, method="Integration", shape=shape)
    ... # doctest: +ELLIPSIS
    array([10.8993917...,  9.6986145...,  6.2540301...])

    The default CMFS are the *CIE 1931 2 Degree Standard Observer*, and the
    default illuminant is *CIE Illuminant E*:

    >>> sd_to_XYZ(sd)
    ... # doctest: +ELLIPSIS
    array([11.7781589...,  9.9585580...,  5.7408602...])
    """

    cmfs, illuminant = handle_spectral_arguments(
        cmfs, illuminant, illuminant_default="E"
    )

    method = validate_method(method, tuple(SD_TO_XYZ_METHODS))

    global _CACHE_SD_TO_XYZ  # noqa: PLW0602

    sd_values = (
        sd.values
        if isinstance(sd, (SpectralDistribution, MultiSpectralDistributions))
        else sd
    )
    cacheable = is_caching_enabled() and not any(
        _is_gradient_tracked(a)
        for a in (sd_values, cmfs.values, illuminant.values, k)
        if a is not None
    )

    hash_key = None
    if cacheable:
        hash_key = hash(
            (
                (
                    sd
                    if isinstance(
                        sd, (SpectralDistribution, MultiSpectralDistributions)
                    )
                    # The shape and dtype are part of the key: distinct arrays can
                    # share ``tobytes`` output, e.g. a (N,) and a (1, N) array, or
                    # an integer and a float array of the same buffer.
                    else (
                        int_digest(as_ndarray(sd).tobytes()),
                        as_ndarray(sd).shape,
                        str(as_ndarray(sd).dtype),
                    )
                ),
                cmfs,
                illuminant,
                k,
                method,
                tuple(kwargs.items()),
                get_domain_range_scale(),
                type(sd_values).__module__,
            )
        )

    if cacheable and hash_key in _CACHE_SD_TO_XYZ:
        XYZ = _CACHE_SD_TO_XYZ[hash_key]

        xp = array_namespace(XYZ)

        return xp.asarray(XYZ, copy=True)

    if isinstance(sd, MultiSpectralDistributions):
        runtime_warning(
            "A multi-spectral distributions was passed, enforcing integration method!"
        )
        function = sd_to_XYZ_integration
    else:
        function = SD_TO_XYZ_METHODS[method]

    XYZ = function(sd, cmfs, illuminant, k=k, **filter_kwargs(function, **kwargs))

    if cacheable:
        xp = array_namespace(XYZ)

        _CACHE_SD_TO_XYZ[hash_key] = xp_as_array(XYZ, xp=xp, copy=True)

    return XYZ


def msds_to_XYZ_integration(
    msds: ArrayLike | SpectralDistribution | MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions | None = None,
    illuminant: SpectralDistribution | None = None,
    k: Real | None = None,
    shape: SpectralShape | None = None,
) -> Range100:
    """
    Convert the specified multi-spectral distributions to *CIE XYZ* tristimulus
    values using the specified colour matching functions and illuminant.

    The multi-spectral distributions can be either a
    :class:`colour.MultiSpectralDistributions` class instance or an
    `ArrayLike` in which case the ``shape`` must be passed.

    Parameters
    ----------
    msds
        Multi-spectral distributions, if an `ArrayLike` the wavelengths are
        expected to be in the last axis, e.g., for a 512x384 multi-spectral
        image with 77 bins, ``msds`` shape should be (384, 512, 77).
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to *CIE Illuminant E*.
    k
        Normalisation constant :math:`k`. For reflecting or transmitting
        object colours, :math:`k` is chosen so that :math:`Y = 100` for
        objects for which the spectral reflectance factor
        :math:`R(\\lambda)` of the object colour or the spectral
        transmittance factor :math:`\\tau(\\lambda)` of the object is equal
        to unity for all wavelengths. For self-luminous objects and
        illuminants, the constants :math:`k` is usually chosen on the
        grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be
        numerically equal to the absolute value of a photometric quantity,
        the constant, :math:`k`, must be put equal to the numerical value
        of :math:`K_m`, the maximum spectral luminous efficacy (which is
        equal to 683 :math:`lm\\cdot W^{-1}`) and
        :math:`\\Phi_\\lambda(\\lambda)` must be the spectral concentration
        of the radiometric quantity corresponding to the photometric
        quantity required.
    shape
        Spectral shape that ``sd``, ``cmfs`` and ``illuminant`` will be
        aligned to if passed.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values, for a 512x384 multi-spectral image with
        77 bins, the output shape will be (384, 512, 3).

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | 100                   | 1             |
    +-----------+-----------------------+---------------+

    -   When :math:`k` is set to a value other than *None*, the computed
        *CIE XYZ* tristimulus values are assumed to be absolute and are thus
        converted from percentages by a final division by 100.
    -   The code path using the `ArrayLike` multi-spectral distributions
        produces results different to the code path using a
        :class:`colour.MultiSpectralDistributions` class instance: the
        former favours execution speed by aligning the colour matching
        functions and illuminant to the specified spectral shape while the
        latter favours precision by aligning the multi-spectral distributions
        to the colour matching functions.
    -   If precision is required, it is possible to interpolate the
        multi-spectral distributions with :py:class:`scipy.interpolate.interp1d`
        class on the last / tail axis as follows:

        .. code-block:: python

            interpolator = scipy.interpolate.interp1d(
                wavelengths,
                values,
                axis=-1,
                kind="linear",
                fill_value="extrapolate",
            )
            values_i = interpolator(wavelengths_i)

    References
    ----------
    :cite:`Wyszecki2000bf`

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
    >>> cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    >>> illuminant = SDS_ILLUMINANTS["D65"]
    >>> shape = SpectralShape(400, 700, 60)
    >>> data = np.array(
    ...     [
    ...         [
    ...             0.0137,
    ...             0.0159,
    ...             0.0096,
    ...             0.0111,
    ...             0.0179,
    ...             0.1057,
    ...             0.0433,
    ...             0.0258,
    ...             0.0248,
    ...             0.0186,
    ...             0.0310,
    ...             0.0473,
    ...         ],
    ...         [
    ...             0.0913,
    ...             0.3145,
    ...             0.2582,
    ...             0.0709,
    ...             0.2971,
    ...             0.4620,
    ...             0.2683,
    ...             0.0831,
    ...             0.1203,
    ...             0.1292,
    ...             0.1682,
    ...             0.3221,
    ...         ],
    ...         [
    ...             0.0152,
    ...             0.0842,
    ...             0.4139,
    ...             0.0220,
    ...             0.5630,
    ...             0.1918,
    ...             0.2373,
    ...             0.0430,
    ...             0.0054,
    ...             0.0079,
    ...             0.3719,
    ...             0.2268,
    ...         ],
    ...         [
    ...             0.0281,
    ...             0.0907,
    ...             0.2228,
    ...             0.1249,
    ...             0.2375,
    ...             0.5625,
    ...             0.0518,
    ...             0.3230,
    ...             0.0065,
    ...             0.4006,
    ...             0.0861,
    ...             0.3161,
    ...         ],
    ...         [
    ...             0.1918,
    ...             0.7103,
    ...             0.0041,
    ...             0.1817,
    ...             0.0024,
    ...             0.4209,
    ...             0.0118,
    ...             0.2302,
    ...             0.1860,
    ...             0.9404,
    ...             0.0041,
    ...             0.1124,
    ...         ],
    ...         [
    ...             0.0430,
    ...             0.0437,
    ...             0.3744,
    ...             0.0020,
    ...             0.5819,
    ...             0.0027,
    ...             0.0823,
    ...             0.0081,
    ...             0.3625,
    ...             0.3213,
    ...             0.7849,
    ...             0.0024,
    ...         ],
    ...     ]
    ... )
    >>> msds = MultiSpectralDistributions(data, shape)
    >>> msds_to_XYZ_integration(msds, cmfs, illuminant)
    ... # doctest: +ELLIPSIS
    array([[ 7.5029704...,  3.9487844...,  8.4034669...],
           [26.9259681..., 15.0724609..., 28.7057807...],
           [16.7032188..., 28.2172346..., 25.6455984...],
           [11.5767013...,  8.6400993...,  6.5768406...],
           [18.7314793..., 35.0750364..., 30.1457266...],
           [45.1656756..., 39.6136917..., 43.6783499...],
           [ 8.1755696..., 13.0934177..., 25.9420944...],
           [22.4676286..., 19.3099080...,  7.9637549...],
           [ 6.5781241...,  2.5255349..., 11.0930768...],
           [43.9147364..., 27.9803924..., 11.7292655...],
           [ 8.5365923..., 19.7030166..., 17.7050933...],
           [23.9088250..., 26.2129529..., 30.6763148...]])
    >>> data = np.reshape(data, (2, 6, 6))
    >>> msds_to_XYZ_integration(data, cmfs, illuminant, shape=shape)
    ... # doctest: +ELLIPSIS
    array([[[ 1.3104332...,  1.1377026...,  1.8267926...],
            [ 2.1875548...,  2.2510619...,  3.0721540...],
            [16.8714661..., 17.7063715..., 35.8709902...],
            [12.1648722..., 12.7222194..., 10.4880888...],
            [16.0419431..., 23.0985768..., 11.1479902...],
            [ 9.2391014...,  3.8301575...,  5.4703803...]],
    <BLANKLINE>
           [[13.8734231..., 17.3942194..., 11.0364103...],
            [27.7096381..., 20.8626722..., 35.5581690...],
            [22.7886687..., 11.4769218..., 78.3300659...],
            [51.1284864..., 52.2463568..., 26.1483754...],
            [14.4749229..., 20.5011495...,  6.6228107...],
            [33.6001365..., 36.3242617...,  2.8254217...]]])

    The default CMFS are the *CIE 1931 2 Degree Standard Observer*, and the
    default illuminant is *CIE Illuminant E*:

    >>> msds_to_XYZ_integration(msds)
    ... # doctest: +ELLIPSIS
    array([[ 8.2415862...,  4.2543993...,  7.6100842...],
           [29.6144619..., 16.1158465..., 25.9015472...],
           [16.6799560..., 27.2350547..., 22.9413337...],
           [12.5597688...,  9.0667136...,  5.9670327...],
           [18.5804689..., 33.6618109..., 26.9249733...],
           [47.7113308..., 40.4573249..., 39.6439145...],
           [ 7.830207 ..., 12.3689624..., 23.3742655...],
           [24.1695370..., 20.0629815...,  7.2718670...],
           [ 7.2333751...,  2.7982097..., 10.0688374...],
           [48.7358074..., 30.2417164..., 10.6753233...],
           [ 8.3231013..., 18.6791507..., 15.8228184...],
           [24.6452277..., 26.0809382..., 27.7106399...]])
    """

    return sd_to_XYZ_integration(msds, cmfs, illuminant, k, shape)


def msds_to_XYZ_tristimulus_weighting_factors_ASTME308(
    msds: MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions | None = None,
    illuminant: SpectralDistribution | None = None,
    k: Real | None = None,
) -> Range100:
    """
    Convert the specified multi-spectral distributions to *CIE XYZ* tristimulus
    values using the specified colour matching functions and illuminant
    with a table of tristimulus weighting factors according to the
    *ASTM E308-15* method.

    Parameters
    ----------
    msds
        Multi-spectral distributions.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to *CIE Illuminant E*.
    k
        Normalisation constant :math:`k`. For reflecting or transmitting
        object colours, :math:`k` is chosen so that :math:`Y = 100` for
        objects for which the spectral reflectance factor
        :math:`R(\\lambda)` of the object colour or the spectral
        transmittance factor :math:`\\tau(\\lambda)` of the object is equal
        to unity for all wavelengths. For self-luminous objects and
        illuminants, the constants :math:`k` is usually chosen on the
        grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be
        numerically equal to the absolute value of a photometric quantity,
        the constant, :math:`k`, must be put equal to the numerical value
        of :math:`K_m`, the maximum spectral luminous efficacy (which is
        equal to 683 :math:`lm\\cdot W^{-1}`) and
        :math:`\\Phi_\\lambda(\\lambda)` must be the spectral concentration
        of the radiometric quantity corresponding to the photometric
        quantity required.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | 100                   | 1             |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational2015b`

    Examples
    --------
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
    >>> cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    >>> illuminant = SDS_ILLUMINANTS["D65"]
    >>> shape = SpectralShape(400, 700, 20)
    >>> values = np.array(
    ...     [
    ...         0.0641,
    ...         0.0645,
    ...         0.0562,
    ...         0.0537,
    ...         0.0559,
    ...         0.0651,
    ...         0.0705,
    ...         0.0772,
    ...         0.0870,
    ...         0.1128,
    ...         0.1360,
    ...         0.1511,
    ...         0.1688,
    ...         0.1996,
    ...         0.2397,
    ...         0.2852,
    ...     ]
    ... )
    >>> data = np.transpose([values, values])
    >>> msds = MultiSpectralDistributions(data, shape)
    >>> msds_to_XYZ_tristimulus_weighting_factors_ASTME308(
    ...     msds, cmfs, illuminant
    ... )  # doctest: +ELLIPSIS
    array([[10.8405832...,  9.6844909...,  6.2155622...],
           [10.8405832...,  9.6844909...,  6.2155622...]])
    """

    cmfs, illuminant = handle_spectral_arguments(
        cmfs,
        illuminant,
        "CIE 1931 2 Degree Standard Observer",
        "E",
        SPECTRAL_SHAPE_ASTME308,
    )

    if cmfs.shape.interval != 1:
        runtime_warning(f'Interpolating "{cmfs.name}" cmfs to 1nm interval.')
        cmfs = reshape_msds(
            cmfs,
            SpectralShape(cmfs.shape.start, cmfs.shape.end, 1),
            "Interpolate",
            copy=False,
        )

    if illuminant.shape != cmfs.shape:
        runtime_warning(
            f'Aligning "{illuminant.name}" illuminant shape to "{cmfs.name}" '
            f"colour matching functions shape."
        )
        illuminant = reshape_sd(illuminant, cmfs.shape, copy=False)

    if msds.shape.boundaries != cmfs.shape.boundaries:
        runtime_warning(
            f'Trimming "{msds.name}" multi-spectral distributions boundaries '
            f'using "{cmfs.name}" colour matching functions shape.'
        )
        msds = reshape_msds(msds, cmfs.shape, "Trim", copy=False)

    W = tristimulus_weighting_factors_ASTME2022(
        cmfs,
        illuminant,
        SpectralShape(cmfs.shape.start, cmfs.shape.end, msds.shape.interval),
        k,
    )
    start_w = cmfs.shape.start
    end_w = cmfs.shape.start + msds.shape.interval * (W.shape[0] - 1)
    W = adjust_tristimulus_weighting_factors_ASTME308(
        W, SpectralShape(start_w, end_w, msds.shape.interval), msds.shape
    )

    R = as_float_array(msds.values)

    xp = array_namespace(R)
    W = xp_as_float_array(W, xp=xp, like=R)

    XYZ = xp.matmul(xp_matrix_transpose(R, xp=xp), W)

    return from_range_100(XYZ)


def msds_to_XYZ_ASTME308(
    msds: MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions | None = None,
    illuminant: SpectralDistribution | None = None,
    k: Real | None = None,
    use_practice_range: bool = True,
    mi_5nm_omission_method: bool = True,
    mi_20nm_interpolation_method: bool = True,
) -> Range100:
    """
    Convert specified multi-spectral distributions to *CIE XYZ* tristimulus
    values using the specified colour matching functions and illuminant according
    to practise *ASTM E308-15* method.

    Parameters
    ----------
    msds
        Multi-spectral distributions.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to *CIE Illuminant E*.
    k
        Normalisation constant :math:`k`. For reflecting or transmitting
        object colours, :math:`k` is chosen so that :math:`Y = 100` for
        objects for which the spectral reflectance factor
        :math:`R(\\lambda)` of the object colour or the spectral
        transmittance factor :math:`\\tau(\\lambda)` of the object is equal
        to unity for all wavelengths. For self-luminous objects and
        illuminants, the constants :math:`k` is usually chosen on the
        grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be
        numerically equal to the absolute value of a photometric quantity,
        the constant, :math:`k`, must be put equal to the numerical value
        of :math:`K_m`, the maximum spectral luminous efficacy (which is
        equal to 683 :math:`lm\\cdot W^{-1}`) and
        :math:`\\Phi_\\lambda(\\lambda)` must be the spectral concentration
        of the radiometric quantity corresponding to the photometric
        quantity required.
    use_practice_range
        Practise *ASTM E308-15* working wavelengths range is [360, 780],
        if *True* this argument will trim the colour matching functions
        appropriately.
    mi_5nm_omission_method
        5 nm measurement intervals multi-spectral distributions conversion to
        tristimulus values will use a 5 nm version of the colour matching
        functions instead of a table of tristimulus weighting factors.
    mi_20nm_interpolation_method
        20 nm measurement intervals multi-spectral distributions conversion to
        tristimulus values will use a dedicated interpolation method instead
        of a table of tristimulus weighting factors.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | 100                   | 1             |
    +-----------+-----------------------+---------------+

    -   When :math:`k` is set to a value other than *None*, the computed
        *CIE XYZ* tristimulus values are assumed to be absolute and are thus
        converted from percentages by a final division by 100.

    References
    ----------
    :cite:`Wyszecki2000bf`

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
    >>> cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    >>> illuminant = SDS_ILLUMINANTS["D65"]
    >>> shape = SpectralShape(400, 700, 60)
    >>> data = np.array(
    ...     [
    ...         [
    ...             0.0137,
    ...             0.0159,
    ...             0.0096,
    ...             0.0111,
    ...             0.0179,
    ...             0.1057,
    ...             0.0433,
    ...             0.0258,
    ...             0.0248,
    ...             0.0186,
    ...             0.0310,
    ...             0.0473,
    ...         ],
    ...         [
    ...             0.0913,
    ...             0.3145,
    ...             0.2582,
    ...             0.0709,
    ...             0.2971,
    ...             0.4620,
    ...             0.2683,
    ...             0.0831,
    ...             0.1203,
    ...             0.1292,
    ...             0.1682,
    ...             0.3221,
    ...         ],
    ...         [
    ...             0.0152,
    ...             0.0842,
    ...             0.4139,
    ...             0.0220,
    ...             0.5630,
    ...             0.1918,
    ...             0.2373,
    ...             0.0430,
    ...             0.0054,
    ...             0.0079,
    ...             0.3719,
    ...             0.2268,
    ...         ],
    ...         [
    ...             0.0281,
    ...             0.0907,
    ...             0.2228,
    ...             0.1249,
    ...             0.2375,
    ...             0.5625,
    ...             0.0518,
    ...             0.3230,
    ...             0.0065,
    ...             0.4006,
    ...             0.0861,
    ...             0.3161,
    ...         ],
    ...         [
    ...             0.1918,
    ...             0.7103,
    ...             0.0041,
    ...             0.1817,
    ...             0.0024,
    ...             0.4209,
    ...             0.0118,
    ...             0.2302,
    ...             0.1860,
    ...             0.9404,
    ...             0.0041,
    ...             0.1124,
    ...         ],
    ...         [
    ...             0.0430,
    ...             0.0437,
    ...             0.3744,
    ...             0.0020,
    ...             0.5819,
    ...             0.0027,
    ...             0.0823,
    ...             0.0081,
    ...             0.3625,
    ...             0.3213,
    ...             0.7849,
    ...             0.0024,
    ...         ],
    ...     ]
    ... )
    >>> msds = MultiSpectralDistributions(data, shape)
    >>> msds = msds.align(SpectralShape(400, 700, 20))
    >>> msds_to_XYZ_ASTME308(msds, cmfs, illuminant)
    ... # doctest: +ELLIPSIS
    array([[ 7.5052758...,  3.9557516...,  8.38929  ...],
           [26.9408494..., 15.0987746..., 28.6631260...],
           [16.7047370..., 28.2089815..., 25.6556751...],
           [11.5711808...,  8.6445071...,  6.5587827...],
           [18.7428858..., 35.0626352..., 30.1778517...],
           [45.1224886..., 39.6238997..., 43.5813345...],
           [ 8.1786985..., 13.0950215..., 25.9326459...],
           [22.4462888..., 19.3115133...,  7.9304333...],
           [ 6.5764361...,  2.5305945..., 11.07253  ...],
           [43.9113380..., 28.0003541..., 11.6852531...],
           [ 8.5496209..., 19.6913570..., 17.7400079...],
           [23.8866733..., 26.2147704..., 30.6297684...]])

    The default CMFS are the *CIE 1931 2 Degree Standard Observer*, and the
    default illuminant is *CIE Illuminant E*:

    >>> msds_to_XYZ_ASTME308(msds)
    ... # doctest: +ELLIPSIS
    array([[ 8.2439318...,  4.2617641...,  7.5977409...],
           [29.6290771..., 16.1443076..., 25.8640484...],
           [16.6819067..., 27.2271403..., 22.9490590...],
           [12.5543694...,  9.0705685...,  5.9516323...],
           [18.5921357..., 33.6508573..., 26.9511144...],
           [47.6698072..., 40.4630866..., 39.5612904...],
           [ 7.8336896..., 12.3711768..., 23.3654245...],
           [24.1486630..., 20.0621956...,  7.2438655...],
           [ 7.2323703...,  2.8033217..., 10.0510790...],
           [48.7322793..., 30.2614779..., 10.6377135...],
           [ 8.3365770..., 18.6690888..., 15.8517212...],
           [24.6240657..., 26.0805317..., 27.6706915...]])
    """

    if not isinstance(msds, MultiSpectralDistributions):
        error = (
            '"ASTM E308-15" method does not support "ArrayLike" '
            "multi-spectral distributions!"
        )

        raise TypeError(error)

    as_percentage = k is not None

    cmfs, illuminant = handle_spectral_arguments(
        cmfs,
        illuminant,
        "CIE 1931 2 Degree Standard Observer",
        "E",
        SPECTRAL_SHAPE_ASTME308,
    )

    if msds.shape.interval not in (1, 5, 10, 20):
        error = (
            "Tristimulus values conversion from spectral data according to "
            'practise "ASTM E308-15" should be performed on spectral data '
            "with measurement interval of 1, 5, 10 or 20nm!"
        )

        raise ValueError(error)

    if msds.shape.interval in (10, 20) and (
        msds.shape.start % 10 != 0 or msds.shape.end % 10 != 0
    ):
        runtime_warning(
            f'"{msds.name}" multi-spectral distributions shape does not start '
            f'at a tenth and will be aligned to "{cmfs.name}" colour matching '
            'functions shape! Note that practise "ASTM E308-15" does not '
            "define a behaviour in this case."
        )

        msds = reshape_msds(msds, cmfs.shape, copy=False)

    if use_practice_range:
        cmfs = reshape_msds(cmfs, SPECTRAL_SHAPE_ASTME308, "Trim", copy=False)

    method = msds_to_XYZ_tristimulus_weighting_factors_ASTME308
    if msds.shape.interval == 1:
        method = msds_to_XYZ_integration
    elif msds.shape.interval == 5 and mi_5nm_omission_method:
        if cmfs.shape.interval != 5:
            cmfs = reshape_msds(
                cmfs,
                SpectralShape(cmfs.shape.start, cmfs.shape.end, 5),
                "Interpolate",
                copy=False,
            )
        method = msds_to_XYZ_integration
    elif msds.shape.interval == 20 and mi_20nm_interpolation_method:
        msds = msds.copy()
        if msds.shape.boundaries != cmfs.shape.boundaries:
            runtime_warning(
                f'Trimming "{msds.name}" multi-spectral distributions shape to '
                f'"{cmfs.name}" colour matching functions shape.'
            )
            msds = reshape_msds(msds, cmfs.shape, "Trim", copy=False)

        # Extrapolation of additional 20nm padding intervals.
        msds = reshape_msds(
            msds,
            SpectralShape(msds.shape.start - 20, msds.shape.end + 20, 10),
            copy=False,
        )

        # ASTM E308-15 prescribes Lagrange interpolants for the four padded
        # endpoints and every odd-indexed wavelength; the operations are
        # applied to the spectral values directly so the same coefficients
        # broadcast across all signals along the wavelength axis. The
        # recurrence is inherently sequential, so it runs on a *NumPy* copy
        # and the result is written back through the MSDS setter.
        R = as_ndarray(msds.values).copy()
        for i in range(2):
            R[i] = 3 * R[i + 2] - 3 * R[i + 4] + R[i + 6]
            i_e = R.shape[0] - 1 - i
            R[i_e] = R[i_e - 6] - 3 * R[i_e - 4] + 3 * R[i_e - 2]

        # Interpolating every odd numbered values.
        for i in range(3, R.shape[0] - 3, 2):
            R[i] = (
                -0.0625 * R[i - 3]
                + 0.5625 * R[i - 1]
                + 0.5625 * R[i + 1]
                - 0.0625 * R[i + 3]
            )

        msds.values = R

        # Discarding the additional 20nm padding intervals.
        msds = reshape_msds(
            msds,
            SpectralShape(msds.shape.start + 20, msds.shape.end - 20, 10),
            "Trim",
            copy=False,
        )

    XYZ = method(msds, cmfs, illuminant, k=k)

    if as_percentage and method is not msds_to_XYZ_integration:
        XYZ /= 100

    return XYZ


MSDS_TO_XYZ_METHODS = CanonicalMapping(
    {"ASTM E308": msds_to_XYZ_ASTME308, "Integration": msds_to_XYZ_integration}
)
MSDS_TO_XYZ_METHODS.__doc__ = """
Supported multi-spectral distributions to *CIE XYZ* tristimulus values
conversion methods.

References
----------
:cite:`ASTMInternational2011a`, :cite:`ASTMInternational2015b`,
:cite:`Wyszecki2000bf`

Aliases:

-   'astm2015': 'ASTM E308'
"""
MSDS_TO_XYZ_METHODS["astm2015"] = MSDS_TO_XYZ_METHODS["ASTM E308"]


def msds_to_XYZ(
    msds: ArrayLike | SpectralDistribution | MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions | None = None,
    illuminant: SpectralDistribution | None = None,
    k: Real | None = None,
    method: Literal["ASTM E308", "Integration"] | str = "ASTM E308",
    **kwargs: Any,
) -> Range100:
    """
    Convert specified multi-spectral distributions to *CIE XYZ* tristimulus
    values using the specified colour matching functions and illuminant. For the
    *Integration* method, the multi-spectral distributions can be either a
    :class:`colour.MultiSpectralDistributions` class instance or an
    `ArrayLike` in which case the ``shape`` must be passed.

    Parameters
    ----------
    msds
        Multi-spectral distributions, if an `ArrayLike` the wavelengths are
        expected to be in the last axis, e.g., for a 512x384 multi-spectral
        image with 77 bins, ``msds`` shape should be (384, 512, 77).
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to *CIE Illuminant E*.
    k
        Normalisation constant :math:`k`. For reflecting or transmitting
        object colours, :math:`k` is chosen so that :math:`Y = 100` for
        objects for which the spectral reflectance factor
        :math:`R(\\lambda)` of the object colour or the spectral
        transmittance factor :math:`\\tau(\\lambda)` of the object is equal
        to unity for all wavelengths. For self-luminous objects and
        illuminants, the constants :math:`k` is usually chosen on the
        grounds of convenience. If, however, in the CIE 1931 standard
        colorimetric system, the :math:`Y` value is required to be
        numerically equal to the absolute value of a photometric quantity,
        the constant, :math:`k`, must be put equal to the numerical value
        of :math:`K_m`, the maximum spectral luminous efficacy (which is
        equal to 683 :math:`lm\\cdot W^{-1}`) and
        :math:`\\Phi_\\lambda(\\lambda)` must be the spectral concentration
        of the radiometric quantity corresponding to the photometric
        quantity required.
    method
        Computation method.

    Other Parameters
    ----------------
    mi_5nm_omission_method
        {:func:`colour.colorimetry.msds_to_XYZ_ASTME308`},
        5 nm measurement intervals multi-spectral distributions conversion to
        tristimulus values will use a 5 nm version of the colour matching
        functions instead of a table of tristimulus weighting factors.
    mi_20nm_interpolation_method
        {:func:`colour.colorimetry.msds_to_XYZ_ASTME308`},
        20 nm measurement intervals multi-spectral distributions conversion to
        tristimulus values will use a dedicated interpolation method instead
        of a table of tristimulus weighting factors.
    shape
        {:func:`colour.colorimetry.msds_to_XYZ_integration`},
        Spectral shape that ``msds``, ``cmfs`` and ``illuminant`` will be
        aligned to if passed.
    use_practice_range
        {:func:`colour.colorimetry.msds_to_XYZ_ASTME308`},
        Practise *ASTM E308-15* working wavelengths range is [360, 780],
        if *True* this argument will trim the colour matching functions
        appropriately.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values, for a 512x384 multi-spectral image
        with 77 wavelengths, the output shape will be (384, 512, 3).

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | 100                   | 1             |
    +-----------+-----------------------+---------------+

    -   When :math:`k` is set to a value other than *None*, the computed
        *CIE XYZ* tristimulus values are assumed to be absolute and are thus
        converted from percentages by a final division by 100.
    -   The code path using the `ArrayLike` multi-spectral distributions
        produces results different to the code path using a
        :class:`colour.MultiSpectralDistributions` class instance: the
        former favours execution speed by aligning the colour matching
        functions and illuminant to the specified spectral shape while the
        latter favours precision by aligning the multi-spectral distributions
        to the colour matching functions.
    -   If precision is required, it is possible to interpolate the
        multi-spectral distributions with
        :py:class:`scipy.interpolate.interp1d` class on the last / tail axis
        as follows:

        .. code-block:: python

            interpolator = scipy.interpolate.interp1d(
                wavelengths,
                values,
                axis=-1,
                kind="linear",
                fill_value="extrapolate",
            )
            values_i = interpolator(wavelengths_i)

    References
    ----------
    :cite:`ASTMInternational2011a`, :cite:`ASTMInternational2015b`,
    :cite:`Wyszecki2000bf`

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
    >>> cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    >>> illuminant = SDS_ILLUMINANTS["D65"]
    >>> shape = SpectralShape(400, 700, 60)
    >>> data = np.array(
    ...     [
    ...         [
    ...             0.0137,
    ...             0.0159,
    ...             0.0096,
    ...             0.0111,
    ...             0.0179,
    ...             0.1057,
    ...             0.0433,
    ...             0.0258,
    ...             0.0248,
    ...             0.0186,
    ...             0.0310,
    ...             0.0473,
    ...         ],
    ...         [
    ...             0.0913,
    ...             0.3145,
    ...             0.2582,
    ...             0.0709,
    ...             0.2971,
    ...             0.4620,
    ...             0.2683,
    ...             0.0831,
    ...             0.1203,
    ...             0.1292,
    ...             0.1682,
    ...             0.3221,
    ...         ],
    ...         [
    ...             0.0152,
    ...             0.0842,
    ...             0.4139,
    ...             0.0220,
    ...             0.5630,
    ...             0.1918,
    ...             0.2373,
    ...             0.0430,
    ...             0.0054,
    ...             0.0079,
    ...             0.3719,
    ...             0.2268,
    ...         ],
    ...         [
    ...             0.0281,
    ...             0.0907,
    ...             0.2228,
    ...             0.1249,
    ...             0.2375,
    ...             0.5625,
    ...             0.0518,
    ...             0.3230,
    ...             0.0065,
    ...             0.4006,
    ...             0.0861,
    ...             0.3161,
    ...         ],
    ...         [
    ...             0.1918,
    ...             0.7103,
    ...             0.0041,
    ...             0.1817,
    ...             0.0024,
    ...             0.4209,
    ...             0.0118,
    ...             0.2302,
    ...             0.1860,
    ...             0.9404,
    ...             0.0041,
    ...             0.1124,
    ...         ],
    ...         [
    ...             0.0430,
    ...             0.0437,
    ...             0.3744,
    ...             0.0020,
    ...             0.5819,
    ...             0.0027,
    ...             0.0823,
    ...             0.0081,
    ...             0.3625,
    ...             0.3213,
    ...             0.7849,
    ...             0.0024,
    ...         ],
    ...     ]
    ... )
    >>> msds = MultiSpectralDistributions(data, shape)
    >>> msds_to_XYZ(msds, cmfs, illuminant, method="Integration")
    ... # doctest: +ELLIPSIS
    array([[ 7.5029704...,  3.9487844...,  8.4034669...],
           [26.9259681..., 15.0724609..., 28.7057807...],
           [16.7032188..., 28.2172346..., 25.6455984...],
           [11.5767013...,  8.6400993...,  6.5768406...],
           [18.7314793..., 35.0750364..., 30.1457266...],
           [45.1656756..., 39.6136917..., 43.6783499...],
           [ 8.1755696..., 13.0934177..., 25.9420944...],
           [22.4676286..., 19.3099080...,  7.9637549...],
           [ 6.5781241...,  2.5255349..., 11.0930768...],
           [43.9147364..., 27.9803924..., 11.7292655...],
           [ 8.5365923..., 19.7030166..., 17.7050933...],
           [23.9088250..., 26.2129529..., 30.6763148...]])
    >>> data = np.reshape(data, (2, 6, 6))
    >>> msds_to_XYZ(data, cmfs, illuminant, method="Integration", shape=shape)
    ... # doctest: +ELLIPSIS
    array([[[ 1.3104332...,  1.1377026...,  1.8267926...],
            [ 2.1875548...,  2.2510619...,  3.0721540...],
            [16.8714661..., 17.7063715..., 35.8709902...],
            [12.1648722..., 12.7222194..., 10.4880888...],
            [16.0419431..., 23.0985768..., 11.1479902...],
            [ 9.2391014...,  3.8301575...,  5.4703803...]],
    <BLANKLINE>
           [[13.8734231..., 17.3942194..., 11.0364103...],
            [27.7096381..., 20.8626722..., 35.5581690...],
            [22.7886687..., 11.4769218..., 78.3300659...],
            [51.1284864..., 52.2463568..., 26.1483754...],
            [14.4749229..., 20.5011495...,  6.6228107...],
            [33.6001365..., 36.3242617...,  2.8254217...]]])

    The default CMFS are the *CIE 1931 2 Degree Standard Observer*, and the
    default illuminant is *CIE Illuminant E*:

    >>> msds_to_XYZ(msds, method="Integration")
    ... # doctest: +ELLIPSIS
    array([[ 8.2415862...,  4.2543993...,  7.6100842...],
           [29.6144619..., 16.1158465..., 25.9015472...],
           [16.6799560..., 27.2350547..., 22.9413337...],
           [12.5597688...,  9.0667136...,  5.9670327...],
           [18.5804689..., 33.6618109..., 26.9249733...],
           [47.7113308..., 40.4573249..., 39.6439145...],
           [ 7.830207 ..., 12.3689624..., 23.3742655...],
           [24.1695370..., 20.0629815...,  7.2718670...],
           [ 7.2333751...,  2.7982097..., 10.0688374...],
           [48.7358074..., 30.2417164..., 10.6753233...],
           [ 8.3231013..., 18.6791507..., 15.8228184...],
           [24.6452277..., 26.0809382..., 27.7106399...]])
    """

    method = validate_method(method, tuple(MSDS_TO_XYZ_METHODS))

    function = MSDS_TO_XYZ_METHODS[method]

    return function(msds, cmfs, illuminant, k, **filter_kwargs(function, **kwargs))


def wavelength_to_XYZ(
    wavelength: ArrayLike,
    cmfs: MultiSpectralDistributions | None = None,
) -> Range1:
    """
    Convert the specified wavelength :math:`\\lambda` to *CIE XYZ* tristimulus
    values using the specified colour matching functions.

    If the wavelength :math:`\\lambda` is not available in the colour
    matching function, its value will be calculated according to
    *CIE 15:2004* recommendation: the method developed by *Sprague (1880)*
    will be used for interpolating functions having a uniformly spaced
    independent variable and the *Cubic Spline* method for non-uniformly
    spaced independent variable.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in nm.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Raises
    ------
    ValueError
        If wavelength :math:`\\lambda` is not contained in the colour
        matching functions domain.

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | 1                     | 1             |
    +-----------+-----------------------+---------------+

    Examples
    --------
    >>> from colour import MSDS_CMFS
    >>> cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    >>> wavelength_to_XYZ(480, cmfs)  # doctest: +ELLIPSIS
    array([0.09564  , 0.13902  , 0.8129501...])
    >>> wavelength_to_XYZ(480.5, cmfs)  # doctest: +ELLIPSIS
    array([0.0914287..., 0.1418350..., 0.7915726...])
    """

    wavelength = as_float_array(wavelength)

    xp = array_namespace(wavelength)

    cmfs, _illuminant = handle_spectral_arguments(cmfs)

    shape = cmfs.shape
    if xp.min(wavelength) < shape.start or xp.max(wavelength) > shape.end:
        error = (
            f'"{wavelength}nm" wavelength is not in '
            f'"[{shape.start}, {shape.end}]" domain!'
        )

        raise ValueError(error)

    return xp_reshape(
        cmfs[xp_reshape(wavelength, (-1,), xp=xp)],
        (*wavelength.shape, 3),
        xp=xp,
    )
