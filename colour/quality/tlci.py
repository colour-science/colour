"""
TLCI-2012 and TLMF-2013
=======================

Define the *EBU Tech 3355* TLCI-2012 and TLMF-2013 computation objects.

-   :class:`colour.quality.ColourQuality_Specification_TLCI2012`
-   :class:`colour.quality.ColourQuality_Specification_TLMF2013`
-   :func:`colour.quality.tlci.colour_differences_TLCI2012`
-   :func:`colour.quality.tlci.quality_index_TLCI2012`
-   :func:`colour.television_lighting_consistency_index`
-   :func:`colour.television_luminaire_matching_factor`

References
----------
-   :cite:`EuropeanBroadcastingUnion2013` : European Broadcasting Union.
    (2013). EBU Tech 3355 - Method for the Assessment of the Colorimetric
    Properties of Luminaires.
    https://tech.ebu.ch/docs/tech/tech3355.pdf
"""

from __future__ import annotations

import typing
from contextlib import suppress
from dataclasses import dataclass

import numpy as np

from colour.algebra import linstep_function, spow
from colour.colorimetry import (
    MSDS_CMFS,
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    msds_to_XYZ,
    reshape_msds,
    reshape_sd,
    sd_to_XYZ,
)
from colour.difference import delta_E
from colour.models import UCS_to_uv, XYZ_to_Lab, XYZ_to_UCS, xy_to_UCS_uv
from colour.models.rgb.transfer_functions import oetf_BT709
from colour.quality.datasets import (
    DATA_DAYLIGHT_LOCUS_TLCI2012,
    DATA_PLANCKIAN_LOCUS_TLCI2012,
    DATA_TCS_TLCI2012,
    MATRIX_TLCI2012_CAMERA,
    MATRIX_TLCI2012_DISPLAY,
    MATRIX_TLCI2012_SATURATION,
    MSDS_CAMERA_SENSITIVITIES_TLCI2012,
    MSDS_DAYLIGHT_BASIS_TLCI2012,
    NAMES_TCS_TLCI2012,
    SPECTRAL_SHAPE_TLCI2012,
)
from colour.temperature import CCT_to_xy_CIE_D
from colour.utilities import (
    CACHE_REGISTRY,
    Structure,
    array_namespace,
    as_float_array,
    as_float_scalar,
    as_int_scalar,
    as_ndarray,
    domain_range_scale,
    is_caching_enabled,
    optional,
    xp_as_float_array,
    xp_isclose,
)

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Literal, NDArrayBoolean, NDArrayFloat

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANTS_TLCI2012",
    "ColourQuality_Specification_TLCI2012",
    "ColourQuality_Specification_TLMF2013",
    "sd_planckian_TLCI2012",
    "sd_daylight_TLCI2012",
    "uv_to_CCT_TLCI2012",
    "sd_reference_illuminant_TLCI2012",
    "colour_differences_TLCI2012",
    "quality_index_TLCI2012",
    "television_lighting_consistency_index",
    "television_luminaire_matching_factor",
]


@dataclass
class ColourQuality_Specification_TLCI2012:
    """
    Define the *Television Lighting Consistency Index* (TLCI-2012)
    colour quality specification.

    Parameters
    ----------
    name
        Name of the test spectral distribution.
    Q_a
        *TLCI-2012* score.
    delta_E_a
        Aggregate colour difference.
    delta_E_s
        Per-sample colour differences used for the final score.
    CCT
        Correlated colour temperature of the test source.
    D_uv
        Distance from the reference locus in *EBU Tech 3355* ``d`` units.
        Following *EBU Tech 3355* section 1.1.1, negative values indicate that
        the test source is towards green (:math:`u_T < u_L` in CIE 1960 UCS)
        and positive values indicate that it is towards magenta. Absolute
        values above 1.0 indicate reduced CCT reliability.

        .. note::

            This sign convention follows the published *EBU Tech 3355*
            specification (section 1.1.1 and equations [16]-[17]). The *EBU
            TLCI* application reports the opposite sign: its manual, section
            2.1.3.1.2, shows :math:`d > +1/2` towards green and
            :math:`d < -1/2` towards magenta. Values compared against that
            tool will therefore appear negated.
    """

    name: str
    Q_a: float
    delta_E_a: float
    delta_E_s: NDArrayFloat
    CCT: float
    D_uv: float


@dataclass
class ColourQuality_Specification_TLMF2013:
    """
    Define the *Television Luminaire Matching Factor* (TLMF-2013)
    colour quality specification.

    Parameters
    ----------
    name
        Names of the test and reference spectral distributions.
    Q_a
        *TLMF-2013* score.
    delta_E_a
        Aggregate colour difference.
    delta_E_s
        Per-sample colour differences used for the final score.
    """

    name: str
    Q_a: float
    delta_E_a: float
    delta_E_s: NDArrayFloat


CONSTANTS_TLCI2012: Structure = Structure(
    # Section 1.4.2, Table 1.
    xy_D65=np.array([0.3127, 0.3290]),
    # Section 1.5.1, equations [58]-[59].
    k=3.16,
    p=2.4,
    # Section 1.4.1, equation [28].
    display_gamma=2.4,
    # Section 2, equation [61], maps the notional camera signal R_C' to
    # television coding as 16 + 219 R_C'. Signals need not be clipped at
    # nominal peak level 235, so this is the full-scale signal at code 255.
    studio_swing_white=(255 - 16) / (235 - 16),
)
"""*EBU Tech 3355* TLCI-2012 and TLMF-2013 constants."""


_CACHE_REFERENCE_LOCI_TLCI2012: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_REFERENCE_LOCI_TLCI2012"
)
_CACHE_MSDS_TCS_TLCI2012: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_MSDS_TCS_TLCI2012"
)


def sd_planckian_TLCI2012(
    CCT: ArrayLike, shape: SpectralShape = SPECTRAL_SHAPE_TLCI2012
) -> SpectralDistribution:
    """
    Return the *EBU Tech 3355* Planckian reference spectral distribution for
    the given correlated colour temperature.

    Parameters
    ----------
    CCT
        Correlated colour temperature :math:`K`.
    shape
        Spectral shape of the returned spectral distribution.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        *EBU Tech 3355* Planckian reference spectral distribution, normalised
        to 100 at 560 nm.

    Notes
    -----
    -   *EBU Tech 3355* section 1.1.2.1, equation [9], uses a simplified
        Planckian expression with wavelength in nanometres, a
        :math:`1.435 \\times 10^7` nm K radiation constant, and normalisation
        at 560 nm.

    References
    ----------
    :cite:`EuropeanBroadcastingUnion2013`

    Examples
    --------
    >>> sd = sd_planckian_TLCI2012(3200)
    >>> np.round(sd[560], 7)
    np.float64(100.0)
    >>> sd[600]  # doctest: +ELLIPSIS
    np.float64(120.81916...)
    """

    CCT = as_float_array(CCT)

    xp = array_namespace(CCT)

    # EBU Tech 3355 section 1.1.2.1, equation [9].
    wavelengths = xp_as_float_array(shape.wavelengths, xp=xp, like=CCT)
    c_2 = 1.435e7
    values = (
        100
        * (560 / wavelengths) ** 5
        * (xp.expm1(c_2 / (560 * CCT)) / xp.expm1(c_2 / (wavelengths * CCT)))
    )

    return SpectralDistribution(
        values,
        shape,
        name=f"TLCI-2012 Planckian {float(as_ndarray(CCT)):.0f}K",
    )


def sd_daylight_TLCI2012(
    CCT: ArrayLike, shape: SpectralShape = SPECTRAL_SHAPE_TLCI2012
) -> SpectralDistribution:
    """
    Return the *EBU Tech 3355* daylight reference spectral distribution for
    the given correlated colour temperature.

    Parameters
    ----------
    CCT
        Correlated colour temperature :math:`K`.
    shape
        Spectral shape of the returned spectral distribution.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        *EBU Tech 3355* daylight reference spectral distribution.

    Notes
    -----
    -   *EBU Tech 3355* section 1.1.2.2, equations [10]-[14], uses the
        Appendix 3 daylight radiation vectors and *EBU Tech 3355* coefficients
        for :math:`M`, :math:`M_1`, and :math:`M_2`. This differs from
        Colour's general :func:`colour.sd_CIE_illuminant_D_series` helper,
        which uses the library CIE D-series basis data and its standard
        coefficient path.

    References
    ----------
    :cite:`EuropeanBroadcastingUnion2013`

    Examples
    --------
    >>> sd = sd_daylight_TLCI2012(5600)
    >>> np.round(sd[560], 7)
    np.float64(100.0)
    """

    CCT = as_float_array(CCT)

    # EBU Tech 3355 section 1.1.2.2, equations [10]-[14], uses the Appendix 3
    # daylight radiation vectors and coefficients.
    x, y = CCT_to_xy_CIE_D(CCT)
    xp = array_namespace(x, y)

    M = 0.02387 + 0.25539 * x - 0.73217 * y
    M1 = (-1.34674 - 1.77861 * x + 5.90757 * y) / M
    M2 = (0.03638 - 31.44464 * x + 30.06400 * y) / M

    daylight_basis = reshape_msds(
        MSDS_DAYLIGHT_BASIS_TLCI2012, shape, "Align", copy=False
    ).values
    daylight_basis = xp_as_float_array(daylight_basis, xp=xp, like=x)

    return SpectralDistribution(
        daylight_basis[..., 0]
        + M1 * daylight_basis[..., 1]
        + M2 * daylight_basis[..., 2],
        shape,
        name=f"TLCI-2012 Daylight {float(as_ndarray(CCT)):.0f}K",
    )


def uv_to_CCT_TLCI2012(uv: NDArrayFloat) -> tuple[float, NDArrayFloat, bool]:
    """
    Compute the *EBU Tech 3355* correlated colour temperature and
    reference-locus point for the given CIE 1960 UCS *uv* chromaticity
    coordinates.

    Parameters
    ----------
    uv
        CIE 1960 UCS *uv* chromaticity coordinates of the test source.

    Returns
    -------
    :class:`tuple`
        Correlated colour temperature :math:`K`, CIE 1960 UCS *uv*
        chromaticity coordinates of the closest reference-locus point, and
        whether that point lies on the daylight locus (as opposed to the
        Planckian locus).

    Notes
    -----
    -   *EBU Tech 3355* section 1.1.1 defines the CCT search against the
        Appendix 2 Planckian and daylight locus values, not the existing
        CCT/D_uv methods such as *Ohno (2013)* and *Robertson (1968)*. The
        Planckian and daylight loci are kept separate because section 1.1.2.3
        states that they do not join in the 3400 K to 5000 K mixed-reference
        region.

    References
    ----------
    :cite:`EuropeanBroadcastingUnion2013`

    Examples
    --------
    >>> CCT, uv_locus, is_daylight = uv_to_CCT_TLCI2012(np.array([0.19, 0.31]))
    >>> CCT  # doctest: +ELLIPSIS
    np.float64(7255.2629...)
    >>> is_daylight
    True
    """

    uv = as_float_array(uv)

    xp = array_namespace(uv)

    cache_key = "Reference Loci"
    if is_caching_enabled() and cache_key in _CACHE_REFERENCE_LOCI_TLCI2012:
        reference_loci = _CACHE_REFERENCE_LOCI_TLCI2012[cache_key]
    else:
        # EBU Tech 3355 section 1.1.1 Appendix 2 Planckian and daylight
        # reference loci.
        reference_loci = (
            (
                DATA_PLANCKIAN_LOCUS_TLCI2012[:, 0],
                xy_to_UCS_uv(DATA_PLANCKIAN_LOCUS_TLCI2012[:, 1:]),
                False,
            ),
            (
                DATA_DAYLIGHT_LOCUS_TLCI2012[:, 0],
                xy_to_UCS_uv(DATA_DAYLIGHT_LOCUS_TLCI2012[:, 1:]),
                True,
            ),
        )
        _CACHE_REFERENCE_LOCI_TLCI2012[cache_key] = reference_loci

    candidates: list[tuple[float, float, NDArrayFloat, bool]] = []
    for temperatures_data, uv_loci_data, is_daylight in reference_loci:
        temperatures = xp_as_float_array(temperatures_data, xp=xp, like=uv)
        uv_loci = xp_as_float_array(uv_loci_data, xp=xp, like=uv)

        # EBU Tech 3355 section 1.1.1, equations [4]-[8], finds the normal
        # intersection with adjacent locus points. Treat the Planckian and
        # daylight loci separately because section 1.1.2.3 states that they
        # do not join in the 3400 K to 5000 K mixed-reference region.
        uv_loci_start = uv_loci[:-1]
        uv_loci_delta = uv_loci[1:] - uv_loci_start
        segment_lengths = xp.linalg.vector_norm(uv_loci_delta, axis=1)

        # Equations [4]-[7]: slope of the locus, distance from the test colour
        # to the locus point, angle to the horizontal, and internal angle to
        # the CCT line.
        slopes = xp.atan2(uv_loci_delta[:, 1], uv_loci_delta[:, 0])
        uv_test_delta = uv - uv_loci_start
        radii = xp.linalg.vector_norm(uv_test_delta, axis=1)
        angles = xp.atan2(uv_test_delta[:, 1], uv_test_delta[:, 0])
        internal_angles = (angles - slopes + xp.pi) % (2 * xp.pi) - xp.pi

        factors = radii * xp.cos(internal_angles) / segment_lengths
        # The angle is undefined when the test colour coincides with a locus
        # sample. Accept that zero-radius endpoint explicitly; otherwise the
        # first daylight sample, D5000, can be rejected in favour of a nearby
        # Planckian projection.
        at_locus_sample = xp_isclose(radii, 0, atol=np.finfo(float).eps, rtol=0, xp=xp)
        valid = at_locus_sample | (
            (xp.abs(internal_angles) <= xp.pi / 2) & (factors >= 0) & (factors <= 1)
        )
        if not xp.any(valid):
            continue

        indices = xp.nonzero(valid)[0]
        uv_intersections = (
            uv_loci_start[indices] + factors[indices, None] * uv_loci_delta[indices]
        )
        distances = (radii[indices] * xp.sin(internal_angles[indices])) ** 2
        index = as_int_scalar(xp.argmin(distances))
        segment_index = indices[index]
        CCT = temperatures[segment_index] + factors[segment_index] * (
            temperatures[segment_index + 1] - temperatures[segment_index]
        )
        candidates.append(
            (
                as_float_scalar(distances[index]),
                as_float_scalar(CCT),
                uv_intersections[index],
                is_daylight,
            )
        )

    if not candidates:
        # Use the nearest Appendix 2 sample when no normal intersection lies
        # inside the tabulated locus range.
        nearest_candidates = [
            (
                xp.sum(
                    (xp_as_float_array(uv_locus, xp=xp, like=uv) - uv) ** 2,
                    axis=1,
                ),
                xp_as_float_array(temperatures, xp=xp, like=uv),
                xp_as_float_array(uv_locus, xp=xp, like=uv),
                is_daylight,
            )
            for temperatures, uv_locus, is_daylight in reference_loci
        ]
        distances, temperatures, uv_loci, is_daylight = min(
            nearest_candidates,
            key=lambda candidate: as_float_scalar(xp.min(candidate[0])),
        )
        index = as_int_scalar(xp.argmin(distances))

        return (
            as_float_scalar(temperatures[index]),
            uv_loci[index],
            is_daylight,
        )

    _, CCT, uv_locus, is_daylight = min(candidates, key=lambda candidate: candidate[0])

    return (
        CCT,
        uv_locus,
        is_daylight,
    )


def sd_reference_illuminant_TLCI2012(
    sd_test: SpectralDistribution,
) -> tuple[SpectralDistribution, float, float]:
    """
    Compute the *TLCI-2012* reference illuminant for the given test spectral
    distribution.

    Parameters
    ----------
    sd_test
        Test spectral distribution.

    Returns
    -------
    :class:`tuple`
        *TLCI-2012* reference illuminant spectral distribution, correlated
        colour temperature :math:`K` of the test source, and the ``D_uv``
        distance from the reference locus in *EBU Tech 3355* ``d`` units.

    Notes
    -----
    -   *EBU Tech 3355* section 1.1.2 uses a Planckian reference below 3400 K,
        a daylight reference above 5000 K, and the section 1.1.2.3 mixed
        reference between them. The mixed reference is a linear interpolation
        between fixed Planckian 3400 K and daylight 5000 K spectra, not
        spectra at the test source CCT.
    -   The ``D_uv`` sign follows the published *EBU Tech 3355* convention;
        see :class:`colour.quality.ColourQuality_Specification_TLCI2012`.

    References
    ----------
    :cite:`EuropeanBroadcastingUnion2013`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd_reference, CCT, D_uv = sd_reference_illuminant_TLCI2012(
    ...     SDS_ILLUMINANTS["D65"]
    ... )
    >>> CCT  # doctest: +ELLIPSIS
    np.float64(6505.0965...)
    """

    shape = SPECTRAL_SHAPE_TLCI2012
    sd_test = reshape_sd(sd_test, shape, "Align", copy=False)

    # EBU Tech 3355 section 1.1.1, equations [1]-[3].
    cmfs = reshape_msds(
        MSDS_CMFS["CIE 1931 2 Degree Standard Observer"], shape, copy=False
    )
    with domain_range_scale("1"):
        XYZ = sd_to_XYZ(sd_test, cmfs, method="Integration")
    uv = UCS_to_uv(XYZ_to_UCS(XYZ))
    CCT, uv_locus, is_daylight = uv_to_CCT_TLCI2012(uv)
    xp = array_namespace(CCT, uv)

    # EBU Tech 3355 section 1.1.2 uses Planckian below 3400 K, daylight
    # above 5000 K, and the section 1.1.2.3 mixed reference between them.
    if CCT < 3400:
        sd_reference = sd_planckian_TLCI2012(CCT, shape)
    elif is_daylight:
        sd_reference = sd_daylight_TLCI2012(CCT, shape)
    else:
        # Sections 1.1.2.1 and 1.1.2.2 normalise P3400 and D5000 at
        # 560 nm before the section 1.1.2.3 interpolation, equation [15].
        sd_planckian_3400 = sd_planckian_TLCI2012(
            xp_as_float_array(3400, xp=xp, like=CCT), shape
        )
        sd_daylight_5000 = sd_daylight_TLCI2012(
            xp_as_float_array(5000, xp=xp, like=CCT), shape
        )
        weight = (CCT - 3400) / (5000 - 3400)
        name = "TLCI-2012 Reference"
        # A traced scalar cannot be materialised only to decorate the
        # spectrum name; the numerical result does not depend on it.
        with suppress(AttributeError, TypeError):
            name = f"{name} {float(as_ndarray(CCT)):.0f}K"
        sd_reference = SpectralDistribution(
            linstep_function(weight, sd_planckian_3400.values, sd_daylight_5000.values),
            shape,
            name=name,
        )

    # EBU Tech 3355 section 1.1.1, equation [8], normalised to 0.0054 per
    # the prose following that equation.
    # Tech 3355 section 1.1.1 reverses the sign for green-side offsets,
    # where uT < uL. Section 1.1.2.3 equations [16]-[17] label d > 0 as
    # magenta and d <= 0 as green.
    uv_locus = xp_as_float_array(uv_locus, xp=xp, like=uv)
    D_uv = as_float_scalar(xp.linalg.vector_norm(uv - uv_locus) / 0.0054)

    if uv[0] < uv_locus[0]:
        D_uv *= -1

    if 4000 <= CCT < 5000:
        # Section 1.1.2.3 equations [16]-[17] adjust d because the
        # Planckian and daylight loci are separated in the mixed region.
        if D_uv > 0:
            D_uv += 0.9 * (5000 - CCT) / (5000 - 4000)
        else:
            D_uv -= 0.9 * (CCT - 4000) / (5000 - 4000)

    return sd_reference, CCT, D_uv


def colour_differences_TLCI2012(
    sd_test: SpectralDistribution,
    sd_reference: SpectralDistribution,
    msds_camera: MultiSpectralDistributions,
    normalise_test_luma_only: bool = False,
) -> tuple[NDArrayFloat, NDArrayBoolean]:
    """
    Compute the per-sample colour differences and clipping flags for all 24
    *EBU Tech 3355* Appendix 4 test-colour samples.

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    sd_reference
        Reference spectral distribution.
    msds_camera
        Camera spectral sensitivities.
    normalise_test_luma_only
        Whether to apply the *TLMF-2013* test-source luma-only normalisation
        instead of independently balancing both sources for *TLCI-2012*.

    Returns
    -------
    :class:`tuple`
        Per-sample CIEDE2000 colour differences and flags indicating samples
        excluded by negative camera RGB clipping.

    References
    ----------
    :cite:`EuropeanBroadcastingUnion2013`
    """

    shape = SPECTRAL_SHAPE_TLCI2012
    msds_camera = reshape_msds(msds_camera, shape, "Align", copy=False)
    cache_key = "Test Colour Samples"
    if is_caching_enabled() and cache_key in _CACHE_MSDS_TCS_TLCI2012:
        msds_tcs = _CACHE_MSDS_TCS_TLCI2012[cache_key]
    else:
        # EBU Tech 3355 Appendix 4 test-colour samples assembled as a single
        # multi-spectral distribution so the camera integration can be
        # vectorised. The first 18 samples are the coloured ColorChecker
        # patches used by TLCI; the remaining 6 grey scale patches are used
        # only by TLMF.
        msds_tcs = MultiSpectralDistributions(
            [
                [DATA_TCS_TLCI2012[name][wavelength] for name in NAMES_TCS_TLCI2012]
                for wavelength in shape.wavelengths
            ],
            shape.wavelengths,
            labels=NAMES_TCS_TLCI2012,
        )
        _CACHE_MSDS_TCS_TLCI2012[cache_key] = msds_tcs

    msds_tcs = reshape_msds(msds_tcs, shape, "Align", copy=False)

    xp = array_namespace(sd_test.values, sd_reference.values, msds_camera.values)
    msds_tcs = msds_tcs.copy(xp=xp)
    msds_camera = msds_camera.copy(xp=xp)

    # EBU Tech 3355 section 1.3.1 sets the neutral reflector level to 0.9 so
    # the ColorChecker white patch generates peak white.
    sd_reflector = SpectralDistribution(
        xp.full((len(shape.wavelengths),), 0.9),
        shape,
        name="90% Flat Reflector",
    )

    with domain_range_scale("1"):
        # EBU Tech 3355 sections 1.2 and 1.3.1, equations [18]-[19], with the
        # camera sensitivities standing in for the colour matching functions.
        RGB_test, RGB_reference = (
            msds_to_XYZ(
                msds_tcs,
                msds_camera,
                reshape_sd(sd_illuminant, shape, "Align", copy=False),
                method="Integration",
            )
            for sd_illuminant in (sd_test, sd_reference)
        )
        RGB_neutral_test, RGB_neutral_reference = (
            sd_to_XYZ(
                sd_reflector,
                msds_camera,
                reshape_sd(sd_illuminant, shape, "Align", copy=False),
                method="Integration",
            )
            for sd_illuminant in (sd_test, sd_reference)
        )

    if normalise_test_luma_only:
        # Section 1.3.1 applies the reference-source balance coefficients to
        # both TLMF sources, then normalises test-source camera luma to unity.
        RGB_test = RGB_test / RGB_neutral_reference
        RGB_neutral_test = RGB_neutral_test / RGB_neutral_reference
        matrix_camera = xp_as_float_array(
            MATRIX_TLCI2012_CAMERA.T, xp=xp, like=RGB_neutral_test
        )
        RGB_neutral_test_matrix = xp.matmul(RGB_neutral_test, matrix_camera)
        luma_test = xp.sum(
            RGB_neutral_test_matrix
            * xp_as_float_array(
                MATRIX_TLCI2012_DISPLAY[1], xp=xp, like=RGB_neutral_test_matrix
            )
        )
        RGB_test = RGB_test / luma_test
    else:
        RGB_test = RGB_test / RGB_neutral_test

    RGB_reference = RGB_reference / RGB_neutral_reference

    Lab_values = []
    clipped_values = []
    for RGB in (RGB_test, RGB_reference):
        xp = array_namespace(RGB)

        # Section 1.3.2, equations [20]-[25].
        RGB_matrix = xp.matmul(
            RGB,
            xp_as_float_array(MATRIX_TLCI2012_CAMERA.T, xp=xp, like=RGB),
        )
        RGB_saturation = xp.matmul(
            RGB_matrix,
            xp_as_float_array(MATRIX_TLCI2012_SATURATION.T, xp=xp, like=RGB_matrix),
        )
        # Section 1.5.1 excludes colours clipped in the mathematics. Display
        # RGB cannot become negative after the clipped OETF input below.
        clipped_values.append(
            xp.any(RGB_matrix < 0, axis=-1) | xp.any(RGB_saturation < 0, axis=-1)
        )

        # Section 1.3.3, equation [26], followed by sections 1.4.1 and 1.4.2,
        # equations [28]-[29]. Preserve the section 2 equation [61] headroom
        # above nominal white, capped at the code-value-255 display drive.
        RGB_prime = xp.clip(
            oetf_BT709(xp.clip(RGB_saturation, 0, None)),
            None,
            CONSTANTS_TLCI2012.studio_swing_white,
        )
        RGB_display = spow(RGB_prime, CONSTANTS_TLCI2012.display_gamma)
        XYZ = xp.matmul(
            RGB_display,
            xp_as_float_array(MATRIX_TLCI2012_DISPLAY.T, xp=xp, like=RGB_display),
        )
        # Section 1.5, equations [30]-[33].
        Lab_values.append(
            XYZ_to_Lab(
                XYZ,
                xp_as_float_array(CONSTANTS_TLCI2012.xy_D65, xp=xp, like=XYZ),
            )
        )

    # EBU Tech 3355 section 1.5, equations [34]-[54], defines the
    # CIEDE2000 colour-difference path with unity k factors; use Colour's
    # existing CIE 2000 implementation for that standard calculation.
    return (
        delta_E(Lab_values[0], Lab_values[1], method="CIE 2000"),
        clipped_values[0] | clipped_values[1],
    )


def quality_index_TLCI2012(delta_E_s: ArrayLike) -> tuple[float, float]:
    """
    Compute the *TLCI-2012* or *TLMF-2013* quality index and aggregate colour
    difference.

    Parameters
    ----------
    delta_E_s
        Per-sample CIEDE2000 colour differences.

    Returns
    -------
    :class:`tuple`
        Quality index and aggregate colour difference.

    Raises
    ------
    ValueError
        If every sample was excluded by negative RGB clipping.

    References
    ----------
    :cite:`EuropeanBroadcastingUnion2013`
    """

    delta_E_s = as_float_array(delta_E_s)

    if 0 in delta_E_s.shape:
        error = "All TLCI/TLMF samples were excluded by negative RGB clipping."
        raise ValueError(error)

    xp = array_namespace(delta_E_s)

    # EBU Tech 3355 section 1.5.1, equations [58]-[59].
    delta_E_a = as_float_scalar(xp.mean(delta_E_s**4) ** 0.25)
    Q_a = as_float_scalar(
        100.0 / (1.0 + (delta_E_a / CONSTANTS_TLCI2012.k) ** CONSTANTS_TLCI2012.p)
    )

    return Q_a, delta_E_a


@typing.overload
def television_lighting_consistency_index(
    sd_test: SpectralDistribution,
    camera: str | None,
    additional_data: Literal[True],
) -> ColourQuality_Specification_TLCI2012: ...


@typing.overload
def television_lighting_consistency_index(
    sd_test: SpectralDistribution,
    camera: str | None = None,
    *,
    additional_data: Literal[True],
) -> ColourQuality_Specification_TLCI2012: ...


@typing.overload
def television_lighting_consistency_index(
    sd_test: SpectralDistribution,
    camera: str | None = None,
    *,
    additional_data: Literal[False],
) -> float: ...


@typing.overload
def television_lighting_consistency_index(
    sd_test: SpectralDistribution,
    camera: str | None = None,
    additional_data: Literal[False] = False,
) -> float: ...


def television_lighting_consistency_index(
    sd_test: SpectralDistribution,
    camera: str | None = None,
    additional_data: bool = False,
) -> float | ColourQuality_Specification_TLCI2012:
    """
    Compute the *Television Lighting Consistency Index* (TLCI-2012) for the
    specified test spectral distribution.

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    camera
        Camera sensitivity dataset name. If *None*, the *EBU Standard Camera*
        defined by *EBU Tech 3355* is used.
    additional_data
        Whether to return the detailed
        :class:`colour.quality.ColourQuality_Specification_TLCI2012` instead
        of only the *TLCI-2012* score.

    Returns
    -------
    :class:`float` or \
    :class:`colour.quality.ColourQuality_Specification_TLCI2012`
        *TLCI-2012* score or detailed specification.

    Notes
    -----
    -   The score is piecewise differentiable with respect to the test
        spectral values while the reference-illuminant branch and excluded
        sample set remain unchanged. Its derivative is undefined where either
        selection changes.

    References
    ----------
    :cite:`EuropeanBroadcastingUnion2013`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> television_lighting_consistency_index(  # doctest: +ELLIPSIS
    ...     SDS_ILLUMINANTS["FL2"]
    ... )
    np.float64(29.49254175...)
    """

    camera = optional(camera, "EBU Standard Camera")
    msds_camera = MSDS_CAMERA_SENSITIVITIES_TLCI2012[camera]

    sd_reference, CCT, D_uv = sd_reference_illuminant_TLCI2012(sd_test)
    delta_E_s, invalid = colour_differences_TLCI2012(sd_test, sd_reference, msds_camera)
    # EBU Tech 3355 section 1.5.1 uses only the first 18 coloured ColorChecker
    # patches for TLCI, excluding the grey scale patches.
    delta_E_s, invalid = delta_E_s[:18], invalid[:18]
    delta_E_s = delta_E_s[~invalid]

    Q_a, delta_E_a = quality_index_TLCI2012(delta_E_s)

    if additional_data:
        return ColourQuality_Specification_TLCI2012(
            name=sd_test.name or "Test",
            Q_a=Q_a,
            delta_E_a=delta_E_a,
            delta_E_s=delta_E_s,
            CCT=CCT,
            D_uv=D_uv,
        )

    return Q_a


@typing.overload
def television_luminaire_matching_factor(
    sd_test: SpectralDistribution,
    sd_reference: SpectralDistribution,
    additional_data: Literal[True],
) -> ColourQuality_Specification_TLMF2013: ...


@typing.overload
def television_luminaire_matching_factor(
    sd_test: SpectralDistribution,
    sd_reference: SpectralDistribution,
    *,
    additional_data: Literal[False],
) -> float: ...


@typing.overload
def television_luminaire_matching_factor(
    sd_test: SpectralDistribution,
    sd_reference: SpectralDistribution,
    additional_data: Literal[False] = False,
) -> float: ...


def television_luminaire_matching_factor(
    sd_test: SpectralDistribution,
    sd_reference: SpectralDistribution,
    additional_data: bool = False,
) -> float | ColourQuality_Specification_TLMF2013:
    """
    Compute the *Television Luminaire Matching Factor* (TLMF-2013).

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    sd_reference
        Reference spectral distribution.
    additional_data
        Whether to return the detailed
        :class:`colour.quality.ColourQuality_Specification_TLMF2013` instead
        of only the *TLMF-2013* score.

    Returns
    -------
    :class:`float` or \
    :class:`colour.quality.ColourQuality_Specification_TLMF2013`
        *TLMF-2013* score or detailed specification.

    Notes
    -----
    -   The score is piecewise differentiable with respect to the test
        spectral values while the excluded sample set remains unchanged. Its
        derivative is undefined where that selection changes.

    References
    ----------
    :cite:`EuropeanBroadcastingUnion2013`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> television_luminaire_matching_factor(  # doctest: +ELLIPSIS
    ...     SDS_ILLUMINANTS["FL2"], SDS_ILLUMINANTS["D65"]
    ... )
    np.float64(5.37607960...)
    """

    # EBU Tech 3355 section 1.5.1 uses all 24 ColorChecker patches for
    # TLMF-2013 and section 1.3.1 normalises the test source by luma only.
    delta_E_s, invalid = colour_differences_TLCI2012(
        sd_test,
        sd_reference,
        MSDS_CAMERA_SENSITIVITIES_TLCI2012["EBU Standard Camera"],
        normalise_test_luma_only=True,
    )
    delta_E_s = delta_E_s[~invalid]

    Q_a, delta_E_a = quality_index_TLCI2012(delta_E_s)

    if additional_data:
        return ColourQuality_Specification_TLMF2013(
            name=f"{sd_test.name or 'Test'} / {sd_reference.name or 'Reference'}",
            Q_a=Q_a,
            delta_E_a=delta_E_a,
            delta_E_s=delta_E_s,
        )

    return Q_a
