"""
Television Lighting Consistency Index (TLCI-2012) and Television Luminaire
Matching Factor (TLMF-2013)
==========================================================================

Define the *EBU Tech 3355* TLCI-2012 and TLMF-2013 computation objects.

-   :class:`colour.quality.ColourQuality_Specification_TLCI2012`
-   :class:`colour.quality.ColourQuality_Specification_TLMF2013`
-   :func:`colour.television_lighting_consistency_index`
-   :func:`colour.television_luminaire_matching_factor`
"""

from __future__ import annotations

import typing
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
    as_float_scalar,
    domain_range_scale,
    optional,
)

if typing.TYPE_CHECKING:
    from colour.hints import Literal, NDArrayBoolean, NDArrayFloat

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ColourQuality_Specification_TLCI2012",
    "ColourQuality_Specification_TLMF2013",
    "sd_planckian_TLCI2012",
    "sd_daylight_TLCI2012",
    "uv_to_CCT_TLCI2012",
    "sd_reference_illuminant_TLCI2012",
    "television_lighting_consistency_index",
    "television_luminaire_matching_factor",
]


@dataclass()
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


@dataclass()
class ColourQuality_Specification_TLMF2013:
    """
    Define the *Television Luminaire Matching Factor* (TLMF-2013)
    colour quality specification.

    Parameters
    ----------
    name
        Name of the test signal.
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


# EBU Tech 3355 section 1.4.2, Table 1.
_D65_XY = np.array([0.3127, 0.3290])
# EBU Tech 3355 section 1.5.1, equations [58]-[59].
_TLCI_K = 3.16
_TLCI_P = 2.4
# EBU Tech 3355 section 1.4.1, equation [28].
_DISPLAY_GAMMA = 2.4
# EBU Tech 3355 section 2, equation [61], maps the notional camera signal
# R_C' to television coding as 16 + 219 R_C'. The text states that signals
# need not be clipped at nominal peak level 235; the corresponding full-scale
# signal at code value 255 is therefore:
_STUDIO_SWING_WHITE = (255 - 16) / (235 - 16)


def _reference_loci_TLCI2012() -> tuple[tuple[NDArrayFloat, NDArrayFloat, bool], ...]:
    """
    Return the *EBU Tech 3355* reference-locus data used for CCT selection.

    *EBU Tech 3355* section 1.1.1 defines the CCT search against Appendix 2
    Planckian and daylight locus values instead of the existing CCT/D_uv
    methods such as Ohno (2013) and Robertson (1968).
    """

    planckian_temperatures = DATA_PLANCKIAN_LOCUS_TLCI2012[:, 0]
    planckian_uv = xy_to_UCS_uv(DATA_PLANCKIAN_LOCUS_TLCI2012[:, 1:])

    daylight_temperatures = DATA_DAYLIGHT_LOCUS_TLCI2012[:, 0]
    daylight_uv = xy_to_UCS_uv(DATA_DAYLIGHT_LOCUS_TLCI2012[:, 1:])

    return (
        (planckian_temperatures, planckian_uv, False),
        (daylight_temperatures, daylight_uv, True),
    )


# EBU Tech 3355 section 1.1.1 Appendix 2 Planckian and daylight reference loci.
_REFERENCE_LOCI_TLCI2012 = _reference_loci_TLCI2012()

# EBU Tech 3355 Appendix 4 test-colour samples assembled as a single
# multi-spectral distribution so the camera integration can be vectorised.
# The first 18 samples are the coloured ColorChecker patches used by TLCI;
# the remaining 6 grey scale patches are used only by TLMF. Built from the
# ``DATA_TCS_TLCI2012`` mapping rather than the lazily-evaluated
# ``SDS_TCS_TLCI2012`` so it can be assembled at module import time.
_MSDS_TCS_TLCI2012 = MultiSpectralDistributions(
    np.transpose(
        [list(DATA_TCS_TLCI2012[name].values()) for name in NAMES_TCS_TLCI2012]
    ),
    SPECTRAL_SHAPE_TLCI2012.wavelengths,
    labels=NAMES_TCS_TLCI2012,
)


def _sd_uv_TLCI2012(sd: SpectralDistribution) -> NDArrayFloat:
    """Compute CIE 1960 UCS *uv* chromaticity coordinates for ``sd``."""

    cmfs = reshape_msds(
        MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
        SPECTRAL_SHAPE_TLCI2012,
        copy=False,
    )
    with domain_range_scale("1"):
        XYZ = sd_to_XYZ(sd, cmfs, method="Integration")

    return UCS_to_uv(XYZ_to_UCS(XYZ))


def _closest_locus_intersection_TLCI2012(
    uv: NDArrayFloat,
    temperatures: NDArrayFloat,
    uv_loci: NDArrayFloat,
    is_daylight: bool,
) -> tuple[float, float, NDArrayFloat, bool] | None:
    """
    Return the closest normal intersection on one *EBU Tech 3355* locus.
    """

    uv_loci_start = uv_loci[:-1]
    uv_loci_end = uv_loci[1:]
    uv_loci_delta = uv_loci_end - uv_loci_start
    segment_lengths = np.linalg.norm(uv_loci_delta, axis=1)

    # EBU Tech 3355 section 1.1.1, equations [4]-[7]: slope of the locus,
    # distance from the test colour to the locus point, angle to the
    # horizontal, and internal angle to the CCT line.
    slopes = np.arctan2(uv_loci_delta[:, 1], uv_loci_delta[:, 0])
    uv_test_delta = uv - uv_loci_start
    radii = np.linalg.norm(uv_test_delta, axis=1)
    angles = np.arctan2(uv_test_delta[:, 1], uv_test_delta[:, 0])
    internal_angles = (angles - slopes + np.pi) % (2 * np.pi) - np.pi

    distances_along_locus = radii * np.cos(internal_angles)
    factors = distances_along_locus / segment_lengths
    distances = (radii * np.sin(internal_angles)) ** 2

    # EBU Tech 3355 section 1.1.1 defines a match as adjacent locus points
    # whose internal angles are both less than 90 degrees in magnitude. This
    # is equivalent to accepting normal projections on the locus segment.
    valid = (np.abs(internal_angles) <= np.pi / 2) & (factors >= 0) & (factors <= 1)
    if not np.any(valid):
        return None

    indices = np.nonzero(valid)[0]
    uv_intersections = (
        uv_loci_start[indices] + factors[indices, None] * uv_loci_delta[indices]
    )
    distances = distances[indices]
    index = np.argmin(distances)
    segment_index = indices[index]
    CCT = temperatures[segment_index] + factors[segment_index] * (
        temperatures[segment_index + 1] - temperatures[segment_index]
    )

    return (
        as_float_scalar(distances[index]),
        as_float_scalar(CCT),
        uv_intersections[index],
        is_daylight,
    )


def _nearest_locus_sample_TLCI2012(
    uv: NDArrayFloat,
) -> tuple[float, NDArrayFloat, bool]:
    """
    Return the nearest *EBU Tech 3355* locus sample for ``uv``.
    """

    # EBU Tech 3355 section 1.1.1 defines the CCT as the locus colour that
    # most closely matches the test colour; use the nearest Appendix 2 sample
    # when no normal intersection is found inside the tabulated locus range.
    candidates = [
        (
            np.sum((uv_locus - uv) ** 2, axis=1),
            temperatures,
            uv_locus,
            is_daylight,
        )
        for temperatures, uv_locus, is_daylight in _REFERENCE_LOCI_TLCI2012
    ]
    distances, temperatures, uv_loci, is_daylight = min(
        candidates, key=lambda candidate: np.min(candidate[0])
    )
    index = np.argmin(distances)

    return (
        as_float_scalar(temperatures[index]),
        uv_loci[index],
        is_daylight,
    )


def uv_to_CCT_TLCI2012(uv: NDArrayFloat) -> tuple[float, NDArrayFloat, bool]:
    """
    Compute the *EBU Tech 3355* correlated colour temperature and
    reference-locus point for given CIE 1960 UCS *uv* chromaticity
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
    :cite:`EuropeanBroadcastingUnion2017`

    Examples
    --------
    >>> CCT, uv_locus, is_daylight = uv_to_CCT_TLCI2012(np.array([0.19, 0.31]))
    >>> CCT  # doctest: +ELLIPSIS
    np.float64(7255.2629...)
    >>> is_daylight
    True
    """

    candidates: list[tuple[float, float, NDArrayFloat, bool]] = []
    for temperatures, uv_loci, is_daylight in _REFERENCE_LOCI_TLCI2012:
        # EBU Tech 3355 section 1.1.1, equations [4]-[8], finds the normal
        # intersection with adjacent locus points. Treat the Planckian and
        # daylight loci separately because section 1.1.2.3 states that they
        # do not join in the 3400 K to 5000 K mixed-reference region.
        candidate = _closest_locus_intersection_TLCI2012(
            uv, temperatures, uv_loci, is_daylight
        )
        if candidate is not None:
            candidates.append(candidate)

    if len(candidates) == 0:
        return _nearest_locus_sample_TLCI2012(uv)

    _, CCT, uv_locus, is_daylight = min(candidates, key=lambda candidate: candidate[0])

    return (
        CCT,
        uv_locus,
        is_daylight,
    )


def _sd_normalise_560_TLCI2012(sd: SpectralDistribution) -> SpectralDistribution:
    """
    Normalise ``sd`` to unity at 560 nm.

    *EBU Tech 3355* sections 1.1.2.1 and 1.1.2.2 specify a value of 100 at
    560 nm. The implementation uses unity because the global scale is absorbed
    by the camera white-balance step and does not affect the computed
    *TLCI-2012* or *TLMF-2013* scores.
    """

    sd = reshape_sd(sd, SPECTRAL_SHAPE_TLCI2012, "Align", copy=False)

    return SpectralDistribution(
        sd.values / sd[560],
        SPECTRAL_SHAPE_TLCI2012,
        name=sd.name,
    )


def sd_planckian_TLCI2012(
    CCT: float, shape: SpectralShape = SPECTRAL_SHAPE_TLCI2012
) -> SpectralDistribution:
    """
    Return the *EBU Tech 3355* Planckian reference spectral distribution for
    given correlated colour temperature.

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
        to unity at 560 nm.

    Notes
    -----
    -   *EBU Tech 3355* section 1.1.2.1, equation [9], uses a simplified
        Planckian expression with wavelength in nanometres, a
        :math:`1.435 \\times 10^7` nm K radiation constant, and normalisation
        at 560 nm. This implementation returns unity at 560 nm instead of the
        published value of 100; the global scale is absorbed by the camera
        white-balance step and does not affect the computed scores.

    References
    ----------
    :cite:`EuropeanBroadcastingUnion2017`

    Examples
    --------
    >>> sd = sd_planckian_TLCI2012(3200)
    >>> sd[560]  # doctest: +ELLIPSIS
    np.float64(1.0...)
    >>> sd[600]  # doctest: +ELLIPSIS
    np.float64(1.2081916...)
    """

    # EBU Tech 3355 section 1.1.2.1, equation [9].
    wavelengths = shape.wavelengths
    c_2 = 1.435e7
    values = (560 / wavelengths) ** 5 * (
        np.expm1(c_2 / (560 * CCT)) / np.expm1(c_2 / (wavelengths * CCT))
    )

    return SpectralDistribution(
        values,
        shape,
        name=f"TLCI-2012 Planckian {CCT:.0f}K",
    )


def sd_daylight_TLCI2012(
    CCT: float, shape: SpectralShape = SPECTRAL_SHAPE_TLCI2012
) -> SpectralDistribution:
    """
    Return the *EBU Tech 3355* daylight reference spectral distribution for
    given correlated colour temperature.

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
    :cite:`EuropeanBroadcastingUnion2017`

    Examples
    --------
    >>> sd = sd_daylight_TLCI2012(5600)
    >>> sd[560]  # doctest: +ELLIPSIS
    np.float64(99.9999...)
    """

    # EBU Tech 3355 section 1.1.2.2, equations [10]-[14], uses the Appendix 3
    # daylight radiation vectors and coefficients.
    x, y = CCT_to_xy_CIE_D(CCT)
    M = 0.02387 + 0.25539 * x - 0.73217 * y
    M1 = (-1.34674 - 1.77861 * x + 5.90757 * y) / M
    M2 = (0.03638 - 31.44464 * x + 30.06400 * y) / M

    daylight_basis = reshape_msds(
        MSDS_DAYLIGHT_BASIS_TLCI2012, shape, "Align", copy=False
    ).values

    return SpectralDistribution(
        daylight_basis[..., 0]
        + M1 * daylight_basis[..., 1]
        + M2 * daylight_basis[..., 2],
        shape,
        name=f"TLCI-2012 Daylight {CCT:.0f}K",
    )


def sd_reference_illuminant_TLCI2012(
    sd_test: SpectralDistribution,
) -> tuple[SpectralDistribution, float, float]:
    """
    Compute the *TLCI-2012* reference illuminant for given test spectral
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
    :cite:`EuropeanBroadcastingUnion2017`

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
    uv = _sd_uv_TLCI2012(sd_test)
    CCT, uv_locus, is_daylight = uv_to_CCT_TLCI2012(uv)

    # EBU Tech 3355 section 1.1.2 uses Planckian below 3400 K, daylight
    # above 5000 K, and the section 1.1.2.3 mixed reference between them.
    if CCT < 3400:
        sd_reference = sd_planckian_TLCI2012(CCT, shape)
    elif is_daylight:
        sd_reference = sd_daylight_TLCI2012(CCT, shape)
    else:
        # Sections 1.1.2.1 and 1.1.2.2 normalise P3400 and D5000 at
        # 560 nm before the section 1.1.2.3 interpolation, equation [15].
        sd_planckian_3400 = sd_planckian_TLCI2012(3400, shape)
        sd_daylight_5000 = _sd_normalise_560_TLCI2012(sd_daylight_TLCI2012(5000, shape))
        weight = (CCT - 3400) / (5000 - 3400)
        sd_reference = SpectralDistribution(
            linstep_function(weight, sd_planckian_3400.values, sd_daylight_5000.values),
            shape,
            name=f"TLCI-2012 Reference {CCT:.0f}K",
        )

    # EBU Tech 3355 section 1.1.1, equation [8], normalised to 0.0054 per
    # the prose following that equation.
    # Tech 3355 section 1.1.1 reverses the sign for green-side offsets,
    # where uT < uL. Section 1.1.2.3 equations [16]-[17] label d > 0 as
    # magenta and d <= 0 as green.
    D_uv = as_float_scalar(np.linalg.norm(uv - uv_locus) / 0.0054)

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


def _camera_RGB_illuminated(
    sd_illuminant: SpectralDistribution,
    msds_tcs: MultiSpectralDistributions,
    msds_camera: MultiSpectralDistributions,
    shape: SpectralShape,
) -> NDArrayFloat:
    """
    Compute raw camera RGB signals for ``msds_tcs`` under ``sd_illuminant``.

    ``msds_tcs`` and ``msds_camera`` are expected to be already aligned to
    ``shape``.
    """

    with domain_range_scale("1"):
        # EBU Tech 3355 sections 1.2 and 1.3.1, equations [18]-[19], with the
        # camera spectral sensitivities standing in for the colour matching
        # functions.
        return msds_to_XYZ(
            msds_tcs,
            msds_camera,
            reshape_sd(sd_illuminant, shape, "Align", copy=False),
            method="Integration",
        )


def _camera_RGB_flat_reflector(
    sd_illuminant: SpectralDistribution,
    msds_camera: MultiSpectralDistributions,
    shape: SpectralShape,
    reflectance: float = 0.9,
) -> NDArrayFloat:
    """
    Compute camera RGB signals for a flat neutral reflector.

    ``msds_camera`` is expected to be already aligned to ``shape``.
    """

    # EBU Tech 3355 section 1.3.1 sets the neutral reflector level to 0.9
    # so the ColorChecker white patch generates peak white.
    sd_reflector = SpectralDistribution(
        np.full(len(shape.wavelengths), reflectance),
        shape,
        name=f"{reflectance:.0%} Flat Reflector",
    )

    with domain_range_scale("1"):
        # EBU Tech 3355 sections 1.2 and 1.3.1, equations [18]-[19].
        return sd_to_XYZ(
            sd_reflector,
            msds_camera,
            reshape_sd(sd_illuminant, shape, "Align", copy=False),
        )


def _camera_to_Lab(RGB: NDArrayFloat) -> tuple[NDArrayFloat, NDArrayBoolean]:
    """
    Process white-balanced camera RGB values through the TLCI display pipeline.
    """

    # EBU Tech 3355 section 1.3.2, equations [20]-[22].
    RGB_matrix = np.matmul(RGB, MATRIX_TLCI2012_CAMERA.T)
    # EBU Tech 3355 section 1.3.2, equations [23]-[25].
    RGB_saturation = np.matmul(RGB_matrix, MATRIX_TLCI2012_SATURATION.T)
    # Section 1.5.1 excludes colours clipped in the mathematics. Negative
    # camera RGB values are excluded here; display RGB values cannot become
    # negative after the clipped OETF input below.
    clipped = np.any(RGB_matrix < 0, axis=-1) | np.any(RGB_saturation < 0, axis=-1)

    # EBU Tech 3355 section 1.3.3, equation [26], produces the R_C' G_C' B_C'
    # signals that drive the display. Keep the section 2 equation [61]
    # headroom above nominal white, but cap at the full-scale display-drive
    # signal implied by code value 255.
    RGB_prime = np.clip(
        oetf_BT709(np.clip(RGB_saturation, 0, None)), None, _STUDIO_SWING_WHITE
    )
    # EBU Tech 3355 section 1.4.1, equation [28].
    RGB_display = spow(RGB_prime, _DISPLAY_GAMMA)
    # EBU Tech 3355 section 1.4.2, equation [29].
    XYZ = np.matmul(RGB_display, MATRIX_TLCI2012_DISPLAY.T)

    # EBU Tech 3355 section 1.5, equations [30]-[33].
    return XYZ_to_Lab(XYZ, _D65_XY), clipped


def _Q_from_delta_E(delta_Es: NDArrayFloat) -> tuple[float, float]:
    """
    Compute the *TLCI-2012* quality index and aggregate colour difference.
    """

    if len(delta_Es) == 0:
        error = "All TLCI/TLMF samples were excluded by negative RGB clipping."
        raise ValueError(error)

    # EBU Tech 3355 section 1.5.1, equations [58]-[59].
    delta_E_a = as_float_scalar(np.mean(delta_Es**4) ** 0.25)
    Q_a = as_float_scalar(100.0 / (1.0 + (delta_E_a / _TLCI_K) ** _TLCI_P))

    return Q_a, delta_E_a


def _tlci_pipeline(
    sd_test: SpectralDistribution,
    sd_reference: SpectralDistribution,
    msds_camera: MultiSpectralDistributions,
    normalise_test_luma_only: bool = False,
) -> tuple[NDArrayFloat, NDArrayBoolean]:
    """
    Compute the per-sample colour differences and clipping flags for all 24
    *EBU Tech 3355* Appendix 4 test-colour samples.
    """

    shape = SPECTRAL_SHAPE_TLCI2012
    msds_camera = reshape_msds(msds_camera, shape, "Align", copy=False)
    msds_tcs = reshape_msds(_MSDS_TCS_TLCI2012, shape, "Align", copy=False)

    RGB_test = _camera_RGB_illuminated(sd_test, msds_tcs, msds_camera, shape)
    RGB_reference = _camera_RGB_illuminated(sd_reference, msds_tcs, msds_camera, shape)

    RGB_neutral_test = _camera_RGB_flat_reflector(sd_test, msds_camera, shape)
    RGB_neutral_reference = _camera_RGB_flat_reflector(sd_reference, msds_camera, shape)

    # EBU Tech 3355 section 1.3.1 colour-balances TLCI test and reference
    # luminaires independently. For TLMF, the test luminaire keeps the
    # reference-luminaire colour balance and is normalised only so the
    # camera luma signal is unity. Tech 3355 does not give a separate luma
    # equation; derive it from the matrixed neutral camera signal using the
    # display luminance row defined in section 1.4.2, equation [29].
    if normalise_test_luma_only:
        RGB_neutral_test_matrix = np.matmul(RGB_neutral_test, MATRIX_TLCI2012_CAMERA.T)
        RGB_neutral_reference_matrix = np.matmul(
            RGB_neutral_reference, MATRIX_TLCI2012_CAMERA.T
        )
        luma_test = np.dot(RGB_neutral_test_matrix, MATRIX_TLCI2012_DISPLAY[1])
        luma_reference = np.dot(
            RGB_neutral_reference_matrix, MATRIX_TLCI2012_DISPLAY[1]
        )
        RGB_test = RGB_test / RGB_neutral_reference
        RGB_test *= luma_reference / luma_test
    else:
        RGB_test = RGB_test / RGB_neutral_test

    RGB_reference = RGB_reference / RGB_neutral_reference

    Lab_test, clipped_test = _camera_to_Lab(RGB_test)
    Lab_reference, clipped_reference = _camera_to_Lab(RGB_reference)

    # EBU Tech 3355 section 1.5, equations [34]-[54], defines the
    # CIEDE2000 colour-difference path with unity k factors; use Colour's
    # existing CIE 2000 implementation for that standard calculation.
    return (
        delta_E(Lab_test, Lab_reference, method="CIE 2000"),
        clipped_test | clipped_reference,
    )


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

    References
    ----------
    :cite:`EuropeanBroadcastingUnion2017`

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
    delta_Es, invalid = _tlci_pipeline(sd_test, sd_reference, msds_camera)
    # EBU Tech 3355 section 1.5.1 uses only the first 18 coloured ColorChecker
    # patches for TLCI, excluding the grey scale patches.
    delta_Es, invalid = delta_Es[:18], invalid[:18]
    delta_Es = delta_Es[~invalid]

    Q_a, delta_E_a = _Q_from_delta_E(delta_Es)

    if additional_data:
        return ColourQuality_Specification_TLCI2012(
            name=sd_test.name or "Test",
            Q_a=Q_a,
            delta_E_a=delta_E_a,
            delta_E_s=delta_Es,
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

    References
    ----------
    :cite:`EuropeanBroadcastingUnion2017`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> television_luminaire_matching_factor(  # doctest: +ELLIPSIS
    ...     SDS_ILLUMINANTS["FL2"], SDS_ILLUMINANTS["D65"]
    ... )
    np.float64(5.39310977...)
    """

    # EBU Tech 3355 section 1.5.1 uses all 24 ColorChecker patches for
    # TLMF-2013 and section 1.3.1 normalises the test source by luma only.
    delta_E_s, invalid = _tlci_pipeline(
        sd_test,
        sd_reference,
        MSDS_CAMERA_SENSITIVITIES_TLCI2012["EBU Standard Camera"],
        normalise_test_luma_only=True,
    )
    delta_E_s = delta_E_s[~invalid]

    Q_a, delta_E_a = _Q_from_delta_E(delta_E_s)

    if additional_data:
        return ColourQuality_Specification_TLMF2013(
            name=f"{sd_test.name or 'Test'} / {sd_reference.name or 'Reference'}",
            Q_a=Q_a,
            delta_E_a=delta_E_a,
            delta_E_s=delta_E_s,
        )

    return Q_a
