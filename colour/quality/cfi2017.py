"""
CIE 2017 Colour Fidelity Index
==============================

Define the *CIE 2017 Colour Fidelity Index* (CFI) computation objects.

- :class:`colour.quality.ColourRendering_Specification_CIE2017`
- :func:`colour.quality.colour_fidelity_index_CIE2017`

References
----------
-   :cite:`CIETC1-902017` : CIE TC 1-90. (2017). CIE 2017 colour fidelity index
    for accurate scientific use. CIE Central Bureau. ISBN:978-3-902842-61-9
"""

from __future__ import annotations

import os
import typing
from dataclasses import dataclass

import numpy as np

from colour.algebra import Extrapolator, euclidean_distance
from colour.appearance import (
    VIEWING_CONDITIONS_CIECAM02,
    CAM_Specification_CIECAM02,
    XYZ_to_CIECAM02,
)
from colour.colorimetry import (
    MSDS_CMFS,
    CIE_illuminant_D_series,
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    msds_to_XYZ,
    planck_law,
    reshape_msds,
)

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, List, Literal, Tuple

from colour.hints import NDArrayFloat, cast
from colour.models import JMh_CIECAM02_to_CAM02UCS, UCS_to_uv, XYZ_to_UCS
from colour.temperature import CCT_to_xy_CIE_D, uv_to_CCT_Ohno2013
from colour.utilities import (
    CACHE_REGISTRY,
    array_namespace,
    as_float,
    as_float_array,
    as_float_scalar,
    as_int_scalar,
    as_ndarray,
    attest,
    is_caching_enabled,
    suppress_warnings,
    tstack,
    usage_warning,
    xp_as_float_array,
    xp_average,
    xp_matrix_transpose,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SPECTRAL_SHAPE_CIE2017",
    "ROOT_RESOURCES_CIE2017",
    "DataColorimetry_TCS_CIE2017",
    "ColourRendering_Specification_CIE2017",
    "colour_fidelity_index_CIE2017",
    "load_TCS_CIE2017",
    "tcs_colorimetry_data",
    "delta_E_to_R_f",
]

SPECTRAL_SHAPE_CIE2017: SpectralShape = SpectralShape(380, 780, 1)
"""
Spectral shape for *CIE 2017 Colour Fidelity Index* (CFI)
standard.
"""

ROOT_RESOURCES_CIE2017: str = os.path.join(os.path.dirname(__file__), "datasets")
"""*CIE 2017 Colour Fidelity Index* resources directory."""

_CACHE_TCS_CIE2017: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_TCS_CIE2017"
)


@dataclass
class DataColorimetry_TCS_CIE2017:
    """
    Store colorimetry data for *test colour samples* used in CIE 2017
    colour fidelity calculations.

    This dataclass encapsulates the colorimetric properties of test colour
    samples as specified by CIE 2017, including their tristimulus values,
    colour appearance model specifications, and perceptual colour
    coordinates in both cylindrical and rectangular representations.

    Attributes
    ----------
    name
        Identifier(s) for the test colour sample(s).
    XYZ
        CIE XYZ tristimulus values of the test colour samples.
    CAM
        CIECAM02 colour appearance model specification containing the
        complete appearance correlates.
    JMh
        Perceptual colour coordinates in cylindrical representation with
        *lightness* (J), *colourfulness* (M), and *hue angle* (h).
    Jpapbp
        Perceptual colour coordinates in rectangular representation with
        *lightness* (J) and opponent colour dimensions (a', b').
    """

    name: str | list[str]
    XYZ: NDArrayFloat
    CAM: CAM_Specification_CIECAM02
    JMh: NDArrayFloat
    Jpapbp: NDArrayFloat


@dataclass
class ColourRendering_Specification_CIE2017:
    """
    Define the *CIE 2017 Colour Fidelity Index* (CFI) colour quality
    specification.

    Parameters
    ----------
    name
        Name of the test spectral distribution.
    sd_reference
        Spectral distribution of the reference illuminant.
    R_f
        *CIE 2017 Colour Fidelity Index* (CFI) :math:`R_f`.
    R_s
        Individual *colour fidelity indexes* data for each sample.
    CCT
        Correlated colour temperature :math:`T_{cp}`.
    D_uv
        Distance from the Planckian locus :math:`\\Delta_{uv}`.
    colorimetry_data
        Colorimetry data for the test and reference computations.
    delta_E_s
        Colour shifts of samples.
    """

    name: str
    sd_reference: SpectralDistribution
    R_f: float
    R_s: NDArrayFloat
    CCT: float
    D_uv: float
    colorimetry_data: Tuple[DataColorimetry_TCS_CIE2017, DataColorimetry_TCS_CIE2017]
    delta_E_s: NDArrayFloat


@typing.overload
def colour_fidelity_index_CIE2017(
    sd_test: SpectralDistribution, additional_data: Literal[False] = False
) -> float: ...


@typing.overload
def colour_fidelity_index_CIE2017(
    sd_test: SpectralDistribution, additional_data: Literal[True]
) -> ColourRendering_Specification_CIE2017: ...


@typing.overload
def colour_fidelity_index_CIE2017(
    sd_test: MultiSpectralDistributions,
    additional_data: Literal[False] = False,
) -> NDArrayFloat: ...


def colour_fidelity_index_CIE2017(
    sd_test: SpectralDistribution | MultiSpectralDistributions,
    additional_data: bool = False,
) -> float | NDArrayFloat | ColourRendering_Specification_CIE2017:
    """
    Compute the *CIE 2017 Colour Fidelity Index* (CFI) :math:`R_f` of the
    specified spectral distribution.

    Parameters
    ----------
    sd_test
        Test spectral distribution. A
        :class:`colour.MultiSpectralDistributions` of ``N`` test
        illuminants is also accepted, in which case ``additional_data``
        must be ``False`` and the return value is a :class:`numpy.ndarray`
        of ``N`` :math:`R_f` values.
    additional_data
        Whether to output additional data.

    Returns
    -------
    :class:`float`, :class:`numpy.ndarray` or \
:class:`colour.quality.ColourRendering_Specification_CIE2017`
        *CIE 2017 Colour Fidelity Index* (CFI) :math:`R_f`.

    References
    ----------
    :cite:`CIETC1-902017`

    Examples
    --------
    >>> from colour.colorimetry import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS["FL2"]
    >>> colour_fidelity_index_CIE2017(sd)  # doctest: +ELLIPSIS
    np.float64(70.1208244...)
    """

    is_msds = isinstance(sd_test, MultiSpectralDistributions)

    if is_msds and additional_data:
        error = (
            '"additional_data=True" is not supported when "sd_test" is a '
            '"MultiSpectralDistributions" instance.'
        )
        raise NotImplementedError(error)

    if sd_test.shape.interval > 5:
        error = (
            "Test spectral distribution interval is greater than "
            "5nm which is the maximum recommended value "
            'for computing the "CIE 2017 Colour Fidelity Index"!'
        )

        raise ValueError(error)

    shape = SpectralShape(
        SPECTRAL_SHAPE_CIE2017.start,
        SPECTRAL_SHAPE_CIE2017.end,
        sd_test.shape.interval,
    )

    if sd_test.shape.start > 380 or sd_test.shape.end < 780:
        usage_warning(
            "Test spectral distribution shape does not span the "
            "recommended 380-780nm range, missing values will be "
            "filled with zeros!"
        )

        # NOTE: "CIE 2017 Colour Fidelity Index" standard recommends filling
        # missing values with zeros.
        sd_test = sd_test.copy()
        sd_test.extrapolator = Extrapolator
        sd_test.extrapolator_kwargs = {
            "method": "constant",
            "left": 0,
            "right": 0,
        }
        sd_test.align(shape=shape)

    if sd_test.shape.boundaries != shape.boundaries:
        sd_test.trim(shape)

    # NOTE: All computations except CCT calculation use the
    # "CIE 1964 10 Degree Standard Observer".
    cmfs_10 = reshape_msds(
        MSDS_CMFS["CIE 1964 10 Degree Standard Observer"], shape, copy=False
    )

    sds_tcs = load_TCS_CIE2017(shape)

    sd_test_values = sd_test.values
    xp = array_namespace(sds_tcs.values, sd_test_values)

    if is_msds:
        test_values = xp_matrix_transpose(
            xp_as_float_array(sd_test_values, xp=xp), xp=xp
        )
    else:
        test_values = xp_as_float_array(sd_test_values, xp=xp)[None, :]

    XYZ_test_2deg = msds_to_XYZ(test_values, method="Integration", shape=shape)
    CCT_Duv = uv_to_CCT_Ohno2013(
        UCS_to_uv(XYZ_to_UCS(XYZ_test_2deg)), start=1000, end=25000
    )
    CCT = CCT_Duv[..., 0]
    D_uv = CCT_Duv[..., 1]

    # ``CIE 2017 CFI`` 3-way reference: Y-normalised Planckian / daylight
    # mixture clipped over the ``[4000, 5000]`` K transition window
    # (:cite:`CIETC1-902017`, Section 4.2). ``planck_law`` squeezes its
    # output, so a single-CCT batch collapses to 1-D; the sample axis is
    # reinstated below. This vectorised mixture is the single source of the
    # reference illuminant; the ``additional_data`` branch wraps the
    # corresponding ``ref_values`` row in a spectral distribution.
    planckian = planck_law(shape.wavelengths * 1e-9, CCT) * 1e-9
    planckian_values = (
        planckian[None, :]
        if planckian.ndim == 1
        else xp_matrix_transpose(planckian, xp=xp)
    )
    # ``CCT_to_xy_CIE_D`` warns for any sample outside ``[4000, 25000]`` K
    # (matching the suppression in :mod:`colour.quality.cri` /
    # :mod:`colour.quality.cqs`); ``m = 0`` nulls the extrapolated daylight
    # so the leaked warning is spurious here.
    with suppress_warnings(colour_usage_warnings=True):
        daylight = CIE_illuminant_D_series(CCT_to_xy_CIE_D(CCT), shape=shape)
    daylight_values = (
        daylight[None, :]
        if daylight.ndim == 1
        else xp_matrix_transpose(daylight, xp=xp)
    )
    Y_planckian = msds_to_XYZ(planckian_values, method="Integration", shape=shape)[
        ..., 1:2
    ]
    Y_daylight = msds_to_XYZ(daylight_values, method="Integration", shape=shape)[
        ..., 1:2
    ]
    m = xp.clip((CCT - 4000) / 1000, 0, 1)[..., None]
    ref_values = (1 - m) * (planckian_values / Y_planckian) + m * (
        daylight_values / Y_daylight
    )

    irradiance_values = xp.stack([test_values, ref_values])
    XYZ_t = msds_to_XYZ(irradiance_values, cmfs_10, method="Integration", shape=shape)
    k = 100 / XYZ_t[..., 1:2]
    XYZ_w = k * XYZ_t
    irradiance_values = irradiance_values * k

    XYZ, specification, JMh, Jpapbp = _tcs_colorimetry_data(
        irradiance_values, XYZ_w, sds_tcs, cmfs_10
    )

    delta_E_s = euclidean_distance(Jpapbp[0], Jpapbp[1])

    R_s = delta_E_to_R_f(delta_E_s)
    R_f = delta_E_to_R_f(xp_average(delta_E_s, axis=-1, xp=xp))

    if is_msds:
        return R_f

    R_f_scalar = as_float_scalar(R_f[0])

    if not additional_data:
        return R_f_scalar

    sd_reference = SpectralDistribution(
        as_ndarray(ref_values[0]),
        shape.wavelengths,
        name=f"{int(CCT[0])}K CIE 2017 Reference Illuminant",
    )

    # Drop the size-1 batch axis from the rank-3 outputs to recover the
    # original (n_irradiances, n_TCS, *) layout.
    XYZ = XYZ[:, 0]
    JMh = JMh[:, 0]
    Jpapbp = Jpapbp[:, 0]
    specification = CAM_Specification_CIECAM02(
        **{
            name: (value[:, 0] if value is not None else None)
            for name, value in specification
        }
    )

    # ``as_float_array`` materialises the dataclass to *NumPy* via
    # ``__array__``, so the transpose stays on *NumPy* regardless of ``xp``.
    specification = np.transpose(as_float_array(specification), (0, 2, 1))
    specifications = [CAM_Specification_CIECAM02(*t) for t in specification]
    test_tcs_colorimetry_data = DataColorimetry_TCS_CIE2017(
        sds_tcs.display_labels,
        XYZ[0],
        specifications[0],
        JMh[0],
        Jpapbp[0],
    )
    reference_tcs_colorimetry_data = DataColorimetry_TCS_CIE2017(
        sds_tcs.display_labels,
        XYZ[1],
        specifications[1],
        JMh[1],
        Jpapbp[1],
    )
    return ColourRendering_Specification_CIE2017(
        sd_test.name,
        sd_reference,
        R_f_scalar,
        R_s[0],
        as_float_scalar(CCT[0]),
        as_float_scalar(D_uv[0]),
        (test_tcs_colorimetry_data, reference_tcs_colorimetry_data),
        delta_E_s[0],
    )


def load_TCS_CIE2017(shape: SpectralShape) -> MultiSpectralDistributions:
    """
    Load the *CIE 2017 Test Colour Samples* dataset appropriate for the
    specified spectral shape.

    The datasets are cached and will not be loaded again on subsequent
    calls to this definition.

    Parameters
    ----------
    shape
        Spectral shape of the tested illuminant.

    Returns
    -------
    :class:`colour.MultiSpectralDistributions`
        *CIE 2017 Test Colour Samples* dataset.

    Examples
    --------
    >>> sds_tcs = load_TCS_CIE2017(SpectralShape(380, 780, 5))
    >>> len(sds_tcs.labels)
    99
    """

    global _CACHE_TCS_CIE2017  # noqa: PLW0602

    interval = shape.interval

    attest(
        interval in (1, 5),
        "Spectral shape interval must be either 1nm or 5nm!",
    )

    filename = f"tcs_cfi2017_{as_int_scalar(interval)}_nm.csv.gz"

    if is_caching_enabled() and filename in _CACHE_TCS_CIE2017:
        return _CACHE_TCS_CIE2017[filename]

    data = np.genfromtxt(
        str(os.path.join(ROOT_RESOURCES_CIE2017, filename)), delimiter=","
    )
    labels = [f"TCS{i} (CIE 2017)" for i in range(99)]

    tcs = MultiSpectralDistributions(data[:, 1:], data[:, 0], labels)

    _CACHE_TCS_CIE2017[filename] = tcs

    return tcs


def _tcs_colorimetry_data(
    irradiance_values: NDArrayFloat,
    XYZ_w: NDArrayFloat,
    sds_tcs: MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions,
) -> Tuple[NDArrayFloat, CAM_Specification_CIECAM02, NDArrayFloat, NDArrayFloat]:
    """
    Compute the *test colour samples* colorimetry arrays under the specified
    irradiance(s) and reference white point(s) for the *CIE 2017 Colour
    Fidelity Index* (CFI) computations.

    Parameters
    ----------
    irradiance_values
        Per-illuminant normalised irradiance spectral values of shape
        ``(..., n_wavelengths)``.
    XYZ_w
        Per-illuminant *CIE XYZ* tristimulus values of shape ``(..., 3)``.
    sds_tcs
        *Test colour samples* spectral reflectance distributions.
    cmfs
        Standard observer colour matching functions.

    Returns
    -------
    :class:`tuple`
        ``(XYZ, specification, JMh, Jpapbp)`` arrays, each with leading
        shape ``(..., n_test_colour_samples)``.
    """

    sds_tcs_values_raw = sds_tcs.values
    xp = array_namespace(irradiance_values, sds_tcs_values_raw)

    sds_tcs_values = xp_as_float_array(
        sds_tcs_values_raw, xp=xp, like=irradiance_values
    )
    sds_tcs_t = (
        xp_matrix_transpose(sds_tcs_values, xp=xp) * irradiance_values[..., None, :]
    )

    XYZ = msds_to_XYZ(
        sds_tcs_t,
        cmfs,
        method="Integration",
        shape=sds_tcs.shape,
    )
    specification = XYZ_to_CIECAM02(
        XYZ,
        XYZ_w[..., None, :],
        100,  # L_A
        20,  # Y_b
        VIEWING_CONDITIONS_CIECAM02["Average"],
        discount_illuminant=True,
        compute_H=False,
    )

    JMh = tstack(
        [
            cast("NDArrayFloat", specification.J),
            cast("NDArrayFloat", specification.M),
            cast("NDArrayFloat", specification.h),
        ]
    )
    Jpapbp = JMh_CIECAM02_to_CAM02UCS(JMh)

    return XYZ, specification, JMh, Jpapbp


def tcs_colorimetry_data(
    sd_irradiance: SpectralDistribution | List[SpectralDistribution],
    sds_tcs: MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions,
) -> Tuple[DataColorimetry_TCS_CIE2017, ...]:
    """
    Compute the *test colour samples* colorimetry data under the specified
    test light source or reference illuminant spectral distribution for the
    *CIE 2017 Colour Fidelity Index* (CFI) computations.

    Parameters
    ----------
    sd_irradiance
        Test light source or reference illuminant spectral distribution,
        i.e., the irradiance emitter.
    sds_tcs
        *Test colour samples* spectral reflectance distributions.
    cmfs
        Standard observer colour matching functions.

    Returns
    -------
    :class:`tuple`
        *Test colour samples* colorimetry data under the specified test
        light source or reference illuminant spectral distribution.

    Examples
    --------
    >>> from colour.colorimetry import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS["FL2"]
    >>> shape = SpectralShape(380, 780, 5)
    >>> cmfs = MSDS_CMFS["CIE 1964 10 Degree Standard Observer"].copy().align(shape)
    >>> test_tcs_colorimetry_data = tcs_colorimetry_data(
    ...     sd, load_TCS_CIE2017(shape), cmfs
    ... )
    >>> len(test_tcs_colorimetry_data)
    1
    """

    if isinstance(sd_irradiance, SpectralDistribution):
        sd_irradiance = [sd_irradiance]

    xp = array_namespace(sds_tcs.values, sd_irradiance[0].values)

    irradiance_values = xp.stack(
        [xp_as_float_array(sd.values, xp=xp) for sd in sd_irradiance]
    )
    XYZ_t = msds_to_XYZ(
        irradiance_values,
        cmfs,
        method="Integration",
        shape=sd_irradiance[0].shape,
    )
    k = 100 / XYZ_t[..., 1:2]
    XYZ_w = k * XYZ_t
    irradiance_values = irradiance_values * k

    XYZ, specification, JMh, Jpapbp = _tcs_colorimetry_data(
        irradiance_values, XYZ_w, sds_tcs, cmfs
    )

    # ``as_float_array`` materialises the dataclass to *NumPy* via
    # ``__array__``, so the transpose stays on *NumPy* regardless of ``xp``.
    specification = np.transpose(as_float_array(specification), (0, 2, 1))
    specifications = [CAM_Specification_CIECAM02(*t) for t in specification]

    return tuple(
        [
            DataColorimetry_TCS_CIE2017(
                sds_tcs.display_labels,
                XYZ[sd_idx],
                specifications[sd_idx],
                JMh[sd_idx],
                Jpapbp[sd_idx],
            )
            for sd_idx in range(len(sd_irradiance))
        ]
    )


def delta_E_to_R_f(delta_E: ArrayLike) -> NDArrayFloat:
    """
    Convert colour-appearance difference to *CIE 2017 Colour Fidelity Index*
    (CFI) :math:`R_f` value.

    Parameters
    ----------
    delta_E
        Euclidean distance between two colours in *CAM02-UCS* colourspace.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding *CIE 2017 Colour Fidelity Index* (CFI) :math:`R_f`
        value.
    """

    delta_E = as_float_array(delta_E)

    xp = array_namespace(delta_E)

    c_f = 6.73

    return as_float(10 * xp.log1p(xp.exp((100 - c_f * delta_E) / 10)))
