"""
Colour Quality Scale
====================

Define the *Colour Quality Scale* (CQS) computation objects.

-   :class:`colour.quality.ColourRendering_Specification_CQS`
-   :func:`colour.colour_quality_scale`

References
----------
-   :cite:`Davis2010a` : Davis, W., & Ohno, Y. (2010). Color quality scale.
    Optical Engineering, 49(3), 033602. doi:10.1117/1.3360335
-   :cite:`Ohno2008a` : Ohno, Yoshiro, & Davis, W. (2008). NIST CQS simulation
    (Version 7.4) [Computer software].
    https://drive.google.com/file/d/1PsuU6QjUJjCX6tQyCud6ul2Tbs8rYWW9/view?\
usp=sharing
-   :cite:`Ohno2013` : Ohno, Yoshiro, & Davis, W. (2008). NIST CQS simulation
    (Version 7.4) [Computer software].
    https://drive.google.com/file/d/1PsuU6QjUJjCX6tQyCud6ul2Tbs8rYWW9/view?\
usp=sharing
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from colour.adaptation import chromatic_adaptation_VonKries
from colour.algebra import euclidean_distance, sdiv, sdiv_mode
from colour.colorimetry import (
    CCS_ILLUMINANTS,
    MSDS_CMFS,
    SPECTRAL_SHAPE_DEFAULT,
    CIE_illuminant_D_series,
    MultiSpectralDistributions,
    SpectralDistribution,
    msds_to_XYZ,
    planck_law,
    reshape_msds,
    reshape_sd,
    sd_to_XYZ,
)

if typing.TYPE_CHECKING:
    from colour.hints import (
        ArrayLike,
        Dict,
        Literal,
        NDArrayFloat,
        Tuple,
    )

from colour.models import Lab_to_LCHab  # pyright: ignore
from colour.models import UCS_to_uv, XYZ_to_Lab, XYZ_to_UCS, XYZ_to_xy, xy_to_XYZ
from colour.quality.datasets.vs import INDEXES_TO_NAMES_VS, SDS_VS
from colour.temperature import CCT_to_xy_CIE_D, uv_to_CCT_Ohno2013
from colour.utilities import (
    array_namespace,
    as_float_array,
    as_float_scalar,
    domain_range_scale,
    suppress_warnings,
    tsplit,
    validate_method,
    xp_as_float_array,
    xp_average,
    xp_matrix_transpose,
)
from colour.utilities.documentation import DocstringTuple, is_documentation_building

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "GAMUT_AREA_D65",
    "DataColorimetry_VS",
    "DataColourQualityScale_VS",
    "ColourRendering_Specification_CQS",
    "COLOUR_QUALITY_SCALE_METHODS",
    "colour_quality_scale",
    "gamut_area",
    "vs_colorimetry_data",
    "CCT_factor",
    "scale_conversion",
    "delta_E_RMS",
    "colour_quality_scales",
]

GAMUT_AREA_D65: int = 8210
"""Gamut area for *CIE Illuminant D Series D65*."""


@dataclass
class DataColorimetry_VS:
    """
    Store colorimetry data for *VS test colour samples*.

    This dataclass encapsulates the colorimetric measurements and derived
    values for Visual Spectrum (VS) test colour samples used in colour
    quality evaluation.

    Attributes
    ----------
    name
        Sample identifier or designation.
    XYZ
        Tristimulus values under the test illuminant.
    Lab
        *CIE L\\*a\\*b\\** colour space coordinates.
    C
        Chroma values calculated from the *CIE L\\*a\\*b\\** coordinates.
    """

    name: str
    XYZ: NDArrayFloat
    Lab: NDArrayFloat
    C: NDArrayFloat


@dataclass
class DataColourQualityScale_VS:
    """
    Store colour quality scale data for *VS test colour samples*.

    This dataclass encapsulates the colour quality metrics computed for VS
    (Visual Samples) test colour samples, including quality assessment and
    colour difference measurements used in colour rendering evaluations.

    Attributes
    ----------
    name
        Identifier or descriptor for the test colour sample.
    Q_a
        Colour quality scale value for the sample.
    D_C_ab
        Chroma difference in *CIE L\\*a\\*b\\** colourspace.
    D_E_ab
        Total colour difference in *CIE L\\*a\\*b\\** colourspace.
    """

    name: str
    Q_a: float
    D_C_ab: float
    D_E_ab: float
    D_Ep_ab: float


@dataclass
class ColourRendering_Specification_CQS:
    """
    Define the *Colour Quality Scale* (CQS) colour rendering (quality)
    specification.

    Parameters
    ----------
    name
        Name of the test spectral distribution.
    Q_a
        Colour quality scale :math:`Q_a`.
    Q_f
        Colour fidelity scale :math:`Q_f` intended to evaluate the
        fidelity of object colour appearances (compared to the reference
        illuminant of the same correlated colour temperature and
        illuminance).
    Q_p
        Colour preference scale :math:`Q_p` similar to colour quality
        scale :math:`Q_a` but placing additional weight on preference of
        object colour appearance, set to *None* in *NIST CQS 9.0* method.
        This metric is based on the notion that increases in chroma are
        generally preferred and should be rewarded.
    Q_g
        Gamut area scale :math:`Q_g` representing the relative gamut
        formed by the (:math:`a^*`, :math:`b^*`) coordinates of the 15
        samples illuminated by the test light source in the
        *CIE L\\*a\\*b\\** object colourspace.
    Q_d
        Relative gamut area scale :math:`Q_d`, set to *None* in
        *NIST CQS 9.0* method.
    Q_as
        Individual *Colour Quality Scale* (CQS) data for each sample.
    colorimetry_data
        Colorimetry data for the test and reference computations.

    References
    ----------
    :cite:`Davis2010a`, :cite:`Ohno2008a`, :cite:`Ohno2013`
    """

    name: str
    Q_a: float
    Q_f: float
    Q_p: float | None
    Q_g: float
    Q_d: float | None
    Q_as: Dict[int, DataColourQualityScale_VS]
    colorimetry_data: Tuple[
        Tuple[DataColorimetry_VS, ...], Tuple[DataColorimetry_VS, ...]
    ]


COLOUR_QUALITY_SCALE_METHODS: tuple = ("NIST CQS 7.4", "NIST CQS 9.0")
if is_documentation_building():  # pragma: no cover
    COLOUR_QUALITY_SCALE_METHODS = DocstringTuple(COLOUR_QUALITY_SCALE_METHODS)
    COLOUR_QUALITY_SCALE_METHODS.__doc__ = """
Supported *Colour Quality Scale* (CQS) computation methods.

References
----------
:cite:`Davis2010a`, :cite:`Ohno2008a`, :cite:`Ohno2013`
"""


@typing.overload
def colour_quality_scale(
    sd_test: SpectralDistribution,
    additional_data: Literal[False] = False,
    method: Literal["NIST CQS 7.4", "NIST CQS 9.0"] | str = ...,
) -> float: ...


@typing.overload
def colour_quality_scale(
    sd_test: SpectralDistribution,
    additional_data: Literal[True],
    method: Literal["NIST CQS 7.4", "NIST CQS 9.0"] | str = ...,
) -> ColourRendering_Specification_CQS: ...


@typing.overload
def colour_quality_scale(
    sd_test: MultiSpectralDistributions,
    additional_data: Literal[False] = False,
    method: Literal["NIST CQS 7.4", "NIST CQS 9.0"] | str = ...,
) -> NDArrayFloat: ...


def colour_quality_scale(
    sd_test: SpectralDistribution | MultiSpectralDistributions,
    additional_data: bool = False,
    method: Literal["NIST CQS 7.4", "NIST CQS 9.0"] | str = "NIST CQS 9.0",
) -> float | NDArrayFloat | ColourRendering_Specification_CQS:
    """
    Compute the *Colour Quality Scale* (CQS) of the specified spectral
    distribution using the specified method.

    Parameters
    ----------
    sd_test
        Test spectral distribution. A
        :class:`colour.MultiSpectralDistributions` of ``N`` test
        illuminants is also accepted, in which case ``additional_data``
        must be ``False`` and the return value is a :class:`numpy.ndarray`
        of ``N`` :math:`Q_a` values.
    additional_data
        Whether to output additional data.
    method
        Computation method.

    Returns
    -------
    :class:`float`, :class:`numpy.ndarray` or \
    :class:`colour.quality.ColourRendering_Specification_CQS`
        *Colour Quality Scale* (CQS).

    Notes
    -----
    -   The score is piecewise differentiable with respect to the test
        spectral values while the reference-illuminant branch remains
        unchanged. Its derivative is undefined where that selection changes.

    References
    ----------
    :cite:`Davis2010a`, :cite:`Ohno2008a`, :cite:`Ohno2013`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS["FL2"]
    >>> colour_quality_scale(sd)  # doctest: +ELLIPSIS
    np.float64(64.1118220...)
    """

    method = validate_method(method, tuple(COLOUR_QUALITY_SCALE_METHODS))

    cmfs = reshape_msds(
        MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
        SPECTRAL_SHAPE_DEFAULT,
        copy=False,
    )

    shape = cmfs.shape
    vs_sds = {
        sd.name: reshape_sd(sd, shape, copy=False) for sd in SDS_VS[method].values()
    }

    is_msds = isinstance(sd_test, MultiSpectralDistributions)
    if is_msds and additional_data:
        error = (
            '"additional_data=True" is not supported when "sd_test" is a '
            '"MultiSpectralDistributions" instance.'
        )
        raise NotImplementedError(error)

    if is_msds:
        sd_test = reshape_msds(sd_test, shape, copy=False)
        sd_test_values = sd_test.values
        xp = array_namespace(sd_test_values)
        test_values = xp_matrix_transpose(
            xp_as_float_array(sd_test_values, xp=xp), xp=xp
        )
    else:
        sd_test = reshape_sd(sd_test, shape, copy=False)
        sd_test_values = sd_test.values
        xp = array_namespace(sd_test_values)
        test_values = xp_as_float_array(sd_test_values, xp=xp)[None, :]

    with domain_range_scale("1"):
        XYZ = (
            msds_to_XYZ(test_values, cmfs, method="Integration", shape=shape)
            if is_msds
            else sd_to_XYZ(sd_test, cmfs)[None, :]
        )

    uv = UCS_to_uv(XYZ_to_UCS(XYZ))
    CCT = uv_to_CCT_Ohno2013(uv)[..., 0]

    # ``planck_law`` squeezes its output, so a single-CCT batch collapses
    # to 1-D; the sample axis is reinstated below.
    planckian = planck_law(shape.wavelengths * 1e-9, CCT) * 1e-9
    planckian_values = (
        planckian[None, :]
        if planckian.ndim == 1
        else xp_matrix_transpose(planckian, xp=xp)
    )
    # See :mod:`colour.quality.cri` for the warning-suppression rationale.
    with suppress_warnings(colour_usage_warnings=True):
        daylight = CIE_illuminant_D_series(CCT_to_xy_CIE_D(CCT), shape=shape)
    daylight_values = (
        daylight[None, :]
        if daylight.ndim == 1
        else xp_matrix_transpose(daylight, xp=xp)
    )
    ref_values = xp.where(CCT[..., None] < 5000, planckian_values, daylight_values)

    test_names, test_XYZ, test_Lab, test_C = _vs_colorimetry_data(
        test_values, ref_values, vs_sds, cmfs, chromatic_adaptation=True
    )
    ref_names, ref_XYZ, ref_Lab, ref_C = _vs_colorimetry_data(
        ref_values, ref_values, vs_sds, cmfs
    )

    D_C_ab = test_C - ref_C
    D_E_ab = euclidean_distance(test_Lab, ref_Lab)
    # ``D_E_ab ** 2 >= D_C_ab ** 2`` by Pythagoras; the inner ``xp.where``
    # guards ``xp.sqrt`` against floating-point cancellation, not a colour
    # data clamp.
    D_Ep_squared = D_E_ab**2 - D_C_ab**2
    D_Ep_ab = xp.where(
        D_C_ab > 0,
        xp.sqrt(xp.where(D_Ep_squared > 0, D_Ep_squared, xp.zeros_like(D_Ep_squared))),
        D_E_ab,
    )

    if method == "nist cqs 9.0":
        CCT_f = xp.ones_like(D_C_ab[..., 0])
        scaling_f = 3.2
    else:
        ref_XYZ_white = msds_to_XYZ(
            ref_values, cmfs, method="Integration", shape=cmfs.shape
        )
        with sdiv_mode():
            ref_XYZ_white_n = sdiv(ref_XYZ_white, ref_XYZ_white[..., 1:2])
        CCT_f = _CCT_factor(ref_XYZ, ref_XYZ_white_n)
        scaling_f = 3.104

    D_E_RMS = xp.sqrt(xp_average(D_E_ab**2, axis=-1, xp=xp))
    D_Ep_RMS = xp.sqrt(xp_average(D_Ep_ab**2, axis=-1, xp=xp))

    Q_a = scale_conversion(D_Ep_RMS, CCT_f, scaling_f)

    scaling_f_Q_f = 2.93 * 1.0343 if method == "nist cqs 9.0" else 2.928
    Q_f = scale_conversion(D_E_RMS, CCT_f, scaling_f_Q_f)

    G_t = gamut_area(test_Lab)
    G_r = gamut_area(ref_Lab)
    Q_g = G_t / GAMUT_AREA_D65 * 100

    Q_p: NDArrayFloat | None
    Q_d: NDArrayFloat | None
    if method == "nist cqs 9.0":
        Q_p = None
        Q_d = None
    else:
        p_delta_C = xp_average(
            xp.where(D_C_ab > 0, D_C_ab, xp.zeros_like(D_C_ab)),
            axis=-1,
            xp=xp,
        )
        Q_p = 100 - 3.6 * (D_Ep_RMS - p_delta_C)
        Q_d = G_t / G_r * CCT_f * 100

    if is_msds:
        return Q_a

    Q_a_scalar = as_float_scalar(Q_a[0])

    if additional_data:
        Q_f_scalar = as_float_scalar(Q_f[0])
        Q_p_scalar = as_float_scalar(Q_p[0]) if Q_p is not None else None
        Q_g_scalar = as_float_scalar(Q_g[0])
        Q_d_scalar = as_float_scalar(Q_d[0]) if Q_d is not None else None

        CCT_f_scalar = as_float_scalar(CCT_f[0])

        Q_as: Dict[int, DataColourQualityScale_VS] = {}
        for i, name in enumerate(test_names):
            D_C_i = as_float_scalar(D_C_ab[0, i])
            D_E_i = as_float_scalar(D_E_ab[0, i])
            D_Ep_i = as_float_scalar(D_Ep_ab[0, i])
            Q_a_i = float(scale_conversion(D_Ep_i, CCT_f_scalar, scaling_f))
            Q_as[i + 1] = DataColourQualityScale_VS(name, Q_a_i, D_C_i, D_E_i, D_Ep_i)

        test_data = tuple(
            DataColorimetry_VS(name, test_XYZ[0, i], test_Lab[0, i], test_C[0, i])
            for i, name in enumerate(test_names)
        )
        ref_data = tuple(
            DataColorimetry_VS(name, ref_XYZ[0, i], ref_Lab[0, i], ref_C[0, i])
            for i, name in enumerate(ref_names)
        )

        return ColourRendering_Specification_CQS(
            sd_test.name,
            Q_a_scalar,
            Q_f_scalar,
            Q_p_scalar,
            Q_g_scalar,
            Q_d_scalar,
            Q_as,
            (test_data, ref_data),
        )

    return Q_a_scalar


def gamut_area(Lab: ArrayLike) -> NDArrayFloat:
    """
    Compute the gamut area :math:`G` covered by the specified
    *CIE L\\*a\\*b\\** colourspace matrices.

    Parameters
    ----------
    Lab
        *CIE L\\*a\\*b\\** colourspace matrices.

    Returns
    -------
    :class:`float`
        Gamut area :math:`G`.

    Examples
    --------
    >>> import numpy as np
    >>> Lab = [
    ...     np.array([39.94996006, 34.59018231, -19.86046321]),
    ...     np.array([38.88395498, 21.44348519, -34.87805301]),
    ...     np.array([36.60576301, 7.06742454, -43.21461177]),
    ...     np.array([46.60142558, -15.90481586, -34.64616865]),
    ...     np.array([56.50196523, -29.54655550, -20.50177194]),
    ...     np.array([55.73912101, -43.39520959, -5.08956953]),
    ...     np.array([56.20776870, -53.68997662, 20.21134410]),
    ...     np.array([66.16683122, -38.64600327, 42.77396631]),
    ...     np.array([76.72952110, -23.92148210, 61.04740432]),
    ...     np.array([82.85370708, -3.98679065, 75.43320144]),
    ...     np.array([69.26458861, 13.11066359, 68.83858372]),
    ...     np.array([69.63154351, 28.24532497, 59.45609803]),
    ...     np.array([61.26281449, 40.87950839, 44.97606172]),
    ...     np.array([41.62567821, 57.34129516, 27.46718170]),
    ...     np.array([40.52565174, 48.87449192, 3.45121680]),
    ... ]
    >>> gamut_area(Lab)  # doctest: +ELLIPSIS
    np.float64(8335.9482018...)
    """

    Lab = as_float_array(Lab)

    xp = array_namespace(Lab)

    Lab_s = xp.roll(Lab, shift=-1, axis=-2)

    A = xp.linalg.vector_norm(Lab[..., 1:3], axis=-1)
    B = xp.linalg.vector_norm(Lab_s[..., 1:3], axis=-1)
    C = xp.linalg.vector_norm(Lab_s[..., 1:3] - Lab[..., 1:3], axis=-1)
    t = (A + B + C) / 2
    S = xp.sqrt(t * (t - A) * (t - B) * (t - C))

    return xp.sum(S, axis=-1)


def _vs_colorimetry_data(
    t_values: NDArrayFloat,
    r_values: NDArrayFloat,
    sds_vs: Dict[str, SpectralDistribution],
    cmfs: MultiSpectralDistributions,
    chromatic_adaptation: bool = False,
) -> Tuple[list[str], NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """
    Compute the *VS test colour samples* colorimetry arrays in a single
    vectorised pass over an arbitrary leading shape of test/reference
    irradiance pairs.

    Parameters
    ----------
    t_values, r_values
        Test and reference irradiance values of shape
        ``(..., n_wavelengths)``.

    Returns
    -------
    :class:`tuple`
        ``(names, XYZ_vs, Lab_vs, C_vs)`` with leading shape
        ``(..., n_test_colour_samples)``.
    """

    XYZ_t = msds_to_XYZ(t_values, cmfs, method="Integration", shape=cmfs.shape)

    with sdiv_mode():
        XYZ_t_n = sdiv(XYZ_t, XYZ_t[..., 1:2])

    XYZ_r = msds_to_XYZ(r_values, cmfs, method="Integration", shape=cmfs.shape)

    with sdiv_mode():
        XYZ_r_n = sdiv(XYZ_r, XYZ_r[..., 1:2])

    xy_r = XYZ_to_xy(XYZ_r_n)

    names: list[str] = []
    vs_values_list = []
    for _key, value in sorted(INDEXES_TO_NAMES_VS.items()):
        if value not in sds_vs:
            continue
        names.append(sds_vs[value].name)
        vs_values_list.append(sds_vs[value].values)

    xp = array_namespace(XYZ_t, t_values)
    vs_values = xp.stack(
        [xp_as_float_array(values, xp=xp, like=XYZ_t) for values in vs_values_list]
    )

    # Vectorised :math:`XYZ_{vs}` across the VS test colour samples; the
    # ``1 / Y_t`` factor recovers the ``domain_range_scale("1")``
    # reflectance-under-illuminant scale.
    sds_vs_t = vs_values * t_values[..., None, :]
    XYZ_vs = (
        msds_to_XYZ(
            sds_vs_t,
            cmfs,
            method="Integration",
            shape=cmfs.shape,
        )
        / XYZ_t[..., 1:2, None]
    )

    if chromatic_adaptation:
        XYZ_vs = chromatic_adaptation_VonKries(
            XYZ_vs,
            XYZ_t_n[..., None, :],
            XYZ_r_n[..., None, :],
            transform="CMCCAT2000",
        )

    Lab_vs = XYZ_to_Lab(XYZ_vs, illuminant=xy_r[..., None, :])
    _L_vs, C_vs, _Hab = tsplit(Lab_to_LCHab(Lab_vs))

    return names, XYZ_vs, Lab_vs, C_vs


def vs_colorimetry_data(
    sd_test: SpectralDistribution,
    sd_reference: SpectralDistribution,
    sds_vs: Dict[str, SpectralDistribution],
    cmfs: MultiSpectralDistributions,
    chromatic_adaptation: bool = False,
) -> Tuple[DataColorimetry_VS, ...]:
    """
    Compute the *VS test colour samples* colorimetry data.

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    sd_reference
        Reference spectral distribution.
    sds_vs
        *VS test colour samples* spectral reflectance distributions.
    cmfs
        Standard observer colour matching functions.
    chromatic_adaptation
        Whether to perform chromatic adaptation.

    Returns
    -------
    :class:`tuple`
        *VS test colour samples* colorimetry data.
    """

    xp = array_namespace(sd_test.values)
    names, XYZ_vs, Lab_vs, C_vs = _vs_colorimetry_data(
        xp_as_float_array(sd_test.values, xp=xp),
        xp_as_float_array(sd_reference.values, xp=xp),
        sds_vs,
        cmfs,
        chromatic_adaptation,
    )

    return tuple(
        DataColorimetry_VS(name, XYZ_vs[i], Lab_vs[i], C_vs[i])
        for i, name in enumerate(names)
    )


def CCT_factor(
    reference_data: Tuple[DataColorimetry_VS, ...], XYZ_r: ArrayLike
) -> float:
    """
    Compute the correlated colour temperature factor that penalizes lamps
    with extremely low correlated colour temperatures.

    Parameters
    ----------
    reference_data
        Reference colorimetry data.
    XYZ_r
        *CIE XYZ* tristimulus values for reference.

    Returns
    -------
    :class:`float`
        Correlated colour temperature factor.
    """

    XYZ_vs = as_float_array(
        [colorimetry_data.XYZ for colorimetry_data in reference_data]
    )

    return as_float_scalar(_CCT_factor(XYZ_vs, as_float_array(XYZ_r)))


def _CCT_factor(XYZ_vs: NDArrayFloat, XYZ_r: NDArrayFloat) -> NDArrayFloat:
    """
    Compute the correlated colour temperature factor for arbitrary leading
    batch shape.

    Parameters
    ----------
    XYZ_vs
        Reference VS sample tristimulus values of shape
        ``(..., n_test_colour_samples, 3)``.
    XYZ_r
        Reference white tristimulus values of shape ``(..., 3)``.
    """

    xy_w = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    XYZ_w = xy_to_XYZ(xy_w)

    Lab = XYZ_to_Lab(
        chromatic_adaptation_VonKries(
            XYZ_vs,
            XYZ_r[..., None, :],
            XYZ_w,
            transform="CMCCAT2000",
        ),
        illuminant=xy_w,
    )

    G_r = gamut_area(Lab) / GAMUT_AREA_D65

    xp = array_namespace(G_r)

    return xp.minimum(G_r, xp.ones_like(G_r))


def scale_conversion(
    D_E_ab: ArrayLike, CCT_f: ArrayLike, scaling_f: float
) -> NDArrayFloat:
    """
    Compute the *Colour Quality Scale* (CQS) for the specified
    :math:`\\Delta E_{ab}` value and correlated colour temperature
    penalizing factor.

    Parameters
    ----------
    D_E_ab
        :math:`\\Delta E_{ab}` value.
    CCT_f
        Correlated colour temperature penalizing factor.
    scaling_f
        Scaling factor constant.

    Returns
    -------
    :class:`numpy.ndarray`
        *Colour Quality Scale* (CQS).
    """

    D_E_ab = as_float_array(D_E_ab)

    xp = array_namespace(D_E_ab)

    return as_float_array(
        10 * xp.log1p(xp.exp((100 - scaling_f * D_E_ab) / 10)) * CCT_f
    )


def delta_E_RMS(
    CQS_data: Dict[int, DataColourQualityScale_VS], attribute: str
) -> float:
    """
    Compute the root-mean-square average for the specified *Colour Quality
    Scale* (CQS) data using the specified colorimetry attribute.

    Parameters
    ----------
    CQS_data
        *Colour Quality Scale* (CQS) data.
    attribute
        Colorimetry data attribute to use for computing the
        root-mean-square average.

    Returns
    -------
    :class:`float`
        Root-mean-square average.
    """

    values = as_float_array(
        [getattr(sample_data, attribute) ** 2 for sample_data in CQS_data.values()]
    )

    xp = array_namespace(values)

    return as_float_scalar(xp.sqrt(1 / len(CQS_data) * xp.sum(values)))


def colour_quality_scales(
    test_data: Tuple[DataColorimetry_VS, ...],
    reference_data: Tuple[DataColorimetry_VS, ...],
    scaling_f: float,
    CCT_f: float,
) -> Dict[int, DataColourQualityScale_VS]:
    """
    Compute the *VS test colour samples* rendering scales.

    Parameters
    ----------
    test_data
        Test data for the VS colour samples.
    reference_data
        Reference data for the VS colour samples.
    scaling_f
        Scaling factor constant for normalizing the colour rendering
        scales.
    CCT_f
        Factor penalizing light sources with extremely low correlated
        colour temperatures.

    Returns
    -------
    :class:`dict`
        *VS test colour samples* colour rendering scales.
    """

    Q_as = {}

    xp = array_namespace(test_data[0].Lab)

    for i in range(len(test_data)):
        D_C_ab = as_float_scalar(test_data[i].C - reference_data[i].C)
        D_E_ab = as_float_scalar(
            euclidean_distance(test_data[i].Lab, reference_data[i].Lab)
        )
        D_Ep_ab_arr = as_float_array(D_E_ab**2 - D_C_ab**2)

        D_Ep_ab = as_float_scalar(xp.sqrt(D_Ep_ab_arr) if D_C_ab > 0 else D_E_ab)

        Q_a = float(scale_conversion(D_Ep_ab, CCT_f, scaling_f))
        Q_as[i + 1] = DataColourQualityScale_VS(
            test_data[i].name, Q_a, D_C_ab, D_E_ab, D_Ep_ab
        )

    return Q_as
