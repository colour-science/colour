"""
Colour Quality Scale
====================

Defines the *Colour Quality Scale* (CQS) computation objects:

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

import numpy as np
from dataclasses import dataclass

from colour.algebra import euclidean_distance
from colour.colorimetry import (
    CCS_ILLUMINANTS,
    MSDS_CMFS,
    MultiSpectralDistributions,
    SPECTRAL_SHAPE_DEFAULT,
    SpectralDistribution,
    reshape_msds,
    reshape_sd,
    sd_CIE_illuminant_D_series,
    sd_blackbody,
    sd_to_XYZ,
)
from colour.hints import (
    ArrayLike,
    Boolean,
    Dict,
    Floating,
    Integer,
    Literal,
    NDArray,
    Optional,
    Tuple,
    Union,
)
from colour.models import (
    Lab_to_LCHab,
    UCS_to_uv,
    XYZ_to_Lab,
    XYZ_to_UCS,
    XYZ_to_xy,
    xy_to_XYZ,
)
from colour.quality.datasets.vs import INDEXES_TO_NAMES_VS, SDS_VS
from colour.temperature import CCT_to_xy_CIE_D, uv_to_CCT_Ohno2013
from colour.adaptation import chromatic_adaptation_VonKries
from colour.utilities import (
    as_float_array,
    as_float_scalar,
    domain_range_scale,
    tsplit,
    validate_method,
)
from colour.utilities.documentation import (
    DocstringTuple,
    is_documentation_building,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "GAMUT_AREA_D65",
    "VS_ColorimetryData",
    "VS_ColourQualityScaleData",
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

GAMUT_AREA_D65: Integer = 8210
"""Gamut area for *CIE Illuminant D Series D65*."""


@dataclass
class VS_ColorimetryData:
    """Define the class storing *VS test colour samples* colorimetry data."""

    name: str
    XYZ: NDArray
    Lab: NDArray
    C: NDArray


@dataclass
class VS_ColourQualityScaleData:
    """
    Define the class storing *VS test colour samples* colour quality scale
    data.
    """

    name: str
    Q_a: Floating
    D_C_ab: Floating
    D_E_ab: Floating
    D_Ep_ab: Floating


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
        Colour fidelity scale :math:`Q_f` intended to evaluate the fidelity
        of object colour appearances (compared to the reference illuminant of
        the same correlated colour temperature and illuminance).
    Q_p
        Colour preference scale :math:`Q_p` similar to colour quality scale
        :math:`Q_a` but placing additional weight on preference of object
        colour appearance, set to *None* in *NIST CQS 9.0* method. This metric
        is based on the notion that increases in chroma are generally preferred
        and should be rewarded.
    Q_g
         Gamut area scale :math:`Q_g` representing the relative gamut formed
         by the (:math:`a^*`, :math:`b^*`) coordinates of the 15 samples
         illuminated by the test light source in the *CIE L\\*a\\*b\\** object
         colourspace.
    Q_d
        Relative gamut area scale :math:`Q_d`, set to *None* in *NIST CQS 9.0*
        method.
    Q_as
        Individual *Colour Quality Scale* (CQS) data for each sample.
    colorimetry_data
        Colorimetry data for the test and reference computations.

    References
    ----------
    :cite:`Davis2010a`, :cite:`Ohno2008a`,  :cite:`Ohno2013`
    """

    name: str
    Q_a: Floating
    Q_f: Floating
    Q_p: Optional[Floating]
    Q_g: Floating
    Q_d: Optional[Floating]
    Q_as: Dict[Integer, VS_ColourQualityScaleData]
    colorimetry_data: Tuple[
        Tuple[VS_ColorimetryData, ...], Tuple[VS_ColorimetryData, ...]
    ]


COLOUR_QUALITY_SCALE_METHODS: Tuple = ("NIST CQS 7.4", "NIST CQS 9.0")
if is_documentation_building():  # pragma: no cover
    COLOUR_QUALITY_SCALE_METHODS = DocstringTuple(COLOUR_QUALITY_SCALE_METHODS)
    COLOUR_QUALITY_SCALE_METHODS.__doc__ = """
Supported *Colour Quality Scale* (CQS) computation methods.

References
----------
:cite:`Davis2010a`, :cite:`Ohno2008a`, :cite:`Ohno2013`
"""


def colour_quality_scale(
    sd_test: SpectralDistribution,
    additional_data: Boolean = False,
    method: Union[
        Literal["NIST CQS 7.4", "NIST CQS 9.0"], str
    ] = "NIST CQS 9.0",
) -> Union[Floating, ColourRendering_Specification_CQS]:
    """
    Return the *Colour Quality Scale* (CQS) of given spectral distribution
    using given method.

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    additional_data
        Whether to output additional data.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.floating` or \
:class:`colour.quality.ColourRendering_Specification_CQS`
        *Colour Quality Scale* (CQS).

    References
    ----------
    :cite:`Davis2010a`, :cite:`Ohno2008a`, :cite:`Ohno2013`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS['FL2']
    >>> colour_quality_scale(sd)  # doctest: +ELLIPSIS
    64.1117031...
    """

    method = validate_method(method, COLOUR_QUALITY_SCALE_METHODS)

    # pylint: disable=E1102
    cmfs = reshape_msds(
        MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
        SPECTRAL_SHAPE_DEFAULT,
    )

    shape = cmfs.shape
    sd_test = reshape_sd(sd_test, shape)
    vs_sds = {sd.name: reshape_sd(sd, shape) for sd in SDS_VS[method].values()}

    with domain_range_scale("1"):
        XYZ = sd_to_XYZ(sd_test, cmfs)

    uv = UCS_to_uv(XYZ_to_UCS(XYZ))
    CCT, _D_uv = uv_to_CCT_Ohno2013(uv)

    if CCT < 5000:
        sd_reference = sd_blackbody(CCT, shape)
    else:
        xy = CCT_to_xy_CIE_D(CCT)
        sd_reference = sd_CIE_illuminant_D_series(xy)
        sd_reference.align(shape)

    test_vs_colorimetry_data = vs_colorimetry_data(
        sd_test, sd_reference, vs_sds, cmfs, chromatic_adaptation=True
    )

    reference_vs_colorimetry_data = vs_colorimetry_data(
        sd_reference, sd_reference, vs_sds, cmfs
    )

    CCT_f: Floating
    if method == "nist cqs 9.0":
        CCT_f = 1
        scaling_f = 3.2
    else:
        XYZ_r = sd_to_XYZ(sd_reference, cmfs)
        XYZ_r /= XYZ_r[1]
        CCT_f = CCT_factor(reference_vs_colorimetry_data, XYZ_r)
        scaling_f = 3.104

    Q_as = colour_quality_scales(
        test_vs_colorimetry_data,
        reference_vs_colorimetry_data,
        scaling_f,
        CCT_f,
    )

    D_E_RMS = delta_E_RMS(Q_as, "D_E_ab")
    D_Ep_RMS = delta_E_RMS(Q_as, "D_Ep_ab")

    Q_a = scale_conversion(D_Ep_RMS, CCT_f, scaling_f)

    if method == "nist cqs 9.0":
        scaling_f = 2.93 * 1.0343
    else:
        scaling_f = 2.928

    Q_f = scale_conversion(D_E_RMS, CCT_f, scaling_f)

    G_t = gamut_area(
        [vs_CQS_data.Lab for vs_CQS_data in test_vs_colorimetry_data]
    )
    G_r = gamut_area(
        [vs_CQS_data.Lab for vs_CQS_data in reference_vs_colorimetry_data]
    )

    Q_g = G_t / GAMUT_AREA_D65 * 100

    if method == "nist cqs 9.0":
        Q_p = Q_d = None
    else:
        p_delta_C = np.average(
            [
                sample_data.D_C_ab if sample_data.D_C_ab > 0 else 0
                for sample_data in Q_as.values()
            ]
        )
        Q_p = as_float_scalar(100 - 3.6 * (D_Ep_RMS - p_delta_C))
        Q_d = as_float_scalar(G_t / G_r * CCT_f * 100)

    if additional_data:
        return ColourRendering_Specification_CQS(
            sd_test.name,
            Q_a,
            Q_f,
            Q_p,
            Q_g,
            Q_d,
            Q_as,
            (test_vs_colorimetry_data, reference_vs_colorimetry_data),
        )
    else:
        return Q_a


def gamut_area(Lab: ArrayLike) -> Floating:
    """
    Return the gamut area :math:`G` covered by given *CIE L\\*a\\*b\\**
    matrices.

    Parameters
    ----------
    Lab
        *CIE L\\*a\\*b\\** colourspace matrices.

    Returns
    -------
    :class:`numpy.floating`
        Gamut area :math:`G`.

    Examples
    --------
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
    ...     np.array([40.52565174, 48.87449192, 3.45121680])
    ... ]
    >>> gamut_area(Lab)  # doctest: +ELLIPSIS
    8335.9482018...
    """

    Lab = as_float_array(Lab)
    Lab_s = np.roll(np.copy(Lab), -3)

    _L, a, b = tsplit(Lab)
    _L_s, a_s, b_s = tsplit(Lab_s)

    A = np.linalg.norm(Lab[..., 1:3], axis=-1)
    B = np.linalg.norm(Lab_s[..., 1:3], axis=-1)
    C = np.linalg.norm(np.dstack([a_s - a, b_s - b]), axis=-1)
    t = (A + B + C) / 2
    S = np.sqrt(t * (t - A) * (t - B) * (t - C))

    return np.sum(S)


def vs_colorimetry_data(
    sd_test: SpectralDistribution,
    sd_reference: SpectralDistribution,
    sds_vs: Dict[str, SpectralDistribution],
    cmfs: MultiSpectralDistributions,
    chromatic_adaptation: Boolean = False,
) -> Tuple[VS_ColorimetryData, ...]:
    """
    Return the *VS test colour samples* colorimetry data.

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    sd_reference
        Reference spectral distribution.
    sds_vs
        *VS test colour samples* spectral distributions.
    cmfs
        Standard observer colour matching functions.
    chromatic_adaptation
        Whether to perform chromatic adaptation.

    Returns
    -------
    :class:`tuple`
        *VS test colour samples* colorimetry data.
    """

    XYZ_t = sd_to_XYZ(sd_test, cmfs)
    XYZ_t /= XYZ_t[1]

    XYZ_r = sd_to_XYZ(sd_reference, cmfs)
    XYZ_r /= XYZ_r[1]
    xy_r = XYZ_to_xy(XYZ_r)

    vs_data = []
    for _key, value in sorted(INDEXES_TO_NAMES_VS.items()):
        sd_vs = sds_vs[value]

        with domain_range_scale("1"):
            XYZ_vs = sd_to_XYZ(sd_vs, cmfs, sd_test)

        if chromatic_adaptation:
            XYZ_vs = chromatic_adaptation_VonKries(
                XYZ_vs, XYZ_t, XYZ_r, transform="CMCCAT2000"
            )

        Lab_vs = XYZ_to_Lab(XYZ_vs, illuminant=xy_r)
        _L_vs, C_vs, _Hab = Lab_to_LCHab(Lab_vs)

        vs_data.append(VS_ColorimetryData(sd_vs.name, XYZ_vs, Lab_vs, C_vs))

    return tuple(vs_data)


def CCT_factor(
    reference_data: Tuple[VS_ColorimetryData, ...], XYZ_r: ArrayLike
) -> Floating:
    """
    Return the correlated colour temperature factor penalizing lamps with
    extremely low correlated colour temperatures.

    Parameters
    ----------
    reference_data
        Reference colorimetry data.
    XYZ_r
        *CIE XYZ* tristimulus values for reference.

    Returns
    -------
    :class:`numpy.floating`
        Correlated colour temperature factor.
    """

    xy_w = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    XYZ_w = xy_to_XYZ(xy_w)

    Lab = XYZ_to_Lab(
        chromatic_adaptation_VonKries(
            [colorimetry_data.XYZ for colorimetry_data in reference_data],
            XYZ_r,
            XYZ_w,
            transform="CMCCAT2000",
        ),
        illuminant=xy_w,
    )

    G_r = gamut_area(Lab) / GAMUT_AREA_D65
    CCT_f = 1 if G_r > 1 else G_r

    return CCT_f


def scale_conversion(
    D_E_ab: Floating, CCT_f: Floating, scaling_f: Floating
) -> Floating:
    """
    Return the *Colour Quality Scale* (CQS) for given :math:`\\Delta E_{ab}`
    value and given correlated colour temperature penalizing factor.

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
    :class:`numpy.floating`
        *Colour Quality Scale* (CQS).
    """

    Q_a = 10 * np.log1p(np.exp((100 - scaling_f * D_E_ab) / 10)) * CCT_f

    return Q_a


def delta_E_RMS(
    CQS_data: Dict[Integer, VS_ColourQualityScaleData], attribute: str
) -> Floating:
    """
    Compute the root-mean-square average for given *Colour Quality Scale*
    (CQS) data.

    Parameters
    ----------
    CQS_data
        *Colour Quality Scale* (CQS) data.
    attribute
        Colorimetry data attribute to use to compute the root-mean-square
        average.

    Returns
    -------
    :class:`numpy.floating`
        Root-mean-square average.
    """

    return np.sqrt(
        1
        / len(CQS_data)
        * np.sum(
            [
                getattr(sample_data, attribute) ** 2
                for sample_data in CQS_data.values()
            ]
        )
    )


def colour_quality_scales(
    test_data: Tuple[VS_ColorimetryData, ...],
    reference_data: Tuple[VS_ColorimetryData, ...],
    scaling_f: Floating,
    CCT_f: Floating,
) -> Dict[Integer, VS_ColourQualityScaleData]:
    """
    Return the *VS test colour samples* rendering scales.

    Parameters
    ----------
    test_data
        Test data.
    reference_data
        Reference data.
    scaling_f
        Scaling factor constant.
    CCT_f
        Factor penalizing lamps with extremely low correlated colour
        temperatures.

    Returns
    -------
    :class:`dict`
        *VS Test colour samples* colour rendering scales.
    """

    Q_as = {}
    for i in range(len(test_data)):
        D_C_ab = test_data[i].C - reference_data[i].C
        D_E_ab = as_float_scalar(
            euclidean_distance(test_data[i].Lab, reference_data[i].Lab)
        )

        if D_C_ab > 0:
            D_Ep_ab = np.sqrt(D_E_ab**2 - D_C_ab**2)
        else:
            D_Ep_ab = D_E_ab

        Q_a = scale_conversion(D_Ep_ab, CCT_f, scaling_f)

        Q_as[i + 1] = VS_ColourQualityScaleData(
            test_data[i].name, Q_a, D_C_ab, D_E_ab, D_Ep_ab
        )
    return Q_as
