"""
CIE 2017 Colour Fidelity Index
==============================

Defines the *CIE 2017 Colour Fidelity Index* (CFI) computation objects:

- :class:`colour.quality.ColourRendering_Specification_CIE2017`
- :func:`colour.quality.colour_fidelity_index_CIE2017`

References
----------
-   :cite:`CIETC1-902017` : CIE TC 1-90. (2017). CIE 2017 colour fidelity index
    for accurate scientific use. CIE Central Bureau. ISBN:978-3-902842-61-9
"""

from __future__ import annotations

import numpy as np
import os
from dataclasses import dataclass

from colour.algebra import Extrapolator, euclidean_distance, linstep_function
from colour.appearance import (
    CAM_Specification_CIECAM02,
    XYZ_to_CIECAM02,
    VIEWING_CONDITIONS_CIECAM02,
)
from colour.colorimetry import (
    MSDS_CMFS,
    MultiSpectralDistributions,
    SpectralShape,
    SpectralDistribution,
    sd_to_XYZ,
    sd_blackbody,
    reshape_msds,
    sd_ones,
    sd_CIE_illuminant_D_series,
)
from colour.hints import (
    Boolean,
    Dict,
    Floating,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    NDArray,
    Tuple,
    Union,
    cast,
)
from colour.models import XYZ_to_UCS, UCS_to_uv, JMh_CIECAM02_to_CAM02UCS
from colour.temperature import uv_to_CCT_Ohno2013, CCT_to_xy_CIE_D
from colour.utilities import (
    CACHE_REGISTRY,
    as_float,
    as_float_array,
    as_float_scalar,
    as_int_scalar,
    attest,
    tsplit,
    tstack,
    usage_warning,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SPECTRAL_SHAPE_CIE2017",
    "RESOURCES_DIRECTORY_CIE2017",
    "TCS_ColorimetryData_CIE2017",
    "ColourRendering_Specification_CIE2017",
    "colour_fidelity_index_CIE2017",
    "load_TCS_CIE2017",
    "CCT_reference_illuminant",
    "sd_reference_illuminant",
    "tcs_colorimetry_data",
    "delta_E_to_R_f",
]

SPECTRAL_SHAPE_CIE2017: SpectralShape = SpectralShape(380, 780, 1)
"""
Spectral shape for *CIE 2017 Colour Fidelity Index* (CFI)
standard.
"""

RESOURCES_DIRECTORY_CIE2017: str = os.path.join(
    os.path.dirname(__file__), "datasets"
)
"""*CIE 2017 Colour Fidelity Index* resources directory."""

_CACHE_TCS_CIE2017: Dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_TCS_CIE2017"
)


@dataclass
class TCS_ColorimetryData_CIE2017:
    """Define the class storing *test colour samples* colorimetry data."""

    name: str
    XYZ: NDArray
    CAM: CAM_Specification_CIECAM02
    JMh: NDArray
    Jpapbp: NDArray


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
    R_f: Floating
    R_s: NDArray
    CCT: Floating
    D_uv: Floating
    colorimetry_data: Tuple[
        Tuple[TCS_ColorimetryData_CIE2017, ...],
        Tuple[TCS_ColorimetryData_CIE2017, ...],
    ]
    delta_E_s: NDArray


def colour_fidelity_index_CIE2017(
    sd_test: SpectralDistribution, additional_data: Boolean = False
) -> Union[Floating, ColourRendering_Specification_CIE2017]:
    """
    Return the *CIE 2017 Colour Fidelity Index* (CFI) :math:`R_f` of given
    spectral distribution.

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    additional_data
        Whether to output additional data.

    Returns
    -------
    :class:`numpy.floating` or \
:class:`colour.quality.ColourRendering_Specification_CIE2017`
        *CIE 2017 Colour Fidelity Index* (CFI) :math:`R_f`.

    References
    ----------
    :cite:`CIETC1-902017`

    Examples
    --------
    >>> from colour.colorimetry import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS['FL2']
    >>> colour_fidelity_index_CIE2017(sd)  # doctest: +ELLIPSIS
    70.1208254...
    """

    if sd_test.shape.start > 380 or sd_test.shape.end < 780:
        usage_warning(
            "Test spectral distribution shape does not span the"
            "recommended 380-780nm range, missing values will be"
            "filled with zeros!"
        )

        # NOTE: "CIE 2017 Colour Fidelity Index" standard recommends filling
        # missing values with zeros.
        sd_test = cast(SpectralDistribution, sd_test.copy())
        sd_test.extrapolator = Extrapolator
        sd_test.extrapolator_kwargs = {
            "method": "constant",
            "left": 0,
            "right": 0,
        }

    if sd_test.shape.interval > 5:
        raise ValueError(
            "Test spectral distribution interval is greater than"
            "5nm which is the maximum recommended value "
            'for computing the "CIE 2017 Colour Fidelity Index"!'
        )

    shape = SpectralShape(
        SPECTRAL_SHAPE_CIE2017.start,
        SPECTRAL_SHAPE_CIE2017.end,
        sd_test.shape.interval,
    )

    CCT, D_uv = tsplit(CCT_reference_illuminant(sd_test))
    sd_reference = sd_reference_illuminant(CCT, shape)

    # NOTE: All computations except CCT calculation use the
    # "CIE 1964 10 Degree Standard Observer".
    # pylint: disable=E1102
    cmfs_10 = reshape_msds(
        MSDS_CMFS["CIE 1964 10 Degree Standard Observer"], shape
    )

    # pylint: disable=E1102
    sds_tcs = reshape_msds(load_TCS_CIE2017(shape), shape)

    test_tcs_colorimetry_data = tcs_colorimetry_data(sd_test, sds_tcs, cmfs_10)
    reference_tcs_colorimetry_data = tcs_colorimetry_data(
        sd_reference, sds_tcs, cmfs_10
    )

    delta_E_s = np.empty(len(sds_tcs.labels))
    for i, _delta_E in enumerate(delta_E_s):
        delta_E_s[i] = euclidean_distance(
            test_tcs_colorimetry_data[i].Jpapbp,
            reference_tcs_colorimetry_data[i].Jpapbp,
        )

    R_s = as_float_array(delta_E_to_R_f(delta_E_s))
    R_f = as_float_scalar(delta_E_to_R_f(np.average(delta_E_s)))

    if additional_data:
        return ColourRendering_Specification_CIE2017(
            sd_test.name,
            sd_reference,
            R_f,
            R_s,
            CCT,
            D_uv,
            (test_tcs_colorimetry_data, reference_tcs_colorimetry_data),
            delta_E_s,
        )
    else:
        return R_f


def load_TCS_CIE2017(shape: SpectralShape) -> MultiSpectralDistributions:
    """
    Load the *CIE 2017 Test Colour Samples* dataset appropriate for the given
    spectral shape.

    The datasets are cached and won't be loaded again on subsequent calls to
    this definition.

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

    global _CACHE_TCS_CIE2017

    interval = shape.interval

    attest(
        interval in (1, 5),
        "Spectral shape interval must be either 1nm or 5nm!",
    )

    filename = f"tcs_cfi2017_{as_int_scalar(interval)}_nm.csv.gz"

    if filename in _CACHE_TCS_CIE2017:
        return _CACHE_TCS_CIE2017[filename]

    data = np.genfromtxt(
        str(os.path.join(RESOURCES_DIRECTORY_CIE2017, filename)), delimiter=","
    )
    labels = [f"TCS{i} (CIE 2017)" for i in range(99)]

    tcs = MultiSpectralDistributions(data[:, 1:], data[:, 0], labels)

    _CACHE_TCS_CIE2017[filename] = tcs

    return tcs


def CCT_reference_illuminant(sd: SpectralDistribution) -> NDArray:
    """
    Compute the reference illuminant correlated colour temperature
    :math:`T_{cp}` and :math:`\\Delta_{uv}` for given test spectral
    distribution using *Ohno (2013)* method.

    Parameters
    ----------
    sd
        Test spectral distribution.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS['FL2']
    >>> CCT_reference_illuminant(sd)  # doctest: +ELLIPSIS
    array([  4.2244697...e+03,   1.7871111...e-03])
    """

    XYZ = sd_to_XYZ(sd)

    return uv_to_CCT_Ohno2013(UCS_to_uv(XYZ_to_UCS(XYZ)))


def sd_reference_illuminant(
    CCT: Floating, shape: SpectralShape
) -> SpectralDistribution:
    """
    Compute the reference illuminant for a given correlated colour temperature
    :math:`T_{cp}` for use in *CIE 2017 Colour Fidelity Index* (CFI)
    computation.

    Parameters
    ----------
    CCT
        Correlated colour temperature :math:`T_{cp}`.
    shape
        Desired shape of the returned spectral distribution.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Reference illuminant for *CIE 2017 Colour Fidelity Index* (CFI)
        computation.

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     sd_reference_illuminant(  # doctest: +ELLIPSIS
    ...         4224.469705295263300, SpectralShape(380, 780, 20))
    SpectralDistribution([[ 380.        ,    0.0034089...],
                          [ 400.        ,    0.0044208...],
                          [ 420.        ,    0.0053260...],
                          [ 440.        ,    0.0062857...],
                          [ 460.        ,    0.0072767...],
                          [ 480.        ,    0.0080207...],
                          [ 500.        ,    0.0086590...],
                          [ 520.        ,    0.0092242...],
                          [ 540.        ,    0.0097686...],
                          [ 560.        ,    0.0101444...],
                          [ 580.        ,    0.0104475...],
                          [ 600.        ,    0.0107642...],
                          [ 620.        ,    0.0110439...],
                          [ 640.        ,    0.0112535...],
                          [ 660.        ,    0.0113922...],
                          [ 680.        ,    0.0115185...],
                          [ 700.        ,    0.0113155...],
                          [ 720.        ,    0.0108192...],
                          [ 740.        ,    0.0111582...],
                          [ 760.        ,    0.0101299...],
                          [ 780.        ,    0.0105638...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    """

    if CCT <= 5000:
        sd_planckian = sd_blackbody(CCT, shape)

    if CCT >= 4000:
        xy = CCT_to_xy_CIE_D(CCT)
        sd_daylight = sd_CIE_illuminant_D_series(xy).align(shape)

    if CCT < 4000:
        sd_reference = sd_planckian
    elif 4000 <= CCT <= 5000:
        # Planckian and daylight illuminant must be normalised so that the
        # mixture isn't biased.
        sd_planckian /= sd_to_XYZ(sd_planckian)[1]  # type: ignore[misc]
        sd_daylight /= sd_to_XYZ(sd_daylight)[1]  # type: ignore[misc]

        # Mixture: 4200K should be 80% Planckian, 20% CIE Illuminant D Series.
        m = (CCT - 4000) / 1000
        values = linstep_function(m, sd_planckian.values, sd_daylight.values)
        name = (
            f"{as_int_scalar(CCT)}K "
            f"Blackbody & CIE Illuminant D Series Mixture - "
            f"{as_float_scalar(100 * m):.1f}%"
        )
        sd_reference = SpectralDistribution(values, shape.range(), name=name)
    elif CCT > 5000:
        sd_reference = sd_daylight

    return sd_reference


def tcs_colorimetry_data(
    sd_irradiance: SpectralDistribution,
    sds_tcs: MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions,
) -> Tuple[TCS_ColorimetryData_CIE2017, ...]:
    """
    Return the *test colour samples* colorimetry data under given test light
    source or reference illuminant spectral distribution for the
    *CIE 2017 Colour Fidelity Index* (CFI) computations.

    Parameters
    ----------
    sd_irradiance
        Test light source or reference illuminant spectral distribution, i.e.
        the irradiance emitter.
    sds_tcs
        *Test colour samples* spectral distributions.
    cmfs
        Standard observer colour matching functions.

    Returns
    -------
    :class:`tuple`
        *Test colour samples* colorimetry data under the given test light
        source or reference illuminant spectral distribution.

    Examples
    --------
    >>> delta_E_to_R_f(4.4410383190)  # doctest: +ELLIPSIS
    70.1208254...
    """

    XYZ_w = sd_to_XYZ(sd_ones(), cmfs, sd_irradiance)
    Y_b = 20
    L_A = 100
    surround = VIEWING_CONDITIONS_CIECAM02["Average"]

    tcs_data = []
    for sd_tcs in sds_tcs.to_sds():
        XYZ = sd_to_XYZ(sd_tcs, cmfs, sd_irradiance)
        CAM = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, True)
        JMh = tstack([CAM.J, CAM.M, CAM.h])
        Jpapbp = JMh_CIECAM02_to_CAM02UCS(JMh)

        tcs_data.append(
            TCS_ColorimetryData_CIE2017(sd_tcs.name, XYZ, CAM, JMh, Jpapbp)
        )

    return tuple(tcs_data)


def delta_E_to_R_f(delta_E: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Convert from colour-appearance difference to
    *CIE 2017 Colour Fidelity Index* (CFI) :math:`R_f` value.

    Parameters
    ----------
    delta_E
        Euclidean distance between two colours in *CAM02-UCS* colourspace.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Corresponding *CIE 2017 Colour Fidelity Index* (CFI) :math:`R_f` value.
    """

    delta_E = as_float_array(delta_E)

    c_f = 6.73

    return as_float(10 * np.log1p(np.exp((100 - c_f * delta_E) / 10)))
