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
from colour.appearance.ciecam02 import CAM_Specification_CIECAM02

import numpy as np
import os
from dataclasses import dataclass

from colour.algebra import Extrapolator, euclidean_distance, linstep_function
from colour.appearance import (
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
    sd_CIE_illuminant_D_series,
)
from colour.colorimetry.tristimulus_values import msds_to_XYZ
from colour.hints import ArrayLike, NDArrayFloat, Tuple, cast
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
    "ROOT_RESOURCES_CIE2017",
    "DataColorimetry_TCS_CIE2017",
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

ROOT_RESOURCES_CIE2017: str = os.path.join(
    os.path.dirname(__file__), "datasets"
)
"""*CIE 2017 Colour Fidelity Index* resources directory."""

_CACHE_TCS_CIE2017: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_TCS_CIE2017"
)


@dataclass
class DataColorimetry_TCS_CIE2017:
    """Define the class storing *test colour samples* colorimetry data."""

    name: str | list[str]
    XYZ: NDArrayFloat
    JMh: NDArrayFloat
    Jpapbp: NDArrayFloat
    CAM: CAM_Specification_CIECAM02


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
    colorimetry_data: Tuple[
        DataColorimetry_TCS_CIE2017, DataColorimetry_TCS_CIE2017, ...
    ]
    delta_E_s: NDArrayFloat


def colour_fidelity_index_CIE2017(
    sd_test: SpectralDistribution, additional_data: bool = False
) -> float | ColourRendering_Specification_CIE2017:
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
    :class:`float` or \
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
    70.1208254...
    """

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

    CCT, D_uv = tsplit(CCT_reference_illuminant(sd_test))
    sd_reference = sd_reference_illuminant(CCT, shape)

    # NOTE: All computations except CCT calculation use the
    # "CIE 1964 10 Degree Standard Observer".
    # pylint: disable=E1102
    cmfs_10 = reshape_msds(
        MSDS_CMFS["CIE 1964 10 Degree Standard Observer"], shape, copy=False
    )

    # pylint: disable=E1102
    sds_tcs = load_TCS_CIE2017(shape)

    (
        test_tcs_colorimetry_data,
        reference_tcs_colorimetry_data,
    ) = tcs_colorimetry_data([sd_test, sd_reference], sds_tcs, cmfs_10)

    delta_E_s = euclidean_distance(
        test_tcs_colorimetry_data.Jpapbp,
        reference_tcs_colorimetry_data.Jpapbp,
    )

    R_s = delta_E_to_R_f(delta_E_s)
    R_f = cast(float, delta_E_to_R_f(np.average(delta_E_s)))

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

    global _CACHE_TCS_CIE2017  # noqa: PLW0602

    interval = shape.interval

    attest(
        interval in (1, 5),
        "Spectral shape interval must be either 1nm or 5nm!",
    )

    filename = f"tcs_cfi2017_{as_int_scalar(interval)}_nm.csv.gz"

    if filename in _CACHE_TCS_CIE2017:
        return _CACHE_TCS_CIE2017[filename]

    data = np.genfromtxt(
        str(os.path.join(ROOT_RESOURCES_CIE2017, filename)), delimiter=","
    )
    labels = [f"TCS{i} (CIE 2017)" for i in range(99)]

    tcs = MultiSpectralDistributions(data[:, 1:], data[:, 0], labels)

    _CACHE_TCS_CIE2017[filename] = tcs

    return tcs


def CCT_reference_illuminant(sd: SpectralDistribution) -> NDArrayFloat:
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
    >>> sd = SDS_ILLUMINANTS["FL2"]
    >>> CCT_reference_illuminant(sd)  # doctest: +ELLIPSIS
    array([  4.2244697...e+03,   1.7871111...e-03])
    """

    XYZ = sd_to_XYZ(sd.values, shape=sd.shape, method="Integration")

    # Use CFI2017 and TM30 range of 1,000K to 25,000K for performance.
    return uv_to_CCT_Ohno2013(
        UCS_to_uv(XYZ_to_UCS(XYZ)), start=1000, end=25000
    )


def sd_reference_illuminant(
    CCT: float, shape: SpectralShape
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
    ...         4224.469705295263300, SpectralShape(380, 780, 20)
    ...     )
    ...
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
                         SpragueInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})
    """

    if CCT <= 5000:
        sd_planckian = sd_blackbody(CCT, shape)

    if CCT >= 4000:
        xy = CCT_to_xy_CIE_D(CCT)
        sd_daylight = sd_CIE_illuminant_D_series(xy, shape=shape)

    if CCT < 4000:
        sd_reference = sd_planckian
    elif 4000 <= CCT <= 5000:
        # Planckian and daylight illuminant must be normalised so that the
        # mixture isn't biased.
        sd_planckian /= sd_to_XYZ(
            sd_planckian.values, shape=shape, method="Integration"
        )[1]
        sd_daylight /= sd_to_XYZ(
            sd_daylight.values, shape=shape, method="Integration"
        )[1]

        # Mixture: 4200K should be 80% Planckian, 20% CIE Illuminant D Series.
        m = (CCT - 4000) / 1000
        values = linstep_function(m, sd_planckian.values, sd_daylight.values)
        name = (
            f"{as_int_scalar(CCT)}K "
            f"Blackbody & CIE Illuminant D Series Mixture - "
            f"{as_float_scalar(100 * m):.1f}%"
        )
        sd_reference = SpectralDistribution(
            values, shape.wavelengths, name=name
        )
    elif CCT > 5000:
        sd_reference = sd_daylight

    return sd_reference


def tcs_colorimetry_data(
    sd_irradiance: SpectralDistribution | list[SpectralDistribution],
    sds_tcs: MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions,
) -> list[DataColorimetry_TCS_CIE2017]:
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
        *Test colour samples* spectral reflectance distributions.
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
    if isinstance(sd_irradiance, SpectralDistribution):
        sd_irradiance = [sd_irradiance]

    XYZ_w = np.full((len(sd_irradiance), 3), np.nan)
    for idx, sd in enumerate(sd_irradiance):
        XYZ_t = sd_to_XYZ(
            sd.values,
            cmfs,
            shape=sd.shape,
            method="Integration",
        )
        k = 100 / XYZ_t[1]
        XYZ_w[idx] = k * XYZ_t
        sd_irradiance[idx] = sd_irradiance[idx].copy() * k
    XYZ_w = as_float_array(XYZ_w)

    Y_b = 20
    L_A = 100
    surround = VIEWING_CONDITIONS_CIECAM02["Average"]

    sds_tcs_t = np.tile(sds_tcs.values.T, (len(sd_irradiance), 1, 1))
    sds_tcs_t = sds_tcs_t * as_float_array(
        [sd.values for sd in sd_irradiance]
    ).reshape(len(sd_irradiance), 1, len(sd_irradiance[0]))

    XYZ = msds_to_XYZ(
        sds_tcs_t,
        cmfs,
        method="Integration",
        shape=sds_tcs.shape,
    )
    specification = XYZ_to_CIECAM02(
        XYZ,
        XYZ_w.reshape((len(sd_irradiance), 1, 3)),
        L_A,
        Y_b,
        surround,
        discount_illuminant=True,
        compute_HQ=False,
    )

    JMh = tstack(
        [
            cast(NDArrayFloat, specification.J),
            cast(NDArrayFloat, specification.M),
            cast(NDArrayFloat, specification.h),
        ]
    )
    Jpapbp = JMh_CIECAM02_to_CAM02UCS(JMh)
    tcs_data = []

    specification = as_float_array(specification).transpose((0, 2, 1))
    specification = [CAM_Specification_CIECAM02(*t) for t in specification]

    for sd_idx in range(len(sd_irradiance)):
        tcs_data.append(
            DataColorimetry_TCS_CIE2017(
                sds_tcs.display_labels,
                XYZ[sd_idx],
                JMh[sd_idx],
                Jpapbp[sd_idx],
                specification[sd_idx],
            )
        )

    return tcs_data


def delta_E_to_R_f(delta_E: ArrayLike) -> NDArrayFloat:
    """
    Convert from colour-appearance difference to
    *CIE 2017 Colour Fidelity Index* (CFI) :math:`R_f` value.

    Parameters
    ----------
    delta_E
        Euclidean distance between two colours in *CAM02-UCS* colourspace.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding *CIE 2017 Colour Fidelity Index* (CFI) :math:`R_f` value.
    """

    delta_E = as_float_array(delta_E)

    c_f = 6.73

    return as_float(10 * np.log1p(np.exp((100 - c_f * delta_E) / 10)))
