"""
Colour Rendering Index
======================

Defines the *Colour Rendering Index* (CRI) computation objects:

-   :class:`colour.quality.ColourRendering_Specification_CRI`
-   :func:`colour.colour_rendering_index`

References
----------
-   :cite:`Ohno2008a` : Ohno, Yoshiro, & Davis, W. (2008). NIST CQS simulation
    (Version 7.4) [Computer software].
    https://drive.google.com/file/d/1PsuU6QjUJjCX6tQyCud6ul2Tbs8rYWW9/view?\
usp=sharing
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from colour.algebra import euclidean_distance, spow
from colour.colorimetry import (
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
    Boolean,
    Dict,
    Floating,
    FloatingOrNDArray,
    Integer,
    NDArray,
    Tuple,
    Union,
)
from colour.models import UCS_to_uv, XYZ_to_UCS, XYZ_to_xyY
from colour.quality.datasets.tcs import INDEXES_TO_NAMES_TCS, SDS_TCS
from colour.temperature import CCT_to_xy_CIE_D, uv_to_CCT_Robertson1968
from colour.utilities import domain_range_scale, as_float_scalar

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TCS_ColorimetryData",
    "TCS_ColourQualityScaleData",
    "ColourRendering_Specification_CRI",
    "colour_rendering_index",
    "tcs_colorimetry_data",
    "colour_rendering_indexes",
]


@dataclass
class TCS_ColorimetryData:
    """Define the class storing *test colour samples* colorimetry data."""

    name: str
    XYZ: NDArray
    uv: NDArray
    UVW: NDArray


@dataclass
class TCS_ColourQualityScaleData:
    """
    Define the class storing *test colour samples* colour rendering
    index data.
    """

    name: str
    Q_a: Floating


@dataclass()
class ColourRendering_Specification_CRI:
    """
    Define the *Colour Rendering Index* (CRI) colour quality specification.

    Parameters
    ----------
    name
        Name of the test spectral distribution.
    Q_a
        *Colour Rendering Index* (CRI) :math:`Q_a`.
    Q_as
        Individual *colour rendering indexes* data for each sample.
    colorimetry_data
        Colorimetry data for the test and reference computations.

    References
    ----------
    :cite:`Ohno2008a`
    """

    name: str
    Q_a: Floating
    Q_as: Dict[Integer, TCS_ColourQualityScaleData]
    colorimetry_data: Tuple[
        Tuple[TCS_ColorimetryData, ...], Tuple[TCS_ColorimetryData, ...]
    ]


def colour_rendering_index(
    sd_test: SpectralDistribution, additional_data: Boolean = False
) -> Union[Floating, ColourRendering_Specification_CRI]:
    """
    Return the *Colour Rendering Index* (CRI) :math:`Q_a` of given spectral
    distribution.

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    additional_data
        Whether to output additional data.

    Returns
    -------
    :class:`numpy.floating` or \
:class:`colour.quality.ColourRendering_Specification_CRI`
        *Colour Rendering Index* (CRI).

    References
    ----------
    :cite:`Ohno2008a`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS['FL2']
    >>> colour_rendering_index(sd)  # doctest: +ELLIPSIS
    64.2337241...
    """

    # pylint: disable=E1102
    cmfs = reshape_msds(
        MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
        SPECTRAL_SHAPE_DEFAULT,
    )

    shape = cmfs.shape
    sd_test = reshape_sd(sd_test, shape)
    tcs_sds = {sd.name: reshape_sd(sd, shape) for sd in SDS_TCS.values()}

    with domain_range_scale("1"):
        XYZ = sd_to_XYZ(sd_test, cmfs)

    uv = UCS_to_uv(XYZ_to_UCS(XYZ))
    CCT, _D_uv = uv_to_CCT_Robertson1968(uv)

    if CCT < 5000:
        sd_reference = sd_blackbody(CCT, shape)
    else:
        xy = CCT_to_xy_CIE_D(CCT)
        sd_reference = sd_CIE_illuminant_D_series(xy)
        sd_reference.align(shape)

    test_tcs_colorimetry_data = tcs_colorimetry_data(
        sd_test, sd_reference, tcs_sds, cmfs, chromatic_adaptation=True
    )

    reference_tcs_colorimetry_data = tcs_colorimetry_data(
        sd_reference, sd_reference, tcs_sds, cmfs
    )

    Q_as = colour_rendering_indexes(
        test_tcs_colorimetry_data, reference_tcs_colorimetry_data
    )

    Q_a = as_float_scalar(
        np.average(
            [v.Q_a for k, v in Q_as.items() if k in (1, 2, 3, 4, 5, 6, 7, 8)]
        )
    )

    if additional_data:
        return ColourRendering_Specification_CRI(
            sd_test.name,
            Q_a,
            Q_as,
            (test_tcs_colorimetry_data, reference_tcs_colorimetry_data),
        )
    else:
        return Q_a


def tcs_colorimetry_data(
    sd_t: SpectralDistribution,
    sd_r: SpectralDistribution,
    sds_tcs: Dict[str, SpectralDistribution],
    cmfs: MultiSpectralDistributions,
    chromatic_adaptation: Boolean = False,
) -> Tuple[TCS_ColorimetryData, ...]:
    """
    Return the *test colour samples* colorimetry data.

    Parameters
    ----------
    sd_t
        Test spectral distribution.
    sd_r
        Reference spectral distribution.
    sds_tcs
        *Test colour samples* spectral distributions.
    cmfs
        Standard observer colour matching functions.
    chromatic_adaptation
        Perform chromatic adaptation.

    Returns
    -------
    :class:`tuple`
        *Test colour samples* colorimetry data.
    """

    XYZ_t = sd_to_XYZ(sd_t, cmfs)
    uv_t = UCS_to_uv(XYZ_to_UCS(XYZ_t))
    u_t, v_t = uv_t[0], uv_t[1]

    XYZ_r = sd_to_XYZ(sd_r, cmfs)
    uv_r = UCS_to_uv(XYZ_to_UCS(XYZ_r))
    u_r, v_r = uv_r[0], uv_r[1]

    tcs_data = []
    for _key, value in sorted(INDEXES_TO_NAMES_TCS.items()):
        sd_tcs = sds_tcs[value]
        XYZ_tcs = sd_to_XYZ(sd_tcs, cmfs, sd_t)
        xyY_tcs = XYZ_to_xyY(XYZ_tcs)
        uv_tcs = UCS_to_uv(XYZ_to_UCS(XYZ_tcs))
        u_tcs, v_tcs = uv_tcs[0], uv_tcs[1]

        if chromatic_adaptation:

            def c(
                x: FloatingOrNDArray, y: FloatingOrNDArray
            ) -> FloatingOrNDArray:
                """Compute the :math:`c` term."""

                return (4 - x - 10 * y) / y

            def d(
                x: FloatingOrNDArray, y: FloatingOrNDArray
            ) -> FloatingOrNDArray:
                """Compute the :math:`d` term."""

                return (1.708 * y + 0.404 - 1.481 * x) / y

            c_t, d_t = c(u_t, v_t), d(u_t, v_t)
            c_r, d_r = c(u_r, v_r), d(u_r, v_r)
            tcs_c, tcs_d = c(u_tcs, v_tcs), d(u_tcs, v_tcs)
            u_tcs = (
                10.872 + 0.404 * c_r / c_t * tcs_c - 4 * d_r / d_t * tcs_d
            ) / (16.518 + 1.481 * c_r / c_t * tcs_c - d_r / d_t * tcs_d)
            v_tcs = 5.52 / (
                16.518 + 1.481 * c_r / c_t * tcs_c - d_r / d_t * tcs_d
            )

        W_tcs = 25 * spow(xyY_tcs[-1], 1 / 3) - 17
        U_tcs = 13 * W_tcs * (u_tcs - u_r)
        V_tcs = 13 * W_tcs * (v_tcs - v_r)

        tcs_data.append(
            TCS_ColorimetryData(
                sd_tcs.name, XYZ_tcs, uv_tcs, np.array([U_tcs, V_tcs, W_tcs])
            )
        )

    return tuple(tcs_data)


def colour_rendering_indexes(
    test_data: Tuple[TCS_ColorimetryData, ...],
    reference_data: Tuple[TCS_ColorimetryData, ...],
) -> Dict[Integer, TCS_ColourQualityScaleData]:
    """
    Return the *test colour samples* rendering indexes :math:`Q_a`.

    Parameters
    ----------
    test_data
        Test data.
    reference_data
        Reference data.

    Returns
    -------
    :class:`dict`
        *Test colour samples* *Colour Rendering Index* (CRI).
    """

    Q_as = {}
    for i in range(len(test_data)):
        Q_as[i + 1] = TCS_ColourQualityScaleData(
            test_data[i].name,
            100
            - 4.6
            * as_float_scalar(
                euclidean_distance(reference_data[i].UVW, test_data[i].UVW)
            ),
        )

    return Q_as
