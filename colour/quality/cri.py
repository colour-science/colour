"""
Colour Rendering Index
======================

Define the *Colour Rendering Index* (CRI) computation objects.

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

import typing
from dataclasses import dataclass

from colour.algebra import euclidean_distance, sdiv, sdiv_mode, spow
from colour.colorimetry import (
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
    from colour.hints import Dict, List, Literal, NDArrayFloat, Tuple

from colour.models import UCS_to_uv, XYZ_to_UCS, XYZ_to_xyY
from colour.quality.datasets.tcs import INDEXES_TO_NAMES_TCS, SDS_TCS
from colour.temperature import CCT_to_xy_CIE_D, uv_to_CCT_Robertson1968
from colour.utilities import (
    array_namespace,
    as_float_scalar,
    domain_range_scale,
    suppress_warnings,
    tstack,
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
    "DataColorimetry_TCS",
    "DataColourQualityScale_TCS",
    "ColourRendering_Specification_CRI",
    "COLOUR_RENDERING_INDEX_METHODS",
    "colour_rendering_index",
    "tcs_colorimetry_data",
    "colour_rendering_indexes",
]


@dataclass
class DataColorimetry_TCS:
    """
    Store colorimetric data for *test colour samples* used in colour
    rendering index calculations.

    This dataclass encapsulates the colorimetric properties of test colour
    samples, including their tristimulus values, chromaticity coordinates,
    and colour appearance attributes required for evaluating light source
    colour rendering performance.

    Attributes
    ----------
    name
        Identifier for the test colour sample.
    XYZ
        *CIE XYZ* tristimulus values of the test colour sample.
    uv
        *CIE 1960 UCS* chromaticity coordinates of the test colour sample.
    UVW
        *CIE 1964 U*V*W** colour space coordinates of the test colour
        sample.
    """

    name: str
    XYZ: NDArrayFloat
    uv: NDArrayFloat
    UVW: NDArrayFloat


@dataclass
class DataColourQualityScale_TCS:
    """
    Store colour rendering index quality scale data for individual *test
    colour samples*.

    Attributes
    ----------
    name
        Identifier of the test colour sample.
    Q_a
        Colour rendering index :math:`Q_a` value for the test colour sample.
    """

    name: str
    Q_a: float


@dataclass()
class ColourRendering_Specification_CRI:
    """
    Define the *Colour Rendering Index* (CRI) colour quality specification.

    This dataclass represents the colour quality assessment results using
    the CRI method, which evaluates how accurately a light source renders
    colours compared to a reference illuminant.

    Parameters
    ----------
    name
        Name of the test spectral distribution.
    Q_a
        *Colour Rendering Index* (CRI) :math:`Q_a` general index value.
    Q_as
        Individual *colour rendering indexes* data for each test colour
        sample.
    colorimetry_data
        Colorimetry data for the test and reference illuminant
        computations.

    References
    ----------
    :cite:`Ohno2008a`
    """

    name: str
    Q_a: float
    Q_as: Dict[int, DataColourQualityScale_TCS]
    colorimetry_data: Tuple[
        Tuple[DataColorimetry_TCS, ...], Tuple[DataColorimetry_TCS, ...]
    ]


COLOUR_RENDERING_INDEX_METHODS: tuple = ("CIE 1995", "CIE 2024")
if is_documentation_building():  # pragma: no cover
    COLOUR_RENDERING_INDEX_METHODS = DocstringTuple(COLOUR_RENDERING_INDEX_METHODS)
    COLOUR_RENDERING_INDEX_METHODS.__doc__ = """
Supported *Colour Rendering Index* (CRI) computation methods.

References
----------
:cite:`Ohno2008a`
"""


@typing.overload
def colour_rendering_index(
    sd_test: SpectralDistribution,
    additional_data: Literal[False] = False,
    method: Literal["CIE 1995", "CIE 2024"] | str = ...,
) -> float: ...


@typing.overload
def colour_rendering_index(
    sd_test: SpectralDistribution,
    additional_data: Literal[True],
    method: Literal["CIE 1995", "CIE 2024"] | str = ...,
) -> ColourRendering_Specification_CRI: ...


@typing.overload
def colour_rendering_index(
    sd_test: MultiSpectralDistributions,
    additional_data: Literal[False] = False,
    method: Literal["CIE 1995", "CIE 2024"] | str = ...,
) -> NDArrayFloat: ...


def colour_rendering_index(
    sd_test: SpectralDistribution | MultiSpectralDistributions,
    additional_data: bool = False,
    method: Literal["CIE 1995", "CIE 2024"] | str = "CIE 1995",
) -> float | NDArrayFloat | ColourRendering_Specification_CRI:
    """
    Compute the *Colour Rendering Index* (CRI) :math:`Q_a` of the specified
    spectral distribution.

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
    :class:`colour.quality.ColourRendering_Specification_CRI`
        *Colour Rendering Index* (CRI).

    Notes
    -----
    -   The score is piecewise differentiable with respect to the test
        spectral values while the reference-illuminant branch remains
        unchanged. Its derivative is undefined where that selection changes.

    References
    ----------
    :cite:`Ohno2008a`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS["FL2"]
    >>> colour_rendering_index(sd)  # doctest: +ELLIPSIS
    np.float64(64.2337241...)
    """

    method = validate_method(method, tuple(COLOUR_RENDERING_INDEX_METHODS))

    cmfs = reshape_msds(
        MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
        SPECTRAL_SHAPE_DEFAULT,
        copy=False,
    )

    shape = cmfs.shape
    sds_tcs = SDS_TCS[method]
    tcs_sds = {sd.name: reshape_sd(sd, shape, copy=False) for sd in sds_tcs.values()}

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
    CCT = uv_to_CCT_Robertson1968(uv)[..., 0]

    # ``planck_law`` squeezes its output, so a single-CCT batch collapses
    # to 1-D; the sample axis is reinstated below.
    planckian = planck_law(shape.wavelengths * 1e-9, CCT) * 1e-9
    planckian_values = (
        planckian[None, :]
        if planckian.ndim == 1
        else xp_matrix_transpose(planckian, xp=xp)
    )
    # ``CCT_to_xy_CIE_D`` warns for any sample outside ``[4000, 25000]`` K
    # even when the ``xp.where`` below will discard those values.
    with suppress_warnings(colour_usage_warnings=True):
        daylight = CIE_illuminant_D_series(CCT_to_xy_CIE_D(CCT), shape=shape)
    daylight_values = (
        daylight[None, :]
        if daylight.ndim == 1
        else xp_matrix_transpose(daylight, xp=xp)
    )
    ref_values = xp.where(CCT[..., None] < 5000, planckian_values, daylight_values)

    test_names, test_XYZ, test_uv, test_UVW = _tcs_colorimetry_data(
        test_values, ref_values, tcs_sds, cmfs, chromatic_adaptation=True, method=method
    )
    ref_names, ref_XYZ, ref_uv, ref_UVW = _tcs_colorimetry_data(
        ref_values, ref_values, tcs_sds, cmfs, method=method
    )

    delta_E = euclidean_distance(test_UVW, ref_UVW)
    # The general *Colour Rendering Index* (CRI) :math:`R_a` is defined over
    # the first 8 test colour samples only, the remaining samples yield
    # special indexes.
    delta_E_8 = delta_E[..., :8]
    Q_a = xp_average(100 - 4.6 * delta_E_8, axis=-1, xp=xp)

    if is_msds:
        return Q_a

    Q_a_scalar = as_float_scalar(Q_a[0])

    if additional_data:
        Q_as = {
            i + 1: DataColourQualityScale_TCS(
                test_names[i], as_float_scalar(100 - 4.6 * delta_E[0, i])
            )
            for i in range(len(test_names))
        }
        test_data = tuple(
            DataColorimetry_TCS(name, test_XYZ[0, i], test_uv[0, i], test_UVW[0, i])
            for i, name in enumerate(test_names)
        )
        ref_data = tuple(
            DataColorimetry_TCS(name, ref_XYZ[0, i], ref_uv[0, i], ref_UVW[0, i])
            for i, name in enumerate(ref_names)
        )
        return ColourRendering_Specification_CRI(
            sd_test.name, Q_a_scalar, Q_as, (test_data, ref_data)
        )

    return Q_a_scalar


def _tcs_colorimetry_data(
    t_values: NDArrayFloat,
    r_values: NDArrayFloat,
    sds_tcs: Dict[str, SpectralDistribution],
    cmfs: MultiSpectralDistributions,
    chromatic_adaptation: bool = False,
    method: Literal["CIE 1995", "CIE 2024"] | str = "CIE 1995",
) -> Tuple[List[str], NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """
    Compute the *test colour samples* colorimetry arrays in a single
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
        ``(names, XYZ_tcs, uv_tcs, UVW_tcs)`` with leading shape
        ``(..., n_test_colour_samples)``.
    """

    method = validate_method(method, tuple(COLOUR_RENDERING_INDEX_METHODS))

    XYZ_t = msds_to_XYZ(t_values, cmfs, method="Integration", shape=cmfs.shape)
    uv_t = UCS_to_uv(XYZ_to_UCS(XYZ_t))
    u_t, v_t = uv_t[..., 0], uv_t[..., 1]

    XYZ_r = msds_to_XYZ(r_values, cmfs, method="Integration", shape=cmfs.shape)
    uv_r = UCS_to_uv(XYZ_to_UCS(XYZ_r))
    u_r, v_r = uv_r[..., 0], uv_r[..., 1]

    names: List[str] = []
    tcs_values_list = []
    for _key, value in sorted(INDEXES_TO_NAMES_TCS[method].items()):
        if value not in sds_tcs:
            continue
        names.append(sds_tcs[value].name)
        tcs_values_list.append(sds_tcs[value].values)

    xp = array_namespace(XYZ_t, t_values)
    tcs_values = xp.stack(
        [xp_as_float_array(values, xp=xp, like=XYZ_t) for values in tcs_values_list]
    )

    # Vectorised :math:`XYZ_{tcs}` across the test colour samples; the
    # ``100 / Y_t`` factor recovers the reflectance-under-illuminant scale
    # of :func:`sd_to_XYZ(sd_tcs, cmfs, sd_t)`.
    sds_tcs_t = tcs_values * t_values[..., None, :]
    XYZ_tcs = msds_to_XYZ(
        sds_tcs_t,
        cmfs,
        method="Integration",
        shape=cmfs.shape,
    ) * (100 / XYZ_t[..., 1:2, None])

    xyY_tcs = XYZ_to_xyY(XYZ_tcs)
    uv_tcs = UCS_to_uv(XYZ_to_UCS(XYZ_tcs))
    u_tcs, v_tcs = uv_tcs[..., 0], uv_tcs[..., 1]

    if chromatic_adaptation:

        def c(x: NDArrayFloat, y: NDArrayFloat) -> NDArrayFloat:
            """Compute the :math:`c` term."""

            with sdiv_mode():
                return sdiv(4 - x - 10 * y, y)

        def d(x: NDArrayFloat, y: NDArrayFloat) -> NDArrayFloat:
            """Compute the :math:`d` term."""

            with sdiv_mode():
                return sdiv(1.708 * y + 0.404 - 1.481 * x, y)

        c_t, d_t = c(u_t, v_t), d(u_t, v_t)
        c_r, d_r = c(u_r, v_r), d(u_r, v_r)
        tcs_c, tcs_d = c(u_tcs, v_tcs), d(u_tcs, v_tcs)

        with sdiv_mode():
            c_r_c_t = sdiv(c_r, c_t)[..., None]
            d_r_d_t = sdiv(d_r, d_t)[..., None]

        # NOTE: ``uv_tcs`` keeps the pre-adaptation value; the adapted
        # ``u``, ``v`` only feed the ``U``, ``V`` derivation below.
        u_tcs = (10.872 + 0.404 * c_r_c_t * tcs_c - 4 * d_r_d_t * tcs_d) / (
            16.518 + 1.481 * c_r_c_t * tcs_c - d_r_d_t * tcs_d
        )
        v_tcs = 5.52 / (16.518 + 1.481 * c_r_c_t * tcs_c - d_r_d_t * tcs_d)

    W_tcs = 25 * spow(xyY_tcs[..., -1], 1 / 3) - 17
    U_tcs = 13 * W_tcs * (u_tcs - u_r[..., None])
    V_tcs = 13 * W_tcs * (v_tcs - v_r[..., None])
    UVW_tcs = tstack([U_tcs, V_tcs, W_tcs])

    return names, XYZ_tcs, uv_tcs, UVW_tcs


def tcs_colorimetry_data(
    sd_t: SpectralDistribution,
    sd_r: SpectralDistribution,
    sds_tcs: Dict[str, SpectralDistribution],
    cmfs: MultiSpectralDistributions,
    chromatic_adaptation: bool = False,
    method: Literal["CIE 1995", "CIE 2024"] | str = "CIE 1995",
) -> Tuple[DataColorimetry_TCS, ...]:
    """
    Compute the *test colour samples* colorimetry data.

    Parameters
    ----------
    sd_t
        Test spectral distribution.
    sd_r
        Reference spectral distribution.
    sds_tcs
        *Test colour samples* spectral reflectance distributions.
    cmfs
        Standard observer colour matching functions.
    chromatic_adaptation
        Perform chromatic adaptation.

    Returns
    -------
    :class:`tuple`
        *Test colour samples* colorimetry data.
    """

    xp = array_namespace(sd_t.values)
    names, XYZ_tcs, uv_tcs, UVW_tcs = _tcs_colorimetry_data(
        xp_as_float_array(sd_t.values, xp=xp),
        xp_as_float_array(sd_r.values, xp=xp),
        sds_tcs,
        cmfs,
        chromatic_adaptation,
        method,
    )

    return tuple(
        DataColorimetry_TCS(name, XYZ_tcs[i], uv_tcs[i], UVW_tcs[i])
        for i, name in enumerate(names)
    )


def colour_rendering_indexes(
    test_data: Tuple[DataColorimetry_TCS, ...],
    reference_data: Tuple[DataColorimetry_TCS, ...],
) -> Dict[int, DataColourQualityScale_TCS]:
    """
    Compute the *test colour samples* rendering indexes :math:`Q_a`.

    Parameters
    ----------
    test_data
        Test data colorimetry for the *test colour samples*.
    reference_data
        Reference data colorimetry for the *test colour samples*.

    Returns
    -------
    :class:`dict`
        *Test colour samples* *Colour Rendering Index* (CRI) values
        mapped by sample number.
    """

    Q_as = {}
    for i in range(len(test_data)):
        Q_as[i + 1] = DataColourQualityScale_TCS(
            test_data[i].name,
            100
            - 4.6
            * as_float_scalar(
                euclidean_distance(reference_data[i].UVW, test_data[i].UVW),
            ),
        )

    return Q_as
