"""
ANSI/IES TM-30-18 Colour Fidelity Index
=======================================

Define the *ANSI/IES TM-30-18 Colour Fidelity Index* (CFI) computation
objects.

- :class:`colour.quality.ColourQuality_Specification_ANSIIESTM3018`
- :func:`colour.quality.colour_fidelity_index_ANSIIESTM3018`

References
----------
-   :cite:`ANSI2018` : ANSI, & IES Color Committee. (2018). ANSI/IES TM-30-18 -
    IES Method for Evaluating Light Source Color Rendition.
    ISBN:978-0-87995-379-9
-   :cite:`VincentJ2017` : Vincent J. (2017). Is there any numpy group by
    function? Retrieved June 30, 2023, from https://stackoverflow.com/a/43094244
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Literal, NDArrayFloat, NDArrayInt, Tuple

from colour.colorimetry import MultiSpectralDistributions, SpectralDistribution
from colour.quality import colour_fidelity_index_CIE2017
from colour.quality.cfi2017 import (
    DataColorimetry_TCS_CIE2017,
    delta_E_to_R_f,
)
from colour.utilities import (
    array_namespace,
    as_float_array,
    as_float_scalar,
    as_int_array,
    xp_as_float_array,
    xp_matrix_transpose,
    xp_nanmean,
    xp_reshape,
)


@dataclass
class ColourQuality_Specification_ANSIIESTM3018:
    """
    Define the *ANSI/IES TM-30-18 Colour Fidelity Index* (CFI) colour
    quality specification.

    Parameters
    ----------
    name
        Name of the test spectral distribution.
    sd_test
        Spectral distribution of the tested illuminant.
    sd_reference
        Spectral distribution of the reference illuminant.
    R_f
        *Colour Fidelity Index* (CFI) :math:`R_f`.
    R_s
        Individual *colour fidelity indexes* data for each sample.
    CCT
        Correlated colour temperature :math:`T_{cp}`.
    D_uv
        Distance from the Planckian locus :math:`\\Delta_{uv}`.
    colorimetry_data
        Colorimetry data for the test and reference computations.
    R_g
        Gamut index :math:`R_g`.
    bins
        List of 16 lists, each containing the indexes of colour samples
        that lie in the respective hue bin.
    averages_test
        Averages of *CAM02-UCS* a', b' coordinates for each hue bin for
        test samples.
    averages_reference
        Averages for reference samples.
    average_norms
        Distance of averages for reference samples from the origin.
    R_fs
        Local colour fidelities for each hue bin.
    R_cs
        Local chromaticity shifts for each hue bin, in percents.
    R_hs
        Local hue shifts for each hue bin.
    """

    name: str
    sd_test: SpectralDistribution
    sd_reference: SpectralDistribution
    R_f: float
    R_s: NDArrayFloat
    CCT: float
    D_uv: float
    colorimetry_data: Tuple[DataColorimetry_TCS_CIE2017, DataColorimetry_TCS_CIE2017]
    R_g: float
    bins: NDArrayInt
    averages_test: NDArrayFloat
    averages_reference: NDArrayFloat
    average_norms: NDArrayFloat
    R_fs: NDArrayFloat
    R_cs: NDArrayFloat
    R_hs: NDArrayFloat


@typing.overload
def colour_fidelity_index_ANSIIESTM3018(
    sd_test: SpectralDistribution, additional_data: Literal[True] = True
) -> ColourQuality_Specification_ANSIIESTM3018: ...


@typing.overload
def colour_fidelity_index_ANSIIESTM3018(
    sd_test: SpectralDistribution, *, additional_data: Literal[False]
) -> float: ...


@typing.overload
def colour_fidelity_index_ANSIIESTM3018(
    sd_test: SpectralDistribution, additional_data: Literal[False]
) -> float: ...


@typing.overload
def colour_fidelity_index_ANSIIESTM3018(
    sd_test: MultiSpectralDistributions,
    additional_data: Literal[False] = False,
) -> NDArrayFloat: ...


def colour_fidelity_index_ANSIIESTM3018(
    sd_test: SpectralDistribution | MultiSpectralDistributions,
    additional_data: bool = False,
) -> float | NDArrayFloat | ColourQuality_Specification_ANSIIESTM3018:
    """
    Compute the *ANSI/IES TM-30-18 Colour Fidelity Index* (CFI) :math:`R_f`
    for the specified test spectral distribution.

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
    :class:`colour.quality.ColourQuality_Specification_ANSIIESTM3018`
        *ANSI/IES TM-30-18 Colour Fidelity Index* (CFI).

    References
    ----------
    :cite:`ANSI2018`, :cite:`VincentJ2017`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS["FL2"]
    >>> colour_fidelity_index_ANSIIESTM3018(sd)  # doctest: +ELLIPSIS
    np.float64(70.1208244...)
    """

    if not additional_data:
        return colour_fidelity_index_CIE2017(sd_test, False)

    if not isinstance(sd_test, SpectralDistribution):
        error = (
            '"additional_data=True" is not supported when "sd_test" is a '
            '"MultiSpectralDistributions" instance.'
        )
        raise NotImplementedError(error)

    specification = colour_fidelity_index_CIE2017(sd_test, True)

    # Setup bins based on where the reference a'b' points are located.
    JMh = specification.colorimetry_data[1].JMh

    xp = array_namespace(JMh)

    bins = as_int_array(xp.floor(JMh[:, 2] / 22.5))

    arange_16 = xp.arange(16)
    bin_mask = bins == xp_reshape(arange_16, (-1, 1), xp=xp)

    # "bin_mask" is used later with broadcasting and "nanmean" to skip a list
    # comprehension and keep all the mean calculation vectorised as per
    # :cite:`VincentJ2017`.
    bin_mask = xp.where(bin_mask == 0, float("nan"), 1.0)

    # Per-bin a'b' averages.
    test_apbp = specification.colorimetry_data[0].Jpapbp[:, 1:]
    ref_apbp = specification.colorimetry_data[1].Jpapbp[:, 1:]

    # Tile the "apbp" data in the third dimension and use broadcasting to place
    # each bin mask along the third dimension. By multiplying these matrices
    # together, the backend automatically expands the apbp data in the third
    # dimension and multiplies by the nan-filled bin mask. Finally,
    # "nanmean" can compute the bin mean apbp positions with the appropriate
    # axis argument.
    averages_test = xp_matrix_transpose(
        xp_nanmean(
            xp_reshape(xp_matrix_transpose(bin_mask, xp=xp), (99, 1, 16), xp=xp)
            * xp_reshape(test_apbp, (*ref_apbp.shape, 1), xp=xp),
            axis=0,
            xp=xp,
        ),
        xp=xp,
    )
    averages_reference = xp_matrix_transpose(
        xp_nanmean(
            xp_reshape(xp_matrix_transpose(bin_mask, xp=xp), (99, 1, 16), xp=xp)
            * xp_reshape(ref_apbp, (*ref_apbp.shape, 1), xp=xp),
            axis=0,
            xp=xp,
        ),
        xp=xp,
    )

    # Gamut Index.
    R_g = 100 * (averages_area(averages_test) / averages_area(averages_reference))

    # Local colour fidelity indexes, i.e., 16 CFIs for each bin.
    bin_delta_E_s = xp_nanmean(
        xp_reshape(specification.delta_E_s, (1, -1), xp=xp) * bin_mask, axis=1, xp=xp
    )
    R_fs = delta_E_to_R_f(bin_delta_E_s)

    # Angles bisecting the 16 hue bins of width ``360 / 16 = 22.5`` degrees,
    # offset by half a bin (*ANSI/IES TM-30-18*, Section 4.5).
    # ``xp.arange`` yields an integer array whose promotion would adopt the
    # backend default float dtype, e.g. float32 for stock *PyTorch*; the
    # samples are anchored to the *Colour* default float dtype instead.
    angles = (
        (22.5 * xp_as_float_array(xp.arange(16), xp=xp, like=averages_test) + 11.25)
        / 180
        * xp.pi
    )
    cosines = xp.cos(angles)
    sines = xp.sin(angles)

    average_norms = xp.linalg.vector_norm(averages_reference, axis=1)
    a_deltas = averages_test[:, 0] - averages_reference[:, 0]
    b_deltas = averages_test[:, 1] - averages_reference[:, 1]

    # Local chromaticity shifts, multiplied by 100 to obtain percentages.
    R_cs = 100 * (a_deltas * cosines + b_deltas * sines) / average_norms

    # Local hue shifts.
    R_hs = (-a_deltas * sines + b_deltas * cosines) / average_norms

    return ColourQuality_Specification_ANSIIESTM3018(
        specification.name,
        sd_test,
        specification.sd_reference,
        specification.R_f,
        specification.R_s,
        specification.CCT,
        specification.D_uv,
        specification.colorimetry_data,
        R_g,
        bins,
        averages_test,
        averages_reference,
        average_norms,
        R_fs,
        R_cs,
        R_hs,
    )


def averages_area(averages: ArrayLike) -> float:
    """
    Compute the area of the polygon formed by the hue bin averages.

    Parameters
    ----------
    averages
        Hue bin averages.

    Returns
    -------
    :class:`float`
        Area of the polygon.
    """

    averages = as_float_array(averages)

    xp = array_namespace(averages)

    # Vectorized shoelace formula
    u = averages
    v = xp.roll(averages, -1, axis=0)
    triangle_areas = (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2

    return as_float_scalar(xp.sum(triangle_areas))
