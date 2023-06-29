"""
ANSI/IES TM-30-18 Colour Fidelity Index
=======================================

Defines the *ANSI/IES TM-30-18 Colour Fidelity Index* (CFI) computation
objects:

- :class:`colour.quality.ColourQuality_Specification_ANSIIESTM3018`
- :func:`colour.quality.colour_fidelity_index_ANSIIESTM3018`

References
----------
-   :cite:`ANSI2018` : ANSI, & IES Color Committee. (2018). ANSI/IES TM-30-18 -
    IES Method for Evaluating Light Source Color Rendition.
    ISBN:978-0-87995-379-9
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from colour.colorimetry import SpectralDistribution
from colour.hints import ArrayLike, List, NDArrayFloat, Tuple, cast
from colour.quality import colour_fidelity_index_CIE2017
from colour.quality.cfi2017 import (
    ColourRendering_Specification_CIE2017,
    DataColorimetry_TCS_CIE2017,
    delta_E_to_R_f,
)
from colour.utilities import as_float_array, as_float_scalar

from colour.utilities.array import as_int_array


@dataclass
class ColourQuality_Specification_ANSIIESTM3018:
    """
    Define the *ANSI/IES TM-30-18 Colour Fidelity Index* (CFI) colour quality
    specification.

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
        List of 16 lists, each containing the indexes of colour samples that
        lie in the respective hue bin.
    averages_test
        Averages of *CAM02-UCS* a', b' coordinates for each hue bin for test
        samples.
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
    colorimetry_data: Tuple[
        DataColorimetry_TCS_CIE2017, DataColorimetry_TCS_CIE2017
    ]
    R_g: float
    bins: List[List[int]]
    averages_test: NDArrayFloat
    averages_reference: NDArrayFloat
    average_norms: NDArrayFloat
    R_fs: NDArrayFloat
    R_cs: NDArrayFloat
    R_hs: NDArrayFloat


def colour_fidelity_index_ANSIIESTM3018(
    sd_test: SpectralDistribution, additional_data: bool = False
) -> (
    float
    | ColourQuality_Specification_ANSIIESTM3018
    | ColourRendering_Specification_CIE2017
):
    """
    Return the *ANSI/IES TM-30-18 Colour Fidelity Index* (CFI) :math:`R_f`
    of given spectral distribution.

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    additional_data
        Whether to output additional data.

    Returns
    -------
    :class:`float` or \
:class:`colour.quality.ColourQuality_Specification_ANSIIESTM3018`
        *ANSI/IES TM-30-18 Colour Fidelity Index* (CFI).

    References
    ----------
    :cite:`ANSI2018`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS["FL2"]
    >>> colour_fidelity_index_ANSIIESTM3018(sd)  # doctest: +ELLIPSIS
    70.1208...
    """

    if not additional_data:
        return colour_fidelity_index_CIE2017(sd_test, False)

    specification = cast(
        ColourRendering_Specification_CIE2017,
        colour_fidelity_index_CIE2017(sd_test, True),
    )

    # Setup bins based on where the reference a'b' points are located.
    bins = as_int_array(
        np.floor(specification.colorimetry_data[1].JMh[:, 2] / 22.5)
    )

    bin_mask = bins == np.arange(16).reshape(-1, 1)

    # We will use the bin mask laster with numpy shape broadcasting and argmean
    # to skip a list comprehension and keep all the mean calculation in numpy's
    # core.
    #
    # To skip the list comprehension for getting the bins list we can use this
    # convenient snippet.
    #
    # https://stackoverflow.com/a/43094244
    bin_mask = np.choose(bin_mask, [np.nan, 1])

    # Per-bin a'b' averages.
    test_apbp = as_float_array(specification.colorimetry_data[0].Jpapbp[:, 1:])
    ref_apbp = as_float_array(specification.colorimetry_data[1].Jpapbp[:, 1:])

    # Tile the apbp data in the 3rd dimension and use broadcasting to place
    # each bin mask along third dimension. By multiplying these matrices
    # together, numpy automatically expands the apbp data in the third
    # dimension and multiplies by the nan-filled bin mask. Finally nanmean can
    # compute the bin mean apbp positions with the appropriate axis argument.
    averages_test = np.transpose(
        np.nanmean(
            np.transpose(bin_mask).reshape((99, 1, 16))
            * test_apbp.reshape((*ref_apbp.shape, 1)),
            axis=0,
        )
    )
    averages_reference = np.transpose(
        np.nanmean(
            np.transpose(bin_mask).reshape((99, 1, 16))
            * ref_apbp.reshape((*ref_apbp.shape, 1)),
            axis=0,
        )
    )

    # Gamut Index.
    R_g = 100 * (
        averages_area(averages_test) / averages_area(averages_reference)
    )

    # Local colour fidelity indexes, i.e. 16 CFIs for each bin.
    bin_delta_E_s = np.nanmean(
        specification.delta_E_s.reshape(1, -1) * bin_mask, axis=1
    )
    R_fs = as_float_array(delta_E_to_R_f(bin_delta_E_s))

    # Angles bisecting the hue bins.
    angles = (22.5 * np.arange(16) + 11.25) / 180 * np.pi
    cosines = np.cos(angles)
    sines = np.sin(angles)

    average_norms = np.linalg.norm(averages_reference, axis=1)
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
        bins.tolist(),
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

    N = averages.shape[0]

    triangle_areas = np.empty(N)
    for i in range(N):
        u = averages[i, :]
        v = averages[(i + 1) % N, :]
        triangle_areas[i] = (u[0] * v[1] - u[1] * v[0]) / 2

    return as_float_scalar(np.sum(triangle_areas))
