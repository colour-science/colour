# -*- coding: utf-8 -*-
"""
ANSI/IES TM-30-18 Colour Fidelity Index
=======================================

Defines *ANSI/IES TM-30-18 Colour Fidelity Index* (CFI) computation objects:

- :class:`TM_30_18_Specification`
- :func:`colour_fidelity_index_TM_30_18`

References
----------
- TODO
"""

import numpy as np
from collections import namedtuple

from colour.quality import colour_fidelity_index_CFI2017
from colour.quality.cfi2017 import intermediate_delta_E_to_R_CFI2017
from colour.utilities import as_float_array


class TM_30_18_Specification(
        namedtuple(
            'TM_30_18_Specification',
            ('name', 'sd_test', 'sd_reference', 'R_f', 'Rs', 'CCT', 'D_uv',
             'colorimetry_data', 'R_g', 'bins', 'averages_test',
             'averages_reference', 'average_norms', 'R_fs', 'R_cs', 'R_hs'))):
    """
    Defines the *ANSI/IES TM-30-18 Colour Fidelity Index* (CFI) colour quality
    specification.

    Parameters
    ----------
    name : unicode
        Name of the test spectral distribution.
    sd_test : SpectralDistribution
        Spectral distribution of the tested illuminant.
    sd_reference : SpectralDistribution
        Spectral distribution of the reference illuminant.
    R_f : numeric
        *Colour Fidelity Index* (CFI) :math:`R_f`.
    Rs : list
        Individual *colour fidelity indexes* data for each sample.
    CCT : numeric
        Correlated colour temperature :math:`T_{cp}`.
    D_uv : numeric
        Distance from the Planckian locus :math:`\\Delta_{uv}`.
    colorimetry_data : tuple
        Colorimetry data for the test and reference computations.
    bins : list of list of int
        List of 16 lists, each containing the indexes of colour samples that
        lie in the respective hue bin.
    averages_test : ndarray, (16, 2)
        Averages of *CAM02-UCS* a', b' coordinates for each hue bin for test
        samples.
    averages_reference : ndarray, (16, 2)
        Averages for reference samples.
    average_norms : ndarray, (16,)
        Distance of averages for reference samples from the origin.
    R_fs : ndarray, (16,)
        Local colour fidelities for each hue bin.
    R_cs : ndarray, (16,)
        Local chromaticity shifts for each hue bin, in percents.
    R_hs : ndarray, (16,)
        Local hue shifts for each hue bin.
    """


def averages_area(averages):
    """
    Computes the area of the polygon formed by hue bin averages.

    Parameters
    ==========
    ndarray, (n, 2)
        Hue bin averages.

    Returns
    =======
    float
        Area of the polygon.
    """

    N = averages.shape[0]

    triangle_areas = np.empty(N)
    for i in range(N):
        u = averages[i, :]
        v = averages[(i + 1) % N, :]
        triangle_areas[i] = (u[0] * v[1] - u[1] * v[0]) / 2

    return np.sum(triangle_areas)


def colour_fidelity_index_TM_30_18(sd_test, additional_data=False):
    """
    Returns the *ANSI/IES TM-30-18 Colour Fidelity Index* (CFI) :math:`R_f`
    of given spectral distribution.

    Parameters
    ----------
    sd_test : SpectralDistribution
        Test spectral distribution.
    additional_data : bool, optional
        Whether to output additional data.

    Returns
    -------
    numeric or TM_30_18_Specification
        *Colour Fidelity Index* (CFI).

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS['FL2']
    >>> colour_fidelity_index_TM_30_18(sd)  # doctest: +ELLIPSIS
    70.1208254...
    """

    if not additional_data:
        return colour_fidelity_index_CFI2017(sd_test, False)

    spec = colour_fidelity_index_CFI2017(sd_test, True)

    # Set up bins based on where the reference a'b' points landed
    bins = [list() for i in range(16)]
    for i, sample in enumerate(spec.colorimetry_data[1]):
        bin_index = int(np.floor(sample.CAM.h / 22.5))
        bins[bin_index].append(i)

    # Compute per-bin a' b' averages
    averages_test = np.empty((16, 2))
    averages_reference = np.empty((16, 2))
    for i in range(16):
        apbps = [spec.colorimetry_data[0][j].Jpapbp[[1, 2]] for j in bins[i]]
        averages_test[i, :] = np.mean(apbps, axis=0)
        apbps = [spec.colorimetry_data[1][j].Jpapbp[[1, 2]] for j in bins[i]]
        averages_reference[i, :] = np.mean(apbps, axis=0)

    # Gamut index
    R_g = 100 * (
        averages_area(averages_test) / averages_area(averages_reference))

    # Local colour fidelities (16 CFIs for each bin)
    bin_delta_Es = [np.mean([spec.delta_Es[bins[i]]]) for i in range(16)]
    R_fs = intermediate_delta_E_to_R_CFI2017(as_float_array(bin_delta_Es))

    # Angles bisecting hue bins
    angles = (22.5 * np.arange(16) + 11.25) / 180 * np.pi
    cosines = np.cos(angles)
    sines = np.sin(angles)

    average_norms = np.linalg.norm(averages_reference, axis=1)
    a_deltas = averages_test[:, 0] - averages_reference[:, 0]
    b_deltas = averages_test[:, 1] - averages_reference[:, 1]

    # Local chromaticity shifts, multiply by 100 so they're in percents
    R_cs = 100 * (a_deltas * cosines + b_deltas * sines) / average_norms

    # Local hue shifts
    R_hs = (-a_deltas * sines + b_deltas * cosines) / average_norms

    return TM_30_18_Specification(
        spec.name, sd_test, spec.sd_reference, spec.R_f, spec.Rs, spec.CCT,
        spec.D_uv, spec.colorimetry_data, R_g, bins, averages_test,
        averages_reference, average_norms, R_fs, R_cs, R_hs)
