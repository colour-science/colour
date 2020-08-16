# -*- coding: utf-8 -*-
"""
CIE 2017 Colour Fidelity Index
==============================

Defines *CIE 2017 Colour Fidelity Index* (CFI) computation objects:

- :class:`TCS_ColorimetryData_CFI2017`
- :class:`CFI2017_Specification`
- :func:`reference_illuminant_CFI2017`
- :func:`intermediate_delta_E_to_R_CFI2017`
- :func:`tcs_colorimetry_data_CFI2017'
- :func:`colour_fidelity_index_CFI2017`

References
----------
- TODO
"""

from __future__ import division, unicode_literals

import numpy as np
import os
from collections import namedtuple

from colour.algebra import euclidean_distance, Extrapolator
from colour.appearance import XYZ_to_CIECAM02, CIECAM02_VIEWING_CONDITIONS
from colour.colorimetry import (
    SpectralShape, SpectralDistribution, MultiSpectralDistributions, sd_to_XYZ,
    sd_blackbody, CMFS, sd_ones, sd_CIE_illuminant_D_series)
from colour.models import XYZ_to_UCS, UCS_to_uv, JMh_CIECAM02_to_CAM02UCS
from colour.temperature import uv_to_CCT, CCT_to_xy
from colour.utilities import usage_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

CFI2017_RESOURCES_DIRECTORY = os.path.join(
    os.path.dirname(__file__), 'datasets')

_TCS_CFI2017_CACHE = {}


def get_tcs_CFI2017(shape):
    """
    Loads the *CIE 2017 test colour sample* dataset appropriate for the given
    spectral shape. Datasets are cached and won't be loaded again on
    subsequent calls to this functions.

    Parameters
    ==========
    shape : SpectralShape
        Spectral shape of the tested illuminant.

    Returns
    =======
    MultiSpectralDistributions
        *Test colour samples* appropriate for the given shape.
    """

    global _TCS_CFI2017_CACHE

    if shape.interval >= 5:
        path = 'tcs_cfi2017_5_nm.csv.gz'
    else:
        path = 'tcs_cfi2017_1_nm.csv.gz'

    if path in _TCS_CFI2017_CACHE:
        return _TCS_CFI2017_CACHE[path]

    data = np.genfromtxt(str(os.path.join(CFI2017_RESOURCES_DIRECTORY, path)))
    labels = ['TCS{} (CIE 2017)'.format(i) for i in range(99)]

    return MultiSpectralDistributions(data[:, 1:], data[:, 0], labels)


class TCS_ColorimetryData_CFI2017(
        namedtuple('TCS_ColorimetryData_CFI2017',
                   ('name', 'XYZ', 'CAM', 'JMh', 'Jpapbp'))):
    """
    Defines the the class storing *test colour samples* colorimetry data.
    """


class CFI2017_Specification(
        namedtuple('CFI_Specfiication_CFI2017',
                   ('name', 'sd_reference', 'R_f', 'Rs', 'CCT', 'D_uv',
                    'colorimetry_data', 'delta_Es'))):
    """
    Defines the *Colour Fidelity Index* (CFI) colour quality specification.

    Parameters
    ----------
    name : unicode
        Name of the test spectral distribution.
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
    delta_Es : ndarray, (16,)
        Colour shifts of samples.
    """


def reference_illuminant_CFI2017(sd, shape, additional_data=False):
    """
    Compute the reference illuminant for a given test illuminant for use in
    *CIE 2017* colour fidelity index computation.


    Parameters
    ==========
    sd : SpectralDistribution
        Spectral distribution of the tested illuminant.
    shape : SpectralShape
        Desired shape of the returned spectral distribution.

    Returns
    =======
    SpectralDistribution
        Reference illuminant for *CIE 2017* colour fidelity index computation.
    float
        Correlated colour temperature :math:`T_{cp}`,
    float
        Distance from the Planckian locus :math:`\\Delta_{uv}`.
    """

    XYZ = sd_to_XYZ(sd)
    UCS = XYZ_to_UCS(XYZ)
    uv = UCS_to_uv(UCS)
    CCT, D_uv = uv_to_CCT(uv, method='Ohno 2013')

    if CCT <= 5000:
        planckian = sd_blackbody(CCT, shape)
    if CCT >= 4000:
        xy = CCT_to_xy(CCT)
        d_series = sd_CIE_illuminant_D_series(xy).copy().align(shape)

    if CCT < 4000:
        sd_reference = planckian
    elif CCT < 5000:
        # SPDs must be normalised so that the mix isn't biased
        Y_planckian = sd_to_XYZ(planckian)[1]
        Y_d_series = sd_to_XYZ(d_series)[1]

        # Linear mixing: 4200 K should be 80% Planckian, 20% D series and so on
        t = (CCT - 4000) / 1000
        values = ((1 - t) * planckian.values / Y_planckian +
                  t * d_series.values / Y_d_series)

        name = ('{:.0f}K Blackbody and CIE Illuminant D series, {:.1f}% mix'
                .format(CCT, 100 * t))
        sd_reference = SpectralDistribution(values, shape.range(), name=name)
    else:
        sd_reference = d_series

    return sd_reference, CCT, D_uv


def intermediate_delta_E_to_R_CFI2017(delta_E):
    """
    Converts from colour-appearance difference to colour fidelity index value.
    Used in *CIE 2017* colour fidelity calculations.

    Parameters
    ==========
    delta_E : float
        Euclidean distance between two colours in *CAM02-UCS* colourspace.

    Returns
    =======
    float
        Corresponding colour fidelity index value.
    """

    return 10 * np.log(np.exp((100 - 6.73 * delta_E) / 10) + 1)


def tcs_colorimetry_data_CFI2017(illuminant, sds_tcs, cmfs):
    """
    Returns the *test colour samples* colorimetry data for *CIE 2017* colour
    fidelity computations.

    Parameters
    ----------
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    sds_tcs : MultiSpectralDistributions
        *Test colour samples* spectral distributions.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.

    Returns
    -------
    list
        *Test colour samples* colorimetry data.
    """

    XYZ_w = sd_to_XYZ(sd_ones(), cmfs, illuminant)
    Y_b = 20
    L_A = 100
    surround = CIECAM02_VIEWING_CONDITIONS['Average']

    tcs_data = []
    for sd in sds_tcs.to_sds():
        XYZ = sd_to_XYZ(sd, cmfs, illuminant)
        CAM = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, True)
        JMh = CAM.J, CAM.M, CAM.h
        Jpapbp = JMh_CIECAM02_to_CAM02UCS(JMh)

        tcs_data.append(
            TCS_ColorimetryData_CFI2017(sd.name, XYZ, CAM, JMh, Jpapbp))

    return tcs_data


def colour_fidelity_index_CFI2017(sd_test, additional_data=False):
    """
    Returns the *CIE 2017 Colour Fidelity Index* (CFI) :math:`R_f`
    of given spectral distribution.

    Parameters
    ----------
    sd_test : SpectralDistribution
        Test spectral distribution.
    additional_data : bool, optional
        Whether to output additional data.

    Returns
    -------
    numeric or CFI2017_Specification
        *Colour Fidelity Index* (CFI).

    Examples
    --------
    >>> from colour.colorimetry import ILLUMINANT_SDS
    >>> sd = ILLUMINANT_SDS['FL2']
    >>> colour_fidelity_index_CFI2017(sd)  # doctest: +ELLIPSIS
    70.1208254...
    """

    if sd_test.shape.start > 380 or sd_test.shape.end < 780:
        # The standard recommends filling the missing values with zeros
        sd_test = sd_test.copy()
        sd_test.extrapolator = Extrapolator
        sd_test.extrapolator_kwargs = {
            'method': 'constant',
            'left': 0,
            'right': 0
        }
        usage_warning('Test spectrum does not cover the 380-780 nm range, '
                      'which is recommended for computing CIE 2017 colour '
                      'fidelity indexes.')

    if sd_test.shape.interval > 5:
        raise ValueError('Test spectrum is not sampled at 5 nm intervals or '
                         'less, which is necessary for computing CIE 2017 '
                         'colour fidelity indexes.')

    shape = SpectralShape(380, 780, sd_test.shape.interval)
    sd_reference, CCT, D_uv = reference_illuminant_CFI2017(sd_test, shape)

    # All computations except CCT calculation use the 10 degree observer.
    cmfs_10 = CMFS['CIE 1964 10 Degree Standard Observer'].copy().align(shape)

    sds_tcs = get_tcs_CFI2017(shape).align(shape)

    test_tcs_colorimetry_data = tcs_colorimetry_data_CFI2017(
        sd_test, sds_tcs, cmfs_10)

    reference_tcs_colorimetry_data = tcs_colorimetry_data_CFI2017(
        sd_reference, sds_tcs, cmfs_10)

    delta_Es = np.empty(99)
    for i in range(99):
        delta_Es[i] = euclidean_distance(
            test_tcs_colorimetry_data[i].Jpapbp,
            reference_tcs_colorimetry_data[i].Jpapbp)

    Rs = intermediate_delta_E_to_R_CFI2017(delta_Es)
    R_f = intermediate_delta_E_to_R_CFI2017(np.mean(delta_Es))

    if additional_data:
        return CFI2017_Specification(
            sd_test.name, sd_reference, R_f, Rs, CCT, D_uv,
            (test_tcs_colorimetry_data, reference_tcs_colorimetry_data),
            delta_Es)
    else:
        return R_f
