#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour Quality Scale
====================

Defines *colour quality scale* computation objects:

-   :func:`colour_quality_scale`

See Also
--------
`Colour Quality Scale IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/quality/cqs.ipynb>`_  # noqa

References
----------
.. [1]  **Davis, W., & Ohno, Y. (2010)**, *Color Quality Scale*,
        *Optical Engineering, 49(3), 033602-033602*,
        DOI: 10.1117/1.3360335

.. [2]  http://cie2.nist.gov/TC1-69/NIST%20CQS%20simulation%207.4.xls
        (Last accessed 18 September 2014)
"""

from __future__ import division, unicode_literals

import math
import numpy as np
from collections import namedtuple

from colour.colorimetry import STANDARD_OBSERVERS_CMFS
from colour.colorimetry import (
    D_illuminant_relative_spd,
    blackbody_spd,
    spectral_to_XYZ)
from colour.quality.dataset.vs import VS_SPDS, VS_INDEXES_TO_NAMES
from colour.models import (
    UCS_to_uv,
    XYZ_to_UCS,
    XYZ_to_xy,
    XYZ_to_Lab,
    Lab_to_LCHab)
from colour.temperature import CCT_to_xy_illuminant_D, uv_to_CCT_robertson1968
from colour.adaptation import chromatic_adaptation as xyz_chromatic_adaptation

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['VsColorimetryData',
           'colour_quality_scale']


class VsColorimetryData(namedtuple('VsColorimetryData_nXYZLabC',
                                   ('name', 'XYZ', 'Lab', 'C'))):
    """
    Defines the the class holding *VS test colour samples* colorimetry data.
    """


def _vs_colorimetry_data(test_spd,
                         reference_spd,
                         vs_spds,
                         cmfs,
                         chromatic_adaptation=False):
    """
    Returns the *VS test colour samples* colorimetry data.

    Parameters
    ----------
    test_spd : SpectralPowerDistribution
        Test spectral power distribution.
    reference_spd : SpectralPowerDistribution
        Reference spectral power distribution.
    vs_spds : dict
        VS Test colour samples.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    chromatic_adaptation : bool, optional
        Perform chromatic adaptation.

    Returns
    -------
    list
        *VS Test colour samples* colorimetry data.
    """

    test_XYZ = spectral_to_XYZ(test_spd, cmfs)
    test_XYZ = test_XYZ / np.max(test_XYZ)
    test_xy = XYZ_to_xy(test_XYZ)

    vs_data = []
    for key, value in sorted(VS_INDEXES_TO_NAMES.items()):
        vs_spd = vs_spds.get(value)
        vs_XYZ = spectral_to_XYZ(vs_spd, cmfs, test_spd)
        vs_XYZ = vs_XYZ / np.max(vs_XYZ)

        if chromatic_adaptation is True:
            reference_XYZ = spectral_to_XYZ(reference_spd, cmfs)
            reference_XYZ = reference_XYZ / np.max(reference_XYZ)
            vs_XYZ = xyz_chromatic_adaptation(vs_XYZ, test_XYZ, reference_XYZ,
                                              method='CMCCAT2000')

        vs_Lab = XYZ_to_Lab(vs_XYZ, illuminant=test_xy)
        _, vs_chroma, _ = Lab_to_LCHab(vs_Lab)

        vs_data.append(
            VsColorimetryData(vs_spd.name,
                              vs_XYZ,
                              vs_Lab,
                              vs_chroma))
    return vs_data


def _colour_quality_scales(test_data, reference_data):
    """
    Returns the *test colour samples* rendering scales.

    Parameters
    ----------
    test_data : list
        Test data.
    reference_data : list
        Reference data.

    Returns
    -------
    dict
        *Test colour samples* colour rendering scales.
    """

    colour_quality_scales = {}
    for i, _ in enumerate(test_data):
        colour_difference = math.sqrt(
            sum((test_data[i].Lab - reference_data[i].Lab) ** 2))
        chroma_difference = test_data[i].C - reference_data[i].C

        if chroma_difference < 0:
            colour_difference = math.sqrt(
                colour_difference ** 2 - chroma_difference ** 2)

        colour_quality_scales[i + 1] = colour_difference
    return colour_quality_scales


def colour_quality_scale(test_spd, additional_data=False):
    """
    Returns the *colour quality scale* of given spectral power distribution.

    Parameters
    ----------
    test_spd : SpectralPowerDistribution
        Test spectral power distribution.
    additional_data : bool, optional
        Output additional data.

    Returns
    -------
    numeric or (numeric, dict)
        Color quality scale, VS data.

    Examples
    --------
    >>> from colour import ILLUMINANTS_RELATIVE_SPDS
    >>> spd = ILLUMINANTS_RELATIVE_SPDS.get('F2')
    >>> colour_quality_scale(spd)  # doctest: +ELLIPSIS
    58.2579269...
    """

    cmfs = STANDARD_OBSERVERS_CMFS.get('CIE 1931 2 Degree Standard Observer')

    shape = cmfs.shape
    test_spd = test_spd.clone().align(shape)

    vs_spds = {}
    for index, vs_spd in sorted(VS_SPDS.items()):
        vs_spds[index] = vs_spd.clone().align(shape)

    XYZ = spectral_to_XYZ(test_spd, cmfs)
    uv = UCS_to_uv(XYZ_to_UCS(XYZ))
    CCT, _ = uv_to_CCT_robertson1968(uv)

    if CCT < 5000:
        reference_spd = blackbody_spd(CCT, shape)
    else:
        xy = CCT_to_xy_illuminant_D(CCT)
        reference_spd = D_illuminant_relative_spd(xy)
        reference_spd.align(shape)

    test_vs_colorimetry_data = _vs_colorimetry_data(
        test_spd,
        reference_spd,
        vs_spds,
        cmfs,
        chromatic_adaptation=True)
    reference_vs_colorimetry_data = _vs_colorimetry_data(
        reference_spd,
        reference_spd,
        vs_spds,
        cmfs)

    colour_quality_scales = _colour_quality_scales(
        test_vs_colorimetry_data, reference_vs_colorimetry_data)

    colour_difference_RMS = math.sqrt(1 / len(colour_quality_scales) *
                                      sum([v ** 2 for k, v in
                                           colour_quality_scales.items()]))

    CQS_RMS = 100 - 3.1 * colour_difference_RMS
    CQS_scaled = 10 * np.log(np.exp(CQS_RMS / 10) + 1)

    if CCT < 3500:
        M_CCT = (CCT ** 3 * (9.2672e-11) - CCT ** 2 * (8.3959e-7) +
                 CCT * (0.00255) - 1.612)
    else:
        M_CCT = 1

    colour_quality_scale = M_CCT * CQS_scaled

    if additional_data:
        return (colour_quality_scale,
                colour_quality_scales,
                [test_vs_colorimetry_data, reference_vs_colorimetry_data])
    else:
        return colour_quality_scale
