#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour Quality Scale
======================

Defines *colour quality scale* computation objects:

-   :func:`colour_quality_scale`

See Also
--------
`Colour Quality Scale IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/quality/cqs.ipynb>`_  # noqa

References
----------
.. [1]  http://opticalengineering.spiedigitallibrary.org/article.aspx?articleid=1096282
        (Last accessed 18 September 2014)
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple
import math

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
    LAB_to_LCHab)
from colour.temperature import CCT_to_xy_illuminant_D, uv_to_CCT_robertson1968
from colour.adaptation import chromatic_adaptation as xyz_chromatic_adaptation

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = []

VS_COLORIMETRY_DATA_NXYZLABC = namedtuple('VsColorimetryData_nXYZLabC',
                                          ('name', 'XYZ', 'Lab', 'C'))


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
    test_xy = XYZ_to_xy(test_XYZ)

    vs_data = []
    for key, value in sorted(VS_INDEXES_TO_NAMES.items()):
        vs_spd = vs_spds.get(value)
        vs_XYZ = spectral_to_XYZ(vs_spd, cmfs, test_spd)

        if chromatic_adaptation is True:
            reference_XYZ = spectral_to_XYZ(reference_spd, cmfs)
            vs_XYZ = xyz_chromatic_adaptation(vs_XYZ, test_XYZ, reference_XYZ, method='CMCCAT2000_CAT')

        vs_CIE_Lab = XYZ_to_Lab(vs_XYZ, illuminant=test_xy)
        _, vs_chromaticity = LAB_to_LCHab(vs_CIE_Lab)

        vs_data.append(
            VS_COLORIMETRY_DATA_NXYZLABC(vs_spd.name,
                                         vs_XYZ,
                                         vs_CIE_Lab,
                                         vs_chromaticity))

    return vs_data


def _colour_quality_spaces(test_data, reference_data):
    """
    Returns the *test colour samples* rendering indexes.

    Parameters
    ----------
    test_data : list
        Test data.
    reference_data : list
        Reference data.

    Returns
    -------
    dict
        *Test colour samples* colour rendering indexes.
    """

    colour_quality_spaces = {}
    for i, _ in enumerate(test_data):
        color_diff = math.sqrt(sum((reference_data[i].Lab - test_data[i].Lab)**2))
        chroma_diff = test_data[i].C - reference_data[i].C

        if chroma_diff > 0:
            color_diff = math.sqrt(color_diff**2 - chroma_diff**2)

        colour_quality_spaces[index] = color_diff
    return colour_quality_spaces