#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour Rendering Index
======================

Defines *colour rendering index* computation objects:

-   :func:`colour_rendering_index`

See Also
--------
`Colour Rendering Index IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/quality/cri.ipynb>`_  # noqa

References
----------
.. [1]  http://cie2.nist.gov/TC1-69/NIST%20CQS%20simulation%207.4.xls
        (Last accessed 10 June 2014)
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.colorimetry import STANDARD_OBSERVERS_CMFS
from colour.colorimetry import (
    D_illuminant_relative_spd,
    blackbody_spd,
    spectral_to_XYZ)
from colour.quality.dataset.tcs import TCS_SPDS, TCS_INDEXES_TO_NAMES
from colour.models import UCS_to_uv, XYZ_to_UCS, XYZ_to_xyY
from colour.temperature import CCT_to_xy_illuminant_D, uv_to_CCT_robertson1968

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TSC_COLORIMETRY_DATA_NXYZUVUVW',
           'colour_rendering_index']

TSC_COLORIMETRY_DATA_NXYZUVUVW = namedtuple('TscColorimetryData_nXYZuvUVW',
                                            ('name', 'XYZ', 'uv', 'UVW'))


def _tcs_colorimetry_data(test_spd,
                          reference_spd,
                          tsc_spds,
                          cmfs,
                          chromatic_adaptation=False):
    """
    Returns the *test colour samples* colorimetry data.

    Parameters
    ----------
    test_spd : SpectralPowerDistribution
        Test spectral power distribution.
    reference_spd : SpectralPowerDistribution
        Reference spectral power distribution.
    tsc_spds : dict
        Test colour samples.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    chromatic_adaptation : bool, optional
        Perform chromatic adaptation.

    Returns
    -------
    list
        *Test colour samples* colorimetry data.
    """

    test_XYZ = spectral_to_XYZ(test_spd, cmfs)
    test_uv = np.ravel(UCS_to_uv(XYZ_to_UCS(test_XYZ)))
    test_u, test_v = test_uv[0], test_uv[1]

    reference_XYZ = spectral_to_XYZ(reference_spd, cmfs)
    reference_uv = np.ravel(UCS_to_uv(XYZ_to_UCS(reference_XYZ)))
    reference_u, reference_v = reference_uv[0], reference_uv[1]

    tcs_data = []
    for key, value in sorted(TCS_INDEXES_TO_NAMES.items()):
        tcs_spd = tsc_spds.get(value)
        tcs_XYZ = spectral_to_XYZ(tcs_spd, cmfs, test_spd)
        tcs_xyY = np.ravel(XYZ_to_xyY(tcs_XYZ))
        tcs_uv = np.ravel(UCS_to_uv(XYZ_to_UCS(tcs_XYZ)))
        tcs_u, tcs_v = tcs_uv[0], tcs_uv[1]

        if chromatic_adaptation:
            c = lambda x, y: (4 - x - 10 * y) / y
            d = lambda x, y: (1.708 * y + 0.404 - 1.481 * x) / y

            test_c, test_d = c(test_u, test_v), d(test_u, test_v)
            reference_c, reference_d = (c(reference_u, reference_v),
                                        d(reference_u, reference_v))
            tcs_c, tcs_d = c(tcs_u, tcs_v), d(tcs_u, tcs_v)
            tcs_u = ((10.872 + 0.404 * reference_c / test_c * tcs_c - 4 *
                      reference_d / test_d * tcs_d) /
                     (16.518 + 1.481 * reference_c / test_c * tcs_c -
                      reference_d / test_d * tcs_d))
            tcs_v = (5.52 / (16.518 + 1.481 * reference_c / test_c * tcs_c -
                             reference_d / test_d * tcs_d))

        tcs_W = 25 * tcs_xyY[-1] ** (1 / 3) - 17
        tcs_U = 13 * tcs_W * (tcs_u - reference_u)
        tcs_V = 13 * tcs_W * (tcs_v - reference_v)

        tcs_data.append(
            TSC_COLORIMETRY_DATA_NXYZUVUVW(tcs_spd.name,
                                           tcs_XYZ,
                                           tcs_uv,
                                           np.array([tcs_U, tcs_V, tcs_W])))

    return tcs_data


def _colour_rendering_indexes(test_data, reference_data):
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

    colour_rendering_indexes = {}
    for i in range(len(test_data)):
        colour_rendering_indexes[i + 1] = 100 - 4.6 * np.linalg.norm(
            reference_data[i].UVW - test_data[i].UVW)
    return colour_rendering_indexes


def colour_rendering_index(test_spd, additional_data=False):
    """
    Returns the *colour rendering index* of given spectral power distribution.

    Parameters
    ----------
    test_spd : SpectralPowerDistribution
        Test spectral power distribution.
    additional_data : bool, optional
        Output additional data.

    Returns
    -------
    numeric or (numeric, dict)
        Colour rendering index, Tsc data.

    Examples
    --------
    >>> from colour import ILLUMINANTS_RELATIVE_SPDS
    >>> spd = ILLUMINANTS_RELATIVE_SPDS.get('F2')
    >>> colour_rendering_index(spd)  # doctest: +ELLIPSIS
    64.1507331...
    """

    cmfs = STANDARD_OBSERVERS_CMFS.get('CIE 1931 2 Degree Standard Observer')

    shape = cmfs.shape
    test_spd = test_spd.clone().align(shape)

    tcs_spds = {}
    for index, tcs_spd in sorted(TCS_SPDS.items()):
        tcs_spds[index] = tcs_spd.clone().align(shape)

    XYZ = spectral_to_XYZ(test_spd, cmfs)
    uv = UCS_to_uv(XYZ_to_UCS(XYZ))
    CCT, Duv = uv_to_CCT_robertson1968(uv)

    if CCT < 5000:
        reference_spd = blackbody_spd(CCT, shape)
    else:
        xy = CCT_to_xy_illuminant_D(CCT)
        reference_spd = D_illuminant_relative_spd(xy)
        reference_spd.align(shape)

    test_tcs_colorimetry_data = _tcs_colorimetry_data(
        test_spd,
        reference_spd,
        tcs_spds,
        cmfs,
        chromatic_adaptation=True)
    reference_tcs_colorimetry_data = _tcs_colorimetry_data(
        reference_spd,
        reference_spd,
        tcs_spds, cmfs)

    colour_rendering_indexes = _colour_rendering_indexes(
        test_tcs_colorimetry_data, reference_tcs_colorimetry_data)

    colour_rendering_index = np.average(
        [v for k, v in colour_rendering_indexes.items()
         if k in (1, 2, 3, 4, 5, 6, 7, 8)])

    if additional_data:
        return (colour_rendering_index,
                colour_rendering_indexes,
                [test_tcs_colorimetry_data, reference_tcs_colorimetry_data])
    else:
        return colour_rendering_index
