# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**cri.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *colour rendering index* calculation objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy
from collections import namedtuple

import colour.algebra.common
import colour.computation.blackbody
import colour.computation.colourspaces.cie_ucs
import colour.computation.difference
import colour.computation.illuminants
import colour.computation.temperature
import colour.computation.colourspaces.cie_xyy
import colour.computation.tristimulus
import colour.dataset.cmfs
import colour.dataset.tcs

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TSC_COLORIMETRY_DATA_NXYZUVUVW",
           "get_colour_rendering_index"]


TSC_COLORIMETRY_DATA_NXYZUVUVW = namedtuple("TscColorimetryData_nXYZuvUVW", ("name", "XYZ", "uv", "UVW"))


def __get_tcs_colorimetry_data(test_spd, reference_spd, tsc_spds, cmfs, chromatic_adaptation=False):
    """
    Returns the *test colour samples* colorimetry data.

    :param test_spd: Test spectral power distribution.
    :type test_spd: SpectralPowerDistribution
    :param reference_spd: Reference spectral power distribution.
    :type reference_spd: SpectralPowerDistribution
    :param tsc_spds: Test colour samples.
    :type tsc_spds: dict
    :param cmfs: Standard observer colour matching functions.
    :type cmfs: XYZ_ColourMatchingFunctions
    :param chromatic_adaptation: Perform chromatic adaptation.
    :type chromatic_adaptation: bool
    :return: *Test colour samples* colorimetry data.
    :rtype: list
    """

    test_XYZ = colour.computation.tristimulus.spectral_to_XYZ(test_spd, cmfs)
    test_uv = numpy.ravel(
        colour.computation.colourspaces.cie_ucs.UCS_to_uv(colour.computation.colourspaces.cie_ucs.XYZ_to_UCS(test_XYZ)))
    test_u, test_v = test_uv[0], test_uv[1]

    reference_XYZ = colour.computation.tristimulus.spectral_to_XYZ(reference_spd, cmfs)
    reference_uv = numpy.ravel(
        colour.computation.colourspaces.cie_ucs.UCS_to_uv(colour.computation.colourspaces.cie_ucs.XYZ_to_UCS(reference_XYZ)))
    reference_u, reference_v = reference_uv[0], reference_uv[1]

    tcs_data = []
    for key, value in sorted(colour.dataset.tcs.TCS_INDEXES_TO_NAMES.iteritems()):
        tcs_spd = tsc_spds.get(value)
        tcs_XYZ = colour.computation.tristimulus.spectral_to_XYZ(tcs_spd, cmfs, test_spd)
        tcs_xyY = numpy.ravel(colour.computation.colourspaces.cie_xyy.XYZ_to_xyY(tcs_XYZ))
        tcs_uv = numpy.ravel(
            colour.computation.colourspaces.cie_ucs.UCS_to_uv(colour.computation.colourspaces.cie_ucs.XYZ_to_UCS(tcs_XYZ)))
        tcs_u, tcs_v = tcs_uv[0], tcs_uv[1]

        if chromatic_adaptation:
            get_c = lambda x, y: (4. - x - 10. * y) / y
            get_d = lambda x, y: (1.708 * y + 0.404 - 1.481 * x) / y

            test_c, test_d = get_c(test_u, test_v), get_d(test_u, test_v)
            reference_c, reference_d = get_c(reference_u, reference_v), get_d(reference_u, reference_v)
            tcs_c, tcs_d = get_c(tcs_u, tcs_v), get_d(tcs_u, tcs_v)
            tcs_u = (10.872 + 0.404 * reference_c / test_c * tcs_c - 4 * reference_d / test_d * tcs_d) / \
                    (16.518 + 1.481 * reference_c / test_c * tcs_c - reference_d / test_d * tcs_d)
            tcs_v = 5.52 / (16.518 + 1.481 * reference_c / test_c * tcs_c - reference_d / test_d * tcs_d)

        tcs_W = 25. * tcs_xyY[-1] ** (1. / 3.) - 17.
        tcs_U = 13. * tcs_W * (tcs_u - reference_u)
        tcs_V = 13. * tcs_W * (tcs_v - reference_v)

        tcs_data.append(TSC_COLORIMETRY_DATA_NXYZUVUVW(tcs_spd.name,
                                                       tcs_XYZ,
                                                       tcs_uv,
                                                       numpy.array([tcs_U, tcs_V, tcs_W])))

    return tcs_data


def __get_colour_rendering_indexes(test_data, reference_data):
    """
    Returns the *test colour samples* rendering indexes.

    :param test_data: Test data.
    :type test_data: list
    :param reference_data: Reference data.
    :type reference_data: list
    :return: *Test colour samples* colour rendering indexes.
    :rtype: dict
    """

    colour_rendering_indexes = {}
    for i in range(len(test_data)):
        colour_rendering_indexes[i + 1] = 100. - 4.6 * numpy.linalg.norm(reference_data[i].UVW - test_data[i].UVW)
    return colour_rendering_indexes


def get_colour_rendering_index(test_spd, additional_data=False):
    """
    Returns the *colour rendering index* of given spectral power distribution.

    References: http://cie2.nist.gov/TC1-69/NIST%20CQS%20simulation%207.4.xls, http://onlinelibrary.wiley.com/store/10.1002/9781119975595.app7/asset/app7.pdf?v=1&t=hw7zl300&s=060f34ef1feb8bfa754b9c63c68bcc0808ac6730

    Usage::

        >>> spd = colour.ILLUMINANTS_RELATIVE_SPDS.get("F2")
        >>> get_colour_rendering_index(spd)
        64.1507331494

    :param test_spd: Test spectral power distribution.
    :type test_spd: SpectralPowerDistribution
    :param additional_data: Output additional data.
    :type additional_data: bool
    :return: Colour rendering index, Tsc data.
    :rtype: float or (float, dict)
    """

    cmfs = colour.dataset.cmfs.STANDARD_OBSERVERS_CMFS.get("CIE 1931 2 Degree Standard Observer")

    start, end, steps = cmfs.shape
    test_spd = test_spd.clone().align(start, end, steps)

    tcs_spds = {}
    for index, tcs_spd in sorted(colour.dataset.tcs.TCS_SPDS.iteritems()):
        tcs_spds[index] = tcs_spd.clone().align(start, end, steps)

    XYZ = colour.computation.tristimulus.spectral_to_XYZ(test_spd, cmfs)
    uv = colour.computation.colourspaces.cie_ucs.UCS_to_uv(colour.computation.colourspaces.cie_ucs.XYZ_to_UCS(XYZ))
    CCT, Duv = colour.computation.temperature.uv_to_CCT_robertson(uv)

    if CCT < 5000.:
        reference_spd = colour.computation.blackbody.blackbody_spectral_power_distribution(CCT, *cmfs.shape)
    else:
        xy = colour.computation.temperature.CCT_to_xy_illuminant_D(CCT)
        reference_spd = colour.computation.illuminants.D_illuminant_relative_spd(xy)
        reference_spd.align(start, end, steps)

    test_tcs_colorimetry_data = __get_tcs_colorimetry_data(test_spd,
                                                           reference_spd,
                                                           tcs_spds,
                                                           cmfs,
                                                           chromatic_adaptation=True)
    reference_tcs_colorimetry_data = __get_tcs_colorimetry_data(reference_spd, reference_spd, tcs_spds, cmfs)

    colour_rendering_indexes = __get_colour_rendering_indexes(test_tcs_colorimetry_data, reference_tcs_colorimetry_data)

    colour_rendering_index = numpy.average(
        [v for k, v in colour_rendering_indexes.iteritems() if k in (1, 2, 3, 4, 5, 6, 7, 8)])

    if additional_data:
        return colour_rendering_index, \
               colour_rendering_indexes, \
               [test_tcs_colorimetry_data,
                reference_tcs_colorimetry_data]
    else:
        return colour_rendering_index