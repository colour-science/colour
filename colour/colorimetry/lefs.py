# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**lefs.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *luminous efficiency functions* objects.

**Others:**

"""

from __future__ import unicode_literals

from colour.algebra import get_closest
from colour.colorimetry import (
    SpectralPowerDistribution,
    PHOTOPIC_LEFS,
    SCOTOPIC_LEFS)
from colour.colorimetry.dataset.lefs import MESOPIC_X_DATA

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["mesopic_weighting_function",
           "mesopic_luminous_efficiency_function"]


def mesopic_weighting_function(wavelength,
                               Lp,
                               source="Blue Heavy",
                               method="MOVE",
                               photopic_lef=PHOTOPIC_LEFS.get(
                                   "CIE 1924 Photopic Standard Observer"),
                               scotopic_lef=SCOTOPIC_LEFS.get(
                                   "CIE 1951 Scotopic Standard Observer")):
    """
    Calculates the mesopic weighting function factor at given wavelength.

    Examples::

        >>> mesopic_weighting_function(500, 0.2)
        0.70522

    :param wavelength: Wavelength to calculate the mesopic weighting function \
    factor.
    :type wavelength: int or float
    :param Lp: Photopic luminance.
    :type Lp: float
    :param source: Light source colour temperature.
    :type source: unicode ("Blue Heavy", "Red Heavy")
    :param method: Method to calculate the weighting factor.
    :type method: unicode ("MOVE", "LRC")
    :param photopic_lef: *V* photopic luminous efficiency function.
    :type photopic_lef: SpectralPowerDistribution
    :param scotopic_lef: *V'* scotopic luminous efficiency function.
    :type scotopic_lef: SpectralPowerDistribution
    :return: Mesopic weighting function factor.
    :rtype: float

    References:

    -  http://en.wikipedia.org/wiki/Mesopic#Mesopic_weighting_function \
    (Last accessed 20 June 2014)
    """

    for function in (photopic_lef, scotopic_lef):
        if function.get(wavelength) is None:
            raise KeyError(
                "'{0} nm' wavelength not available in '{1}' luminous efficiency function with '{2}' shape!".format(
                    wavelength,
                    function.name,
                    function.shape))

    mesopic_x_luminance_values = sorted(MESOPIC_X_DATA.keys())
    index = mesopic_x_luminance_values.index(
        get_closest(mesopic_x_luminance_values, Lp))
    x = MESOPIC_X_DATA.get(
        mesopic_x_luminance_values[index]).get(source).get(method)

    Vm = ((1. - x) *
          scotopic_lef.get(wavelength) + x * photopic_lef.get(wavelength))

    return Vm


def mesopic_luminous_efficiency_function(Lp,
                                         source="Blue Heavy",
                                         method="MOVE",
                                         photopic_lef=PHOTOPIC_LEFS.get(
                                             "CIE 1924 Photopic Standard Observer"),
                                         scotopic_lef=SCOTOPIC_LEFS.get(
                                             "CIE 1951 Scotopic Standard Observer")):
    """
    Converts given spectral power distribution to *CIE XYZ* colourspace using
    given colour matching functions and illuminant.

    Examples::

        >>> mesopic_luminous_efficiency_function(0.2)
        <colour.colorimetry.spectrum.SpectralPowerDistribution at 0x105f606d0>

    :param Lp: Photopic luminance.
    :type Lp: float
    :param source: Light source colour temperature.
    :type source: unicode ("Blue Heavy", "Red Heavy")
    :param method: Method to calculate the weighting factor.
    :type method: unicode ("MOVE", "LRC")
    :param photopic_lef: *V* photopic luminous efficiency function.
    :type photopic_lef: SpectralPowerDistribution
    :param scotopic_lef: *V'* scotopic luminous efficiency function.
    :type scotopic_lef: SpectralPowerDistribution
    :return: Mesopic luminous efficiency function.
    :rtype: SpectralPowerDistribution

    References:

    -  http://en.wikipedia.org/wiki/Mesopic#Mesopic_weighting_function\
    (Last accessed 20 June 2014)
    """

    photopic_lef_shape, scotopic_lef_shape = photopic_lef.shape, scotopic_lef.shape
    start, end, steps = (max(photopic_lef_shape[0], scotopic_lef_shape[0]),
                         min(photopic_lef_shape[1], scotopic_lef_shape[1]),
                         max(photopic_lef_shape[2], scotopic_lef_shape[2]))

    spd_data = dict((i,
                     mesopic_weighting_function(
                         i,
                         Lp,
                         source,
                         method,
                         photopic_lef,
                         scotopic_lef))
                    for i in range(start, end, steps))

    spd = SpectralPowerDistribution(
        "{0} Lp Mesopic Luminous Efficiency Function".format(Lp),
        spd_data)

    return spd.normalise()