# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**photometry.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package color spectral *photometry* objects.

**Others:**

"""

from __future__ import unicode_literals

import color.algebra.common
import color.spectrum.lefs
import color.utilities.exceptions
import color.utilities.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["MESOPIC_X_DATA",
           "mesopic_weighting_function"]

LOGGER = color.utilities.verbose.install_logger()

MESOPIC_X_DATA = {
    0.01: {
        "Blue Heavy": {
            "MOVE": 0.13, "LRC": 0.04},
        "Red Heavy": {
            "MOVE": 0.00, "LRC": 0.01}},
    0.1: {
        "Blue Heavy": {
            "MOVE": 0.42, "LRC": 0.28},
        "Red Heavy": {
            "MOVE": 0.34, "LRC": 0.11}},
    1.0: {
        "Blue Heavy": {
            "MOVE": 0.70, "LRC": 1.00},
        "Red Heavy": {
            "MOVE": 0.68, "LRC": 1.00}},
    10: {
        "Blue Heavy": {
            "MOVE": 0.98, "LRC": 1.00},
        "Red Heavy": {
            "MOVE": 0.98, "LRC": 1.00}}}


def mesopic_weighting_function(wavelength,
                               Lp,
                               source="Blue Heavy",
                               method="MOVE",
                               photopic_lef=color.spectrum.lefs.PHOTOPIC_LEFS.get(
                                   "CIE 1924 Photopic Standard Observer"),
                               scotopic_lef=color.spectrum.lefs.SCOTOPIC_LEFS.get(
                                   "CIE 1951 Scotopic Standard Observer")):
    """
    Converts given spectral power distribution to *CIE XYZ* colorspace using given color
    matching functions and illuminant.

    Reference: http://en.wikipedia.org/wiki/Mesopic#Mesopic_weighting_function

    Usage::

        >>> mesopic_weighting_function(500, 0.2)
        0.70522

    :param wavelength: Wavelength to calculate the mesopic function.
    :type wavelength: int or float
    :param Lp: Photopic luminance.
    :type Lp: float
    :param source: Light source color temperature.
    :type source: unicode ("Blue Heavy", "Red Heavy")
    :param method: Method to calculate the weighting factor.
    :type method: unicode ("MOVE", "LRC")
    :param photopic_lef: *V* photopic luminous efficiency function.
    :type photopic_lef: SpectralPowerDistribution
    :param scotopic_lef: *V'* scotopic luminous efficiency function.
    :type scotopic_lef: SpectralPowerDistribution
    :return: *CIE XYZ* matrix.
    :rtype: matrix (3x1)
    """

    for function in (photopic_lef, scotopic_lef):
        if function.get(wavelength) is None:
            raise color.utilities.exceptions.ProgrammingError(
                "'{0} nm' wavelength not available in '{1}' luminous efficiency function with '{2}' shape!".format(
                    wavelength,
                    function.name,
                    function.shape))

    mesopic_x_luminance_values = sorted(MESOPIC_X_DATA.keys())
    index = mesopic_x_luminance_values.index(color.algebra.common.get_closest(mesopic_x_luminance_values, Lp))
    x = MESOPIC_X_DATA.get(mesopic_x_luminance_values[index]).get(source).get(method)

    Vm = (1. - x) * scotopic_lef.get(wavelength) + x * photopic_lef.get(wavelength)

    return Vm