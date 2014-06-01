#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**transformations.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package color spectral *transformations* objects.

**Others:**

"""

from __future__ import unicode_literals

import bisect
import numpy

import color.algebra.matrix
import color.exceptions
import color.spectral.spd
import color.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "wavelength_to_XYZ",
           "spectral_to_XYZ"]

LOGGER = color.verbose.install_logger()


def wavelength_to_XYZ(wavelength, cmfs):
    """
    Converts given wavelength to *CIE XYZ* colorspace using given color matching functions, if the retrieved
    wavelength is not available in the color matching function, its value will be calculated using linear interpolation
    between the two closest wavelengths.

    Usage::

        >>> wavelength_to_XYZ(480, color.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get("Standard CIE 1931 2 Degree Observer"))
        matrix([[ 0.09564],
                [ 0.13902],
                [ 0.81295]])

    :param wavelength: Wavelength in nm.
    :type wavelength: float
    :param cmfs: Standard observer color matching functions.
    :type cmfs: dict
    :return: *CIE XYZ* matrix.
    :rtype: matrix
    """

    start, end, steps = cmfs.shape
    if wavelength < start or wavelength > end:
        raise color.exceptions.ProgrammingError(
            "'{0}' nm wavelength not in '{1} - {2}' nm supported wavelengths range!".format(wavelength, start, end))

    wavelengths = numpy.arange(start, end, steps)
    index = bisect.bisect(wavelengths, wavelength)
    if index < len(wavelengths):
        left = wavelengths[index - 1]
        right = wavelengths[index]
    else:
        left = right = wavelengths[-1]

    left_XYZ = numpy.matrix(cmfs.get(left)).reshape((3, 1))
    right_XYZ = numpy.matrix(cmfs.get(right)).reshape((3, 1))

    return color.algebra.matrix.linear_interpolate_matrices(left, right, left_XYZ, right_XYZ, wavelength)


def spectral_to_XYZ(spd,
                    cmfs,
                    illuminant=None):
    """
    Converts given relative spectral power distribution to *CIE XYZ* colorspace using given color
    matching functions and illuminant.

    Reference: **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Page 158.

    Usage::

        >>> cmfs = color.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get("Standard CIE 1931 2 Degree Observer")
        >>> spd = color.SpectralPowerDistribution("Custom", {380: 0.0600, 390: 0.0600}).zeros(*cmfs.shape)
        >>> illuminant = color.ILLUMINANTS_RELATIVE_SPD.get("D50").zeros(*cmfs.shape)
        >>> spectral_to_XYZ(spd, cmfs, illuminant)
        matrix([[  4.57648522e-04]
                [  1.29648668e-05]
                [  2.16158075e-03]])

    :param spd: Spectral power distribution.
    :type spd: SpectralPowerDistribution
    :param cmfs: Standard observer color matching functions.
    :type cmfs: XYZ_ColorMatchingFunctions
    :param illuminant: *Illuminant* spectral power distribution.
    :type illuminant: SpectralPowerDistribution
    :return: *CIE XYZ* matrix.
    :rtype: matrix

    :note: Spectral power distribution, standard observer color matching functions and illuminant shapes must be aligned.
    """

    if spd.shape != cmfs.shape:
        raise color.exceptions.ProgrammingError(
            "Spectral power distribution and standard observer color matching functions shapes are not aligned: '{0}', '{1}'.".format(
                spd.shape, cmfs.shape))

    if illuminant is None:
        start, end, steps = cmfs.shape
        range = numpy.arange(start, end + steps, steps)
        illuminant = color.spectral.spd.SpectralPowerDistribution(name="1.0",
                                                                  spd=dict(zip(*(list(range),
                                                                                 [1.] * len(range)))))
    else:
        if illuminant.shape != cmfs.shape:
            raise color.exceptions.ProgrammingError(
                "Illuminant and standard observer color matching functions shapes are not aligned: '{0}', '{1}'.".format(
                    illuminant.shape, cmfs.shape))

    illuminant = illuminant.values
    spd = spd.values

    x_bar, y_bar, z_bar = cmfs.x_bar.values, cmfs.y_bar.values, cmfs.z_bar.values

    x_products = spd * x_bar * illuminant
    y_products = spd * y_bar * illuminant
    z_products = spd * z_bar * illuminant

    normalising_factor = 100. / (y_bar * illuminant).sum()

    XYZ = numpy.matrix([normalising_factor * x_products.sum(),
                        normalising_factor * y_products.sum(),
                        normalising_factor * z_products.sum()])

    return XYZ.reshape((3, 1))