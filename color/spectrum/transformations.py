# !/usr/bin/env python
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

import numpy

import color.algebra.matrix
import color.spectrum.cmfs
import color.spectrum.lefs
import color.spectrum.spd
import color.utilities.exceptions
import color.utilities.decorators
import color.utilities.verbose
from color.algebra.interpolation import SpragueInterpolator

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["spectral_to_XYZ",
           "wavelength_to_XYZ",
           "RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs",
           "RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs"]

LOGGER = color.utilities.verbose.install_logger()


def spectral_to_XYZ(spd,
                    cmfs=color.spectrum.cmfs.STANDARD_OBSERVERS_CMFS.get("CIE 1931 2 Degree Standard Observer"),
                    illuminant=None):
    """
    Converts given spectral power distribution to *CIE XYZ* colorspace using given color
    matching functions and illuminant.

    Reference: **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Page 158.

    Usage::

        >>> cmfs = color.CMFS.get("CIE 1931 2 Degree Standard Observer")
        >>> spd = color.SpectralPowerDistribution("Custom", {380: 0.0600, 390: 0.0600}).zeros(*cmfs.shape)
        >>> illuminant = color.ILLUMINANTS_RELATIVE_SPDS.get("D50").zeros(*cmfs.shape)
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
    :rtype: matrix (3x1)
    """

    shape = cmfs.shape
    if spd.shape != cmfs.shape:
        LOGGER.debug(
            "> {0} | Spectral power distribution and standard observer color matching functions shapes are not aligned: '{1}', '{2}'.".format(
                __name__,
                spd.shape, cmfs.shape))
        spd = spd.clone().zeros(*shape)

    if illuminant is None:
        start, end, steps = shape
        range = numpy.arange(start, end + steps, steps)
        illuminant = color.spectrum.spd.SpectralPowerDistribution(name="1.0",
                                                                  spd=dict(zip(*(list(range),
                                                                                 [1.] * len(range)))))
    else:
        if illuminant.shape != cmfs.shape:
            LOGGER.debug(
                "> {0} | Illuminant and standard observer color matching functions shapes are not aligned: '{1}', '{2}'.".format(
                    __name__,
                    illuminant.shape,
                    shape))
            illuminant = illuminant.clone().zeros(*shape)

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


@color.utilities.decorators.memoize(None)
def wavelength_to_XYZ(wavelength,
                      cmfs=color.spectrum.cmfs.STANDARD_OBSERVERS_CMFS.get("CIE 1931 2 Degree Standard Observer")):
    """
    Converts given wavelength to *CIE XYZ* colorspace using given color matching functions, if the retrieved
    wavelength is not available in the color matching function, its value will be calculated using *CIE* recommendations:
    The method developed by *Sprague* (1880) should be used for interpolating functions having a uniformly spaced
    independent variable and a *Cubic Spline* method for non-uniformly spaced independent variable.

    Usage::

        >>> wavelength_to_XYZ(480, color.CMFS.get("CIE 1931 2 Degree Standard Observer"))
        matrix([[ 0.09564],
                [ 0.13902],
                [ 0.81295]])

    :param wavelength: Wavelength in nm.
    :type wavelength: float
    :param cmfs: Standard observer color matching functions.
    :type cmfs: XYZ_ColorMatchingFunctions
    :return: *CIE XYZ* matrix.
    :rtype: matrix (3x1)
    :note: If *Scipy* is not unavailable the *Cubic Spline* method will fallback to legacy *Linear* interpolation.
    """

    start, end, steps = cmfs.shape
    if wavelength < start or wavelength > end:
        raise color.utilities.exceptions.ProgrammingError(
            "'{0} nm' wavelength not in '{1} - {2}' nm supported wavelengths range!".format(wavelength, start, end))

    wavelengths, values, = cmfs.wavelengths, cmfs.values

    if wavelength not in cmfs:
        if cmfs.is_uniform():
            interpolators = [SpragueInterpolator(wavelengths, values[:, i]) for i in range(values.shape[-1])]
        else:
            try:
                from scipy.interpolate import interp1d

                interpolators = [interp1d(wavelengths, values[:, i], kind="cubic") for i in range(values.shape[-1])]
            except ImportError as error:
                LOGGER.warning(
                    "!> {0} | 'scipy.interpolate.interp1d' interpolator is unavailable, using 'numpy.interp' interpolator!".format(
                        __name__))

                x_interpolator = lambda x: numpy.interp(x, wavelengths, values[:, 0])
                y_interpolator = lambda x: numpy.interp(x, wavelengths, values[:, 1])
                z_interpolator = lambda x: numpy.interp(x, wavelengths, values[:, 2])
                interpolators = (x_interpolator, y_interpolator, z_interpolator)

        return numpy.matrix([interpolator(wavelength) for interpolator in interpolators]).reshape((3, 1))
    else:
        return numpy.matrix(cmfs.get(wavelength)).reshape((3, 1))


def RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wavelength):
    """
    Converts given *Wright & Guild 1931 2 Degree RGB CMFs* color matching functions into the
    *CIE 1931 2 Degree Standard Observer* color matching functions.

    Reference: Wyszecki & Stiles, Color Science - Concepts and Methods Data and Formulae - Second Edition, Pages 138, 139.

    Usage::

        >>> RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700)
        [[ 0.01135774]
         [ 0.004102  ]
         [ 0.        ]]

    :param wavelength: Wavelength in nm.
    :type wavelength: float
    :return: *CIE 1931 2 Degree Standard Observer* spectral tristimulus values.
    :rtype: matrix (3x1)
    :note: Data for the *CIE 1931 2 Degree Standard Observer* already exists, this definition is intended for educational purpose.
    """

    cmfs = color.spectrum.cmfs.CIE_RGB_CMFS.get("Wright & Guild 1931 2 Degree RGB CMFs")
    r_bar, y_bar, z_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, y_bar, z_bar):
        raise color.utilities.exceptions.ProgrammingError(
            "'{0} nm' wavelength not available in '{1}' color matching functions with '{2}' shape!".format(wavelength,
                                                                                                          cmfs.name,
                                                                                                          cmfs.shape))

    r = r_bar / (r_bar + y_bar + z_bar)
    g = y_bar / (r_bar + y_bar + z_bar)
    b = z_bar / (r_bar + y_bar + z_bar)

    x = (0.49000 * r + 0.31000 * g + 0.20000 * b) / (0.66697 * r + 1.13240 * g + 1.20063 * b)
    y = (0.17697 * r + 0.81240 * g + 0.01063 * b) / (0.66697 * r + 1.13240 * g + 1.20063 * b)
    z = (0.00000 * r + 0.01000 * g + 0.99000 * b) / (0.66697 * r + 1.13240 * g + 1.20063 * b)

    V = color.spectrum.lefs.PHOTOPIC_LEFS.get("CIE 1924 Photopic Standard Observer").clone()
    V.align(*cmfs.shape)
    L = V.get(wavelength)

    x_bar = x / y * L
    y_bar = L
    z_bar = z / y * L

    return numpy.array([x_bar, y_bar, z_bar])


def RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wavelength):
    """
    Converts given *Stiles & Burch 1959 10 Degree RGB CMFs* color matching
    functions into the *CIE 1964 10 Degree Standard Observer* color matching functions.

    Reference: Wyszecki & Stiles, Color Science - Concepts and Methods Data and Formulae - Second Edition, Page 141.

    Usage::

        >>> RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700)
        [[ 0.01135774]
         [ 0.004102  ]
         [ 0.        ]]

    :param wavelength: Wavelength in nm.
    :type wavelength: float
    :return: *CIE 1964 10 Degree Standard Observer* spectral tristimulus values.
    :rtype: matrix (3x1)
    :note: Data for the *CIE 1964 10 Degree Standard Observer* already exists, this definition is intended for educational purpose.
    """

    cmfs = color.spectrum.cmfs.CIE_RGB_CMFS.get("Stiles & Burch 1959 10 Degree RGB CMFs")
    r_bar, y_bar, z_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, y_bar, z_bar):
        raise color.utilities.exceptions.ProgrammingError(
            "'{0} nm' wavelength not available in '{1}' color matching functions with '{2}' shape!".format(wavelength,
                                                                                                          cmfs.name,
                                                                                                          cmfs.shape))

    x_bar = 0.341080 * r_bar + 0.189145 * y_bar + 0.387529 * z_bar
    y_bar = 0.139058 * r_bar + 0.837460 * y_bar + 0.073316 * z_bar
    z_bar = 0.000000 * r_bar + 0.039553 * y_bar + 2.026200 * z_bar

    return numpy.array([x_bar, y_bar, z_bar])