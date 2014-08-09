# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tristimulus Values
==================

Defines objects for tristimulus values computation from spectral data.
"""

from __future__ import unicode_literals

import numpy as np

from colour.algebra import SplineInterpolator, SpragueInterpolator

from colour.colorimetry import (
    SpectralPowerDistribution,
    STANDARD_OBSERVERS_CMFS)
from colour.utilities import memoize

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["spectral_to_XYZ",
           "wavelength_to_XYZ"]

_WAVELENGTH_TO_XYZ_CACHE = {}


def spectral_to_XYZ(spd,
                    cmfs=STANDARD_OBSERVERS_CMFS.get(
                        "CIE 1931 2 Degree Standard Observer"),
                    illuminant=None):
    """
    Converts given spectral power distribution to *CIE XYZ* colourspace using
    given colour matching functions and illuminant.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralPowerDistribution
        *Illuminant* spectral power distribution.

    Returns
    -------
    ndarray, (3, 1)
        *CIE XYZ* colourspace matrix.

    Notes
    -----
    -   Output *CIE XYZ* colourspace matrix is in domain [0, 1].

    References
    ----------
    .. [1]  **Wyszecki & Stiles**,
            *Color Science - Concepts and Methods Data and Formulae -
            Second Edition*,
            Wiley Classics Library Edition, published 2000,
            ISBN-10: 0-471-39918-3,
            Page 158.

    Examples
    --------
    >>> cmfs = colour.CMFS.get("CIE 1931 2 Degree Standard Observer")
    >>> data = {380: 0.0600, 390: 0.0600}
    >>> spd = colour.SpectralPowerDistribution("Custom", data)
    >>> illuminant = colour.ILLUMINANTS_RELATIVE_SPDS.get("D50")
    >>> colour.spectral_to_XYZ(spd, cmfs, illuminant)
    array([[  4.57648522e-04],
           [  1.29648668e-05],
           [  2.16158075e-03]])
    """

    shape = cmfs.shape
    if spd.shape != cmfs.shape:
        spd = spd.clone().zeros(*shape)

    if illuminant is None:
        start, end, steps = shape
        range = np.arange(start, end + steps, steps)
        illuminant = SpectralPowerDistribution(
            name="1.0",
            data=dict(zip(*(tuple(range), [1.] * len(range)))))
    else:
        if illuminant.shape != cmfs.shape:
            illuminant = illuminant.clone().zeros(*shape)

    illuminant = illuminant.values
    spd = spd.values

    x_bar, y_bar, z_bar = (cmfs.x_bar.values,
                           cmfs.y_bar.values,
                           cmfs.z_bar.values)

    x_products = spd * x_bar * illuminant
    y_products = spd * y_bar * illuminant
    z_products = spd * z_bar * illuminant

    normalising_factor = 100. / np.sum(y_bar * illuminant)

    XYZ = np.array([normalising_factor * np.sum(x_products),
                    normalising_factor * np.sum(y_products),
                    normalising_factor * np.sum(z_products)])

    return XYZ.reshape((3, 1))


@memoize(_WAVELENGTH_TO_XYZ_CACHE)
def wavelength_to_XYZ(wavelength,
                      cmfs=STANDARD_OBSERVERS_CMFS.get(
                          "CIE 1931 2 Degree Standard Observer")):
    """
    Converts given wavelength :math:`\lambda` to *CIE XYZ* colourspace using
    given colour matching functions.

    If the wavelength :math:`\lambda` is not available in the colour matching
    function, its value will be calculated using *CIE* recommendations:
    The method developed by *Sprague (1880)* should be used for interpolating
    functions having a uniformly spaced independent variable and a
    *Cubic Spline* method for non-uniformly spaced independent variable.

    Parameters
    ----------
    wavelength : float
        Wavelength :math:`\lambda` in nm.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.

    Returns
    -------
    ndarray, (3, 1)
        *CIE XYZ* colourspace matrix.

    Notes
    -----
    -   Output *CIE XYZ* colourspace matrix is in domain [0, 1].
    -   If *scipy* is not unavailable the *Cubic Spline* method will
        fallback to legacy *Linear* interpolation.

    Examples
    --------
    >>> cmfs = colour.CMFS.get("CIE 1931 2 Degree Standard Observer")
    >>> colour.wavelength_to_XYZ(480)
    array([[ 0.09564  ],
           [ 0.13902  ],
           [ 0.8129501]])
    """

    start, end, steps = cmfs.shape
    if wavelength < start or wavelength > end:
        raise ValueError("'{0} nm' wavelength not in '{1} - {2}' nm supported"
                         "wavelengths range!".format(wavelength, start, end))

    if wavelength not in cmfs:
        wavelengths, values, = cmfs.wavelengths, cmfs.values
        interpolator = (SpragueInterpolator
                        if cmfs.is_uniform() else
                        SplineInterpolator)

        interpolators = [interpolator(wavelengths, values[:, i])
                         for i in range(values.shape[-1])]

        return np.array([interpolator(wavelength)
                         for interpolator in interpolators]).reshape((3, 1))
    else:
        return np.array(cmfs.get(wavelength)).reshape((3, 1))