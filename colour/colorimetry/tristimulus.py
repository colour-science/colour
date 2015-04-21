#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tristimulus Values
==================

Defines objects for tristimulus values computation from spectral data.

See Also
--------
`Colour Matching Functions IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/colorimetry/cmfs.ipynb>`_  # noqa
`Spectrum IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/colorimetry/spectrum.ipynb>`_  # noqa
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import SplineInterpolator, SpragueInterpolator
from colour.colorimetry import STANDARD_OBSERVERS_CMFS, ones_spd

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['spectral_to_XYZ',
           'wavelength_to_XYZ']


def spectral_to_XYZ(spd,
                    cmfs=STANDARD_OBSERVERS_CMFS.get(
                        'CIE 1931 2 Degree Standard Observer'),
                    illuminant=None):
    """
    Converts given spectral power distribution to *CIE XYZ* tristimulus values
    using given colour matching functions and illuminant.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralPowerDistribution, optional
        *Illuminant* spectral power distribution.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* tristimulus values.

    Warning
    -------
    The output domain of that definition is non standard!

    Notes
    -----
    -   Output *CIE XYZ* tristimulus values are in domain [0, 100].

    References
    ----------
    .. [1]  Wyszecki, G., & Stiles, W. S. (2000). Integration Replace by
            Summation. In Color Science: Concepts and Methods, Quantitative
            Data and Formulae (pp. 158–163). Wiley. ISBN:978-0471399186

    Examples
    --------
    >>> from colour import CMFS, ILLUMINANTS_RELATIVE_SPDS, SpectralPowerDistribution  # noqa
    >>> cmfs = CMFS.get('CIE 1931 2 Degree Standard Observer')
    >>> data = {380: 0.0600, 390: 0.0600}
    >>> spd = SpectralPowerDistribution('Custom', data)
    >>> illuminant = ILLUMINANTS_RELATIVE_SPDS.get('D50')
    >>> spectral_to_XYZ(spd, cmfs, illuminant)  # doctest: +ELLIPSIS
    array([  4.5764852...e-04,   1.2964866...e-05,   2.1615807...e-03])
    """

    shape = cmfs.shape
    if spd.shape != cmfs.shape:
        spd = spd.clone().zeros(shape)

    if illuminant is None:
        illuminant = ones_spd(shape)
    else:
        if illuminant.shape != cmfs.shape:
            illuminant = illuminant.clone().zeros(shape)

    illuminant = illuminant.values
    spd = spd.values

    x_bar, y_bar, z_bar = (cmfs.x_bar.values,
                           cmfs.y_bar.values,
                           cmfs.z_bar.values)

    x_products = spd * x_bar * illuminant
    y_products = spd * y_bar * illuminant
    z_products = spd * z_bar * illuminant

    normalising_factor = 100 / np.sum(y_bar * illuminant)

    XYZ = np.array([normalising_factor * np.sum(x_products),
                    normalising_factor * np.sum(y_products),
                    normalising_factor * np.sum(z_products)])

    return XYZ


def wavelength_to_XYZ(wavelength,
                      cmfs=STANDARD_OBSERVERS_CMFS.get(
                          'CIE 1931 2 Degree Standard Observer')):
    """
    Converts given wavelength :math:`\lambda` to *CIE XYZ* tristimulus values
    using given colour matching functions.

    If the wavelength :math:`\lambda` is not available in the colour matching
    function, its value will be calculated using *CIE* recommendations:
    The method developed by Sprague (1880) should be used for interpolating
    functions having a uniformly spaced independent variable and a
    *Cubic Spline* method for non-uniformly spaced independent variable.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in nm.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Raises
    ------
    ValueError
        If wavelength :math:`\lambda` is not contained in the colour matching
        functions domain.

    Notes
    -----
    -   Output *CIE XYZ* tristimulus values are in domain [0, 1].
    -   If *scipy* is not unavailable the *Cubic Spline* method will fallback
        to legacy *Linear* interpolation.

    Examples
    --------
    >>> from colour import CMFS
    >>> cmfs = CMFS.get('CIE 1931 2 Degree Standard Observer')
    >>> wavelength_to_XYZ(480)  # doctest: +ELLIPSIS
    array([ 0.09564  ,  0.13902  ,  0.812950...])
    """

    cmfs_shape = cmfs.shape
    if (np.min(wavelength) < cmfs_shape.start or
                np.max(wavelength) > cmfs_shape.end):
        raise ValueError(
            '"{0} nm" wavelength is not in "[{1}, {2}]" domain!'.format(
                wavelength, cmfs_shape.start, cmfs_shape.end))

    if wavelength not in cmfs:
        wavelengths, values, = cmfs.wavelengths, cmfs.values
        interpolator = (SpragueInterpolator
                        if cmfs.is_uniform() else
                        SplineInterpolator)

        interpolators = [interpolator(wavelengths, values[..., i])
                         for i in range(values.shape[-1])]

        XYZ = np.dstack([interpolator(np.ravel(wavelength))
                         for interpolator in interpolators])
    else:
        XYZ = cmfs.get(wavelength)

    XYZ = np.reshape(XYZ, np.asarray(wavelength).shape + (3,))

    return XYZ
