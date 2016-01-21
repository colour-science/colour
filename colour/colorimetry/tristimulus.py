#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tristimulus Values
==================

Defines objects for tristimulus values computation from spectral data.

See Also
--------
`Colour Matching Functions IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/colorimetry/cmfs.ipynb>`_
`Spectrum IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/colorimetry/spectrum.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import (
    CubicSplineInterpolator,
    LinearInterpolator,
    PchipInterpolator,
    SpragueInterpolator,
    lagrange_coefficients)
from colour.colorimetry import STANDARD_OBSERVERS_CMFS, ones_spd
from colour.utilities import CaseInsensitiveMapping, is_string

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['lagrange_coefficients_ASTME202211',
           'spectral_to_XYZ',
           'wavelength_to_XYZ']

_TRISTIMULUS_WEIGHTING_FACTORS_CACHE = None

_LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE = None


def lagrange_coefficients_ASTME202211(
        interval=10,
        interval_type='inner'):
    """
    Computes the *Lagrange Coefficients* accordingly to
    *ASTM Designation: E2022 – 11* method for given interval size.

    Parameters
    ----------
    interval : int
        Interval size in nm.
    interval_type : unicode, optional
        **{'inner', 'boundary'}**,
        If the interval is an *inner* interval *Lagrange Coefficients* are
        computed for degree 4. Degree 3 is used for a *boundary* interval.

    Returns
    -------
    ndarray
        *Lagrange Coefficients*.

    See Also
    --------
    colour.lagrange_coefficients

    Examples
    --------
    >>> lagrange_coefficients_ASTME202211(  # doctest: +ELLIPSIS
    ...     10, 'inner')
    array([[-0.028...,  0.940...,  0.104..., -0.016...],
           [-0.048...,  0.864...,  0.216..., -0.032...],
           [-0.059...,  0.773...,  0.331..., -0.045...],
           [-0.064...,  0.672...,  0.448..., -0.056...],
           [-0.062...,  0.562...,  0.562..., -0.062...],
           [-0.056...,  0.448...,  0.672..., -0.064...],
           [-0.045...,  0.331...,  0.773..., -0.059...],
           [-0.032...,  0.216...,  0.864..., -0.048...],
           [-0.016...,  0.104...,  0.940..., -0.028...]])
    >>> lagrange_coefficients_ASTME202211(  # doctest: +ELLIPSIS
    ...     10, 'boundary')
    array([[ 0.85...,  0.19..., -0.04...],
           [ 0.72...,  0.36..., -0.08...],
           [ 0.59...,  0.51..., -0.10...],
           [ 0.48...,  0.64..., -0.12...],
           [ 0.37...,  0.75..., -0.12...],
           [ 0.28...,  0.84..., -0.12...],
           [ 0.19...,  0.91..., -0.10...],
           [ 0.12...,  0.96..., -0.08...],
           [ 0.05...,  0.99..., -0.04...]])
    """

    global _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE
    if _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE is None:
        _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE = CaseInsensitiveMapping()

    name_lica = ', '.join((str(interval), interval_type))
    if name_lica in _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE:
        return _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE[name_lica]

    r_n = np.linspace(1 / interval, 1 - (1 / interval), interval - 1)
    d = 3
    if interval_type.lower() == 'inner':
        r_n += 1
        d = 4

    lica = _LAGRANGE_INTERPOLATING_COEFFICIENTS_CACHE[name_lica] = (
        np.asarray([lagrange_coefficients(r, d) for r in r_n]))

    return lica


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
    >>> from colour import (
    ...     CMFS, ILLUMINANTS_RELATIVE_SPDS, SpectralPowerDistribution)
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

    spd = spd.values
    x_bar, y_bar, z_bar = (cmfs.x_bar.values,
                           cmfs.y_bar.values,
                           cmfs.z_bar.values)
    illuminant = illuminant.values

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
                          'CIE 1931 2 Degree Standard Observer'),
                      method=None):
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
    method : unicode, optional
        {None, 'Cubic Spline', 'Linear', 'Pchip', 'Sprague'},
        Enforce given interpolation method.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Raises
    ------
    RuntimeError
        If Sprague (1880) interpolation method is forced with a
        non-uniformly spaced independent variable.
    ValueError
        If the interpolation method is not defined or if wavelength
        :math:`\lambda` is not contained in the colour matching functions
        domain.

    Notes
    -----
    -   Output *CIE XYZ* tristimulus values are in domain [0, 1].
    -   If *scipy* is not unavailable the *Cubic Spline* method will fallback
        to legacy *Linear* interpolation.
    -   Sprague (1880) interpolator cannot be used for interpolating
        functions having a non-uniformly spaced independent variable.

    Warning
    -------
    -   If *scipy* is not unavailable the *Cubic Spline* method will fallback
        to legacy *Linear* interpolation.
    -   *Cubic Spline* interpolator requires at least 3 wavelengths
        :math:`\lambda_n` for interpolation.
    -   *Linear* interpolator requires at least 2 wavelengths :math:`\lambda_n`
        for interpolation.
    -   *Pchip* interpolator requires at least 2 wavelengths :math:`\lambda_n`
        for interpolation.
    -   Sprague (1880) interpolator requires at least 6 wavelengths
        :math:`\lambda_n` for interpolation.

    Examples
    --------
    Uniform data is using Sprague (1880) interpolation by default:

    >>> from colour import CMFS
    >>> cmfs = CMFS.get('CIE 1931 2 Degree Standard Observer')
    >>> wavelength_to_XYZ(480, cmfs)  # doctest: +ELLIPSIS
    array([ 0.09564  ,  0.13902  ,  0.812950...])
    >>> wavelength_to_XYZ(480.5, cmfs)  # doctest: +ELLIPSIS
    array([ 0.0914287...,  0.1418350...,  0.7915726...])

    Enforcing *Cubic Spline* interpolation:

    >>> wavelength_to_XYZ(480.5, cmfs, 'Cubic Spline')  # doctest: +ELLIPSIS
    array([ 0.0914288...,  0.1418351...,  0.7915729...])

    Enforcing *Linear* interpolation:

    >>> wavelength_to_XYZ(480.5, cmfs, 'Linear')  # doctest: +ELLIPSIS
    array([ 0.0914697...,  0.1418482...,  0.7917337...])

    Enforcing *Pchip* interpolation:

    >>> wavelength_to_XYZ(480.5, cmfs, 'Pchip')  # doctest: +ELLIPSIS
    array([ 0.0914280...,  0.1418341...,  0.7915711...])
    """

    cmfs_shape = cmfs.shape
    if (np.min(wavelength) < cmfs_shape.start or
            np.max(wavelength) > cmfs_shape.end):
        raise ValueError(
            '"{0} nm" wavelength is not in "[{1}, {2}]" domain!'.format(
                wavelength, cmfs_shape.start, cmfs_shape.end))

    if wavelength not in cmfs:
        wavelengths, values, = cmfs.wavelengths, cmfs.values

        if is_string(method):
            method = method.lower()

        is_uniform = cmfs.is_uniform()

        if method is None:
            if is_uniform:
                interpolator = SpragueInterpolator
            else:
                interpolator = CubicSplineInterpolator
        elif method == 'cubic spline':
            interpolator = CubicSplineInterpolator
        elif method == 'linear':
            interpolator = LinearInterpolator
        elif method == 'pchip':
            interpolator = PchipInterpolator
        elif method == 'sprague':
            if is_uniform:
                interpolator = SpragueInterpolator
            else:
                raise RuntimeError(
                    ('"Sprague" interpolator can only be used for '
                     'interpolating functions having a uniformly spaced '
                     'independent variable!'))
        else:
            raise ValueError(
                'Undefined "{0}" interpolator!'.format(method))

        interpolators = [interpolator(wavelengths, values[..., i])
                         for i in range(values.shape[-1])]

        XYZ = np.dstack([i(np.ravel(wavelength)) for i in interpolators])
    else:
        XYZ = cmfs.get(wavelength)

    XYZ = np.reshape(XYZ, np.asarray(wavelength).shape + (3,))

    return XYZ
