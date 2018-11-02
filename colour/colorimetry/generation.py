# -*- coding: utf-8 -*-
"""
Spectral Generation
===================

Defines various objects performing spectral generation:

-   :func:`colour.constant_spd`
-   :func:`colour.zeros_spd`
-   :func:`colour.ones_spd`
-   :func:`colour.colorimetry.gaussian_spd_normal`
-   :func:`colour.colorimetry.gaussian_spd_fwhm`
-   :func:`colour.colorimetry.single_led_spd_Ohno2005`
-   :func:`colour.colorimetry.multi_led_spd_Ohno2005`

See Also
--------
`Spectrum Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/generation.ipynb>`_

References
----------
-   :cite:`Ohno2005` : ﻿Ohno, Y. (2005). Spectral design considerations for
    white LED color rendering. Optical Engineering, 44(11), 111302.
    doi:10.1117/1.2130694
-   :cite:`Ohno2008a` : Ohno, Y., & Davis, W. (2008). NIST CQS simulation 7.4.
Retrieved from https://drive.google.com/file/d/\
1PsuU6QjUJjCX6tQyCud6ul2Tbs8rYWW9/view?usp=sharing
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.colorimetry import (DEFAULT_SPECTRAL_SHAPE,
                                SpectralPowerDistribution)
from colour.utilities import CaseInsensitiveMapping, as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'constant_spd', 'zeros_spd', 'ones_spd', 'gaussian_spd_normal',
    'gaussian_spd_fwhm', 'GAUSSIAN_SPD_METHODS', 'gaussian_spd',
    'single_led_spd_Ohno2005', 'SINGLE_LED_SPD_METHODS', 'single_led_spd',
    'multi_led_spd_Ohno2005', 'MULTI_LED_SPD_METHODS', 'multi_led_spd'
]


def constant_spd(k, shape=DEFAULT_SPECTRAL_SHAPE, dtype=DEFAULT_FLOAT_DTYPE):
    """
    Returns a spectral power distribution of given spectral shape filled with
    constant :math:`k` values.

    Parameters
    ----------
    k : numeric
        Constant :math:`k` to fill the spectral power distribution with.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.
    dtype : type
        Data type used for the spectral power distribution.

    Returns
    -------
    SpectralPowerDistribution
        Constant :math:`k` to filled spectral power distribution.

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`colour.DEFAULT_SPECTRAL_SHAPE` attribute.

    Examples
    --------
    >>> spd = constant_spd(100)
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[400]
    100.0
    """

    wavelengths = shape.range(dtype)
    values = np.full(len(wavelengths), k, dtype)

    name = '{0} Constant'.format(k)
    return SpectralPowerDistribution(
        values, wavelengths, name=name, dtype=dtype)


def zeros_spd(shape=DEFAULT_SPECTRAL_SHAPE):
    """
    Returns a spectral power distribution of given spectral shape filled with
    zeros.

    Parameters
    ----------
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.

    Returns
    -------
    SpectralPowerDistribution
        Zeros filled spectral power distribution.

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`colour.DEFAULT_SPECTRAL_SHAPE` attribute.

    Examples
    --------
    >>> spd = zeros_spd()
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[400]
    0.0
    """

    return constant_spd(0, shape)


def ones_spd(shape=DEFAULT_SPECTRAL_SHAPE):
    """
    Returns a spectral power distribution of given spectral shape filled with
    ones.

    Parameters
    ----------
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.

    Returns
    -------
    SpectralPowerDistribution
        Ones filled spectral power distribution.

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`colour.DEFAULT_SPECTRAL_SHAPE` attribute.

    Examples
    --------
    >>> spd = ones_spd()
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[400]
    1.0
    """

    return constant_spd(1, shape)


def gaussian_spd_normal(mu, sigma, shape=DEFAULT_SPECTRAL_SHAPE):
    """
    Returns a gaussian spectral power distribution of given spectral shape at
    given mean wavelength :math:`\\mu` and standard deviation :math:`sigma`.

    Parameters
    ----------
    mu : numeric
        Mean wavelength :math:`\\mu` the gaussian spectral power distribution
        will peak at.
    sigma : numeric
        Standard deviation :math:`sigma` of the gaussian spectral power
        distribution.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.

    Returns
    -------
    SpectralPowerDistribution
        Gaussian spectral power distribution.

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`colour.DEFAULT_SPECTRAL_SHAPE` attribute.

    Examples
    --------
    >>> spd = gaussian_spd_normal(555, 25)
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[555]  # doctest: +ELLIPSIS
    1.0000000...
    >>> spd[530]  # doctest: +ELLIPSIS
    0.6065306...
    """

    wavelengths = shape.range()

    values = np.exp(-(wavelengths - mu) ** 2 / (2 * sigma ** 2.))

    name = '{0}nm - {1} Sigma - Gaussian'.format(mu, sigma)

    return SpectralPowerDistribution(values, wavelengths, name=name)


def gaussian_spd_fwhm(peak_wavelength, fwhm, shape=DEFAULT_SPECTRAL_SHAPE):
    """
    Returns a gaussian spectral power distribution of given spectral shape at
    given peak wavelength and full width at half maximum.

    Parameters
    ----------
    peak_wavelength : numeric
        Wavelength the gaussian spectral power distribution will peak at.
    fwhm : numeric
        Full width at half maximum, i.e., width of the gaussian spectral power
        distribution measured between those points on the *y* axis which are
        half the maximum amplitude.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.

    Returns
    -------
    SpectralPowerDistribution
        Gaussian spectral power distribution.

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`colour.DEFAULT_SPECTRAL_SHAPE` attribute.

    Examples
    --------
    >>> spd = gaussian_spd_fwhm(555, 25)
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[555]
    1.0
    >>> spd[530]  # doctest: +ELLIPSIS
    0.3678794...
    """

    wavelengths = shape.range()

    values = np.exp(-((wavelengths - peak_wavelength) / fwhm) ** 2)

    name = '{0}nm - {1} FWHM - Gaussian'.format(peak_wavelength, fwhm)

    return SpectralPowerDistribution(values, wavelengths, name=name)


GAUSSIAN_SPD_METHODS = CaseInsensitiveMapping({
    'Normal': gaussian_spd_normal,
    'FWHM': gaussian_spd_fwhm
})
GAUSSIAN_SPD_METHODS.__doc__ = """
Supported gaussian spectral power distribution computation methods.

GAUSSIAN_SPD_METHODS : CaseInsensitiveMapping
    **{'Normal', 'FWHM'}**
"""


def gaussian_spd(mu_peak_wavelength,
                 sigma_fwhm,
                 shape=DEFAULT_SPECTRAL_SHAPE,
                 method='Normal'):
    """
    Returns a gaussian spectral power distribution of given spectral shape
    using given method.

    Parameters
    ----------
    mu_peak_wavelength : numeric
        Mean wavelength :math:`\\mu` the gaussian spectral power distribution
        will peak at.
    sigma_fwhm : numeric
        Standard deviation :math:`sigma` of the gaussian spectral power
        distribution  or Full width at half maximum, i.e., width of the
        gaussian spectral power distribution measured between those points on
        the *y* axis which are half the maximum amplitude.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.
    method : unicode, optional
        **{'Normal', 'FWHM'}**,
        Computation method.

    Returns
    -------
    SpectralPowerDistribution
        Gaussian spectral power distribution.

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`colour.DEFAULT_SPECTRAL_SHAPE` attribute.

    Examples
    --------
    >>> spd = gaussian_spd(555, 25)
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[555]  # doctest: +ELLIPSIS
    1.0000000...
    >>> spd[530]  # doctest: +ELLIPSIS
    0.6065306...
    >>> spd = gaussian_spd(555, 25, method='FWHM')
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[555]
    1.0
    >>> spd[530]  # doctest: +ELLIPSIS
    0.3678794...
    """

    return GAUSSIAN_SPD_METHODS[method](mu_peak_wavelength, sigma_fwhm, shape)


def single_led_spd_Ohno2005(peak_wavelength,
                            fwhm,
                            shape=DEFAULT_SPECTRAL_SHAPE):
    """
    Returns a single *LED* spectral power distribution of given spectral shape
    at given peak wavelength and full width at half maximum according to
    *Ohno (2005)* method.

    Parameters
    ----------
    peak_wavelength : numeric
        Wavelength the single *LED* spectral power distribution will peak at.
    fwhm : numeric
        Full width at half maximum, i.e., width of the underlying gaussian
        spectral power distribution measured between those points on the *y*
        axis which are half the maximum amplitude.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.

    Returns
    -------
    SpectralPowerDistribution
        Single *LED* spectral power distribution.

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`colour.DEFAULT_SPECTRAL_SHAPE` attribute.

    References
    ----------
    :cite:`Ohno2005`, :cite:`Ohno2008a`

    Examples
    --------
    >>> spd = single_led_spd_Ohno2005(555, 25)
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[555]  # doctest: +ELLIPSIS
    1.0000000...
    """

    spd = gaussian_spd_fwhm(peak_wavelength, fwhm, shape)

    spd.values = (spd.values + 2 * spd.values ** 5) / 3

    spd.name = '{0}nm - {1} FWHM LED - Ohno (2005)'.format(
        peak_wavelength, fwhm)

    return spd


SINGLE_LED_SPD_METHODS = CaseInsensitiveMapping({
    'Ohno 2005': single_led_spd_Ohno2005,
})
SINGLE_LED_SPD_METHODS.__doc__ = """
Supported single *LED* spectral power distribution computation methods.

SINGLE_LED_SPD_METHODS : CaseInsensitiveMapping
    **{'Ohno 2005'}**
"""


def single_led_spd(peak_wavelength,
                   fwhm,
                   shape=DEFAULT_SPECTRAL_SHAPE,
                   method='Ohno 2005'):
    """
    Returns a single *LED* spectral power distribution of given spectral shape
    at given peak wavelength and full width at half maximum according to
    given method.

    Parameters
    ----------
    peak_wavelength : numeric
        Wavelength the single *LED* spectral power distribution will peak at.
    fwhm : numeric
        Full width at half maximum, i.e., width of the underlying gaussian
        spectral power distribution measured between those points on the *y*
        axis which are half the maximum amplitude.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.
    method : unicode, optional
        **{'Ohno 2005'}**,
        Computation method.

    Returns
    -------
    SpectralPowerDistribution
        Single *LED* spectral power distribution.

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`colour.DEFAULT_SPECTRAL_SHAPE` attribute.

    References
    ----------
    :cite:`Ohno2005`, :cite:`Ohno2008a`

    Examples
    --------
    >>> spd = single_led_spd(555, 25)
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[555]  # doctest: +ELLIPSIS
    1.0000000...
    """

    return SINGLE_LED_SPD_METHODS[method](peak_wavelength, fwhm, shape)


def multi_led_spd_Ohno2005(peak_wavelengths,
                           fwhm,
                           peak_power_ratios=None,
                           shape=DEFAULT_SPECTRAL_SHAPE):
    """
    Returns a multi *LED* spectral power distribution of given spectral shape
    at given peak wavelengths and full widths at half maximum according to
    *Ohno (2005)* method.

    The multi *LED* spectral power distribution is generated using many single
    *LED* spectral power distributions generated with
    :func:`colour.single_led_spd_Ohno2005` definition.

    Parameters
    ----------
    peak_wavelengths : array_like
        Wavelengths the multi *LED* spectral power distribution will peak at,
        i.e., the peaks for each generated single *LED* spectral power
        distributions.
    fwhm : array_like
        Full widths at half maximum, i.e., widths of the underlying gaussian
        spectral power distributions measured between those points on the *y*
        axis which are half the maximum amplitude.
    peak_power_ratios : array_like, optional
        Peak power ratios for each generated single *LED* spectral power
        distributions.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.

    Returns
    -------
    SpectralPowerDistribution
        Multi *LED* spectral power distribution.

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`colour.DEFAULT_SPECTRAL_SHAPE` attribute.

    References
    ----------
    :cite:`Ohno2005`, :cite:`Ohno2008a`

    Examples
    --------
    >>> spd = multi_led_spd_Ohno2005(
    ...     np.array([457, 530, 615]),
    ...     np.array([20, 30, 20]),
    ...     np.array([0.731, 1.000, 1.660]),
    ... )
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[500]  # doctest: +ELLIPSIS
    0.1295132...
    """

    peak_wavelengths = as_float_array(peak_wavelengths)
    fwhm = np.resize(fwhm, peak_wavelengths.shape)
    if peak_power_ratios is None:
        peak_power_ratios = np.ones(peak_wavelengths.shape)
    else:
        peak_power_ratios = np.resize(peak_power_ratios,
                                      peak_wavelengths.shape)

    spd = zeros_spd(shape)

    for (peak_wavelength, fwhm_s, peak_power_ratio) in zip(
            peak_wavelengths, fwhm, peak_power_ratios):
        spd += single_led_spd_Ohno2005(peak_wavelength,
                                       fwhm_s) * peak_power_ratio

    def _format_array(a):
        """
        Formats given array :math:`a`.

        Parameters
        ----------
        a : array_like
            Array to format

        Returns
        -------
        unicode
            Formatted array :math:`a`.
        """

        return ', '.join([str(e) for e in a])

    spd.name = (
        '{0}nm - {1}FWHM - {2} Peak Power Ratios - LED - Ohno (2005)'.format(
            _format_array(peak_wavelengths),
            _format_array(fwhm),
            _format_array(peak_power_ratios),
        ))

    return spd


MULTI_LED_SPD_METHODS = CaseInsensitiveMapping({
    'Ohno 2005': multi_led_spd_Ohno2005,
})
MULTI_LED_SPD_METHODS.__doc__ = """
Supported multi *LED* spectral power distribution computation methods.

MULTI_LED_SPD_METHODS : CaseInsensitiveMapping
    **{'Ohno 2005'}**
"""


def multi_led_spd(peak_wavelengths,
                  fwhm,
                  peak_power_ratios=None,
                  shape=DEFAULT_SPECTRAL_SHAPE,
                  method='Ohno 2005'):
    """
    Returns a multi *LED* spectral power distribution of given spectral shape
    at given peak wavelengths and full widths at half maximum according to
    given method.

    Parameters
    ----------
    peak_wavelengths : array_like
        Wavelengths the multi *LED* spectral power distribution will peak at,
        i.e., the peaks for each generated single *LED* spectral power
        distributions.
    fwhm : array_like
        Full widths at half maximum, i.e., widths of the underlying gaussian
        spectral power distributions measured between those points on the *y*
        axis which are half the maximum amplitude.
    peak_power_ratios : array_like, optional
        Peak power ratios for each generated single *LED* spectral power
        distributions.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral power distribution.
    method : unicode, optional
        **{'Ohno 2005'}**,
        Computation method.

    Returns
    -------
    SpectralPowerDistribution
        Multi *LED* spectral power distribution.

    Notes
    -----
    -   By default, the spectral power distribution will use the shape given
        by :attr:`colour.DEFAULT_SPECTRAL_SHAPE` attribute.

    References
    ----------
    :cite:`Ohno2005`, :cite:`Ohno2008a`

    Examples
    --------
    >>> spd = multi_led_spd(
    ...     np.array([457, 530, 615]),
    ...     np.array([20, 30, 20]),
    ...     np.array([0.731, 1.000, 1.660]),
    ... )
    >>> spd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> spd[500]  # doctest: +ELLIPSIS
    0.1295132...
    """

    return MULTI_LED_SPD_METHODS[method](peak_wavelengths, fwhm,
                                         peak_power_ratios, shape)
