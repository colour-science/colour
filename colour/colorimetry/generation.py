# -*- coding: utf-8 -*-
"""
Spectral Generation
===================

Defines various objects performing spectral generation:

-   :func:`colour.sd_constant`
-   :func:`colour.sd_zeros`
-   :func:`colour.sd_ones`
-   :func:`colour.msds_constant`
-   :func:`colour.msds_zeros`
-   :func:`colour.msds_ones`
-   :func:`colour.colorimetry.sd_gaussian_normal`
-   :func:`colour.colorimetry.sd_gaussian_fwhm`
-   :attr:`colour.SD_GAUSSIAN_METHODS`
-   :func:`colour.sd_gaussian`
-   :func:`colour.colorimetry.sd_single_led_Ohno2005`
-   :attr:`colour.SD_SINGLE_LED_METHODS`
-   :func:`colour.sd_single_led`
-   :func:`colour.colorimetry.sd_multi_leds_Ohno2005`
-   :attr:`colour.SD_MULTI_LEDS_METHODS`
-   :func:`colour.sd_multi_leds`

References
----------
-   :cite:`Ohno2005` : Ohno, Yoshi. (2005). Spectral design considerations for
    white LED color rendering. Optical Engineering, 44(11), 111302.
    doi:10.1117/1.2130694
-   :cite:`Ohno2008a` : Ohno, Yoshiro, & Davis, W. (2008). NIST CQS simulation
    (Version 7.4) [Computer software].
    https://drive.google.com/file/d/1PsuU6QjUJjCX6tQyCud6ul2Tbs8rYWW9/view?\
usp=sharing
"""

from __future__ import division, unicode_literals

import colour.ndarray as np

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.colorimetry import (
    SPECTRAL_SHAPE_DEFAULT, MultiSpectralDistributions, SpectralDistribution)
from colour.utilities import CaseInsensitiveMapping, as_float_array, full, ones

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'sd_constant', 'sd_zeros', 'sd_ones', 'msds_constant', 'msds_zeros',
    'msds_ones', 'sd_gaussian_normal', 'sd_gaussian_fwhm',
    'SD_GAUSSIAN_METHODS', 'sd_gaussian', 'sd_single_led_Ohno2005',
    'SD_SINGLE_LED_METHODS', 'sd_single_led', 'sd_multi_leds_Ohno2005',
    'SD_MULTI_LEDS_METHODS', 'sd_multi_leds'
]


def sd_constant(k, shape=SPECTRAL_SHAPE_DEFAULT, dtype=None):
    """
    Returns a spectral distribution of given spectral shape filled with
    constant :math:`k` values.

    Parameters
    ----------
    k : numeric
        Constant :math:`k` to fill the spectral distribution with.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral distribution.
    dtype : type
        Data type used for the spectral distribution.

    Returns
    -------
    SpectralDistribution
        Constant :math:`k` filled spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape given by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> sd = sd_constant(100)
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[400]
    100.0
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    wavelengths = np.array(shape.range(dtype))
    values = full(len(wavelengths), k, dtype)

    name = '{0} Constant'.format(k)
    return SpectralDistribution(values, wavelengths, name=name, dtype=dtype)


def sd_zeros(shape=SPECTRAL_SHAPE_DEFAULT):
    """
    Returns a spectral distribution of given spectral shape filled with zeros.

    Parameters
    ----------
    shape : SpectralShape, optional
        Spectral shape used to create the spectral distribution.

    Returns
    -------
    SpectralDistribution
        Zeros filled spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape given by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> sd = sd_zeros()
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[400]
    0.0
    """

    return sd_constant(0, shape)


def sd_ones(shape=SPECTRAL_SHAPE_DEFAULT):
    """
    Returns a spectral distribution of given spectral shape filled with ones.

    Parameters
    ----------
    shape : SpectralShape, optional
        Spectral shape used to create the spectral distribution.

    Returns
    -------
    SpectralDistribution
        Ones filled spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape given by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> sd = sd_ones()
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[400]
    1.0
    """

    return sd_constant(1, shape)


def msds_constant(k, labels, shape=SPECTRAL_SHAPE_DEFAULT, dtype=None):
    """
    Returns the multi-spectral distributions with given labels and given
    spectral shape filled with constant :math:`k` values.

    Parameters
    ----------
    k : numeric
        Constant :math:`k` to fill the multi-spectral distributions with.
    labels : array_like
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.
    shape : SpectralShape, optional
        Spectral shape used to create the multi-spectral distributions.
    dtype : type
        Data type used for the multi-spectral distributions.

    Returns
    -------
    MultiSpectralDistributions
        Constant :math:`k` filled multi-spectral distributions.

    Notes
    -----
    -   By default, the multi-spectral distributions will use the shape given
        by :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> msds = msds_constant(100, labels=['a', 'b', 'c'])
    >>> msds.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> msds[400]
    array([ 100.,  100.,  100.])
    >>> msds.labels  # doctest: +SKIP
    ['a', 'b', 'c']
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    wavelengths = np.array(shape.range(dtype))
    values = full([len(wavelengths), len(labels)], k, dtype)

    name = '{0} Constant'.format(k)
    return MultiSpectralDistributions(
        values, wavelengths, name=name, labels=labels, dtype=dtype)


def msds_zeros(labels, shape=SPECTRAL_SHAPE_DEFAULT):
    """
    Returns the multi-spectral distributionss with given labels and given
    spectral shape filled with zeros.

    Parameters
    ----------
    labels : array_like
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.
    shape : SpectralShape, optional
        Spectral shape used to create the multi-spectral distributions.

    Returns
    -------
    MultiSpectralDistributions
        Zeros filled multi-spectral distributions.

    Notes
    -----
    -   By default, the multi-spectral distributions will use the shape given
        by :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> msds = msds_zeros(labels=['a', 'b', 'c'])
    >>> msds.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> msds[400]
    array([ 0.,  0.,  0.])
    >>> msds.labels  # doctest: +SKIP
    ['a', 'b', 'c']
    """

    return msds_constant(0, labels, shape)


def msds_ones(labels, shape=SPECTRAL_SHAPE_DEFAULT):
    """
    Returns the multi-spectral distributionss with given labels and given
    spectral shape filled with ones.

    Parameters
    ----------
    labels : array_like
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.
    shape : SpectralShape, optional
        Spectral shape used to create the multi-spectral distributions.

    Returns
    -------
    MultiSpectralDistributions
        Ones filled multi-spectral distributions.

    Notes
    -----
    -   By default, the multi-spectral distributions will use the shape given
        by :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> msds = msds_ones(labels=['a', 'b', 'c'])
    >>> msds.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> msds[400]
    array([ 1.,  1.,  1.])
    >>> msds.labels  # doctest: +SKIP
    ['a', 'b', 'c']
    """

    return msds_constant(1, labels, shape)


def sd_gaussian_normal(mu, sigma, shape=SPECTRAL_SHAPE_DEFAULT):
    """
    Returns a gaussian spectral distribution of given spectral shape at
    given mean wavelength :math:`\\mu` and standard deviation :math:`sigma`.

    Parameters
    ----------
    mu : numeric
        Mean wavelength :math:`\\mu` the gaussian spectral distribution will
        peak at.
    sigma : numeric
        Standard deviation :math:`sigma` of the gaussian spectral distribution.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral distribution.

    Returns
    -------
    SpectralDistribution
        Gaussian spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape given by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> sd = sd_gaussian_normal(555, 25)
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[555]  # doctest: +ELLIPSIS
    1.0000000...
    >>> sd[530]  # doctest: +ELLIPSIS
    0.6065306...
    """

    wavelengths = np.array(shape.range())

    values = np.exp(-(wavelengths - mu) ** 2 / (2 * sigma ** 2.))

    name = '{0}nm - {1} Sigma - Gaussian'.format(mu, sigma)

    return SpectralDistribution(values, wavelengths, name=name)


def sd_gaussian_fwhm(peak_wavelength, fwhm, shape=SPECTRAL_SHAPE_DEFAULT):
    """
    Returns a gaussian spectral distribution of given spectral shape at given
    peak wavelength and full width at half maximum.

    Parameters
    ----------
    peak_wavelength : numeric
        Wavelength the gaussian spectral distribution will peak at.
    fwhm : numeric
        Full width at half maximum, i.e. width of the gaussian spectral
        distribution measured between those points on the *y* axis which are
        half the maximum amplitude.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral distribution.

    Returns
    -------
    SpectralDistribution
        Gaussian spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape given by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> sd = sd_gaussian_fwhm(555, 25)
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[555]
    1.0
    >>> sd[530]  # doctest: +ELLIPSIS
    0.3678794...
    """

    wavelengths = np.array(shape.range())

    values = np.exp(-((wavelengths - peak_wavelength) / fwhm) ** 2)

    name = '{0}nm - {1} FWHM - Gaussian'.format(peak_wavelength, fwhm)

    return SpectralDistribution(values, wavelengths, name=name)


SD_GAUSSIAN_METHODS = CaseInsensitiveMapping({
    'Normal': sd_gaussian_normal,
    'FWHM': sd_gaussian_fwhm
})
SD_GAUSSIAN_METHODS.__doc__ = """
Supported gaussian spectral distribution computation methods.

SD_GAUSSIAN_METHODS : CaseInsensitiveMapping
    **{'Normal', 'FWHM'}**
"""


def sd_gaussian(mu_peak_wavelength,
                sigma_fwhm,
                shape=SPECTRAL_SHAPE_DEFAULT,
                method='Normal'):
    """
    Returns a gaussian spectral distribution of given spectral shape using
    given method.

    Parameters
    ----------
    mu_peak_wavelength : numeric
        Mean wavelength :math:`\\mu` the gaussian spectral distribution will
        peak at.
    sigma_fwhm : numeric
        Standard deviation :math:`sigma` of the gaussian spectral distribution
        or Full width at half maximum, i.e. width of the gaussian spectral
        distribution measured between those points on the *y* axis which are
        half the maximum amplitude.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral distribution.
    method : unicode, optional
        **{'Normal', 'FWHM'}**,
        Computation method.

    Returns
    -------
    SpectralDistribution
        Gaussian spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape given by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> sd = sd_gaussian(555, 25)
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[555]  # doctest: +ELLIPSIS
    1.0000000...
    >>> sd[530]  # doctest: +ELLIPSIS
    0.6065306...
    >>> sd = sd_gaussian(555, 25, method='FWHM')
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[555]
    1.0
    >>> sd[530]  # doctest: +ELLIPSIS
    0.3678794...
    """

    return SD_GAUSSIAN_METHODS[method](mu_peak_wavelength, sigma_fwhm, shape)


def sd_single_led_Ohno2005(peak_wavelength, fwhm,
                           shape=SPECTRAL_SHAPE_DEFAULT):
    """
    Returns a single *LED* spectral distribution of given spectral shape at
    given peak wavelength and full width at half maximum according to
    *Ohno (2005)* method.

    Parameters
    ----------
    peak_wavelength : numeric
        Wavelength the single *LED* spectral distribution will peak at.
    fwhm : numeric
        Full width at half maximum, i.e. width of the underlying gaussian
        spectral distribution measured between those points on the *y* axis
        which are half the maximum amplitude.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral distribution.

    Returns
    -------
    SpectralDistribution
        Single *LED* spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape given by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    References
    ----------
    :cite:`Ohno2005`, :cite:`Ohno2008a`

    Examples
    --------
    >>> sd = sd_single_led_Ohno2005(555, 25)
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[555]  # doctest: +ELLIPSIS
    1.0000000...
    """

    sd = sd_gaussian_fwhm(peak_wavelength, fwhm, shape)

    sd.values = (sd.values + 2 * sd.values ** 5) / 3

    sd.name = '{0}nm - {1} FWHM LED - Ohno (2005)'.format(
        peak_wavelength, fwhm)

    return sd


SD_SINGLE_LED_METHODS = CaseInsensitiveMapping({
    'Ohno 2005': sd_single_led_Ohno2005,
})
SD_SINGLE_LED_METHODS.__doc__ = """
Supported single *LED* spectral distribution computation methods.

SD_SINGLE_LED_METHODS : CaseInsensitiveMapping
    **{'Ohno 2005'}**
"""


def sd_single_led(peak_wavelength,
                  fwhm,
                  shape=SPECTRAL_SHAPE_DEFAULT,
                  method='Ohno 2005'):
    """
    Returns a single *LED* spectral distribution of given spectral shape at
    given peak wavelength and full width at half maximum according to given
    method.

    Parameters
    ----------
    peak_wavelength : numeric
        Wavelength the single *LED* spectral distribution will peak at.
    fwhm : numeric
        Full width at half maximum, i.e. width of the underlying gaussian
        spectral distribution measured between those points on the *y*
        axis which are half the maximum amplitude.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral distribution.
    method : unicode, optional
        **{'Ohno 2005'}**,
        Computation method.

    Returns
    -------
    SpectralDistribution
        Single *LED* spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape given by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    References
    ----------
    :cite:`Ohno2005`, :cite:`Ohno2008a`

    Examples
    --------
    >>> sd = sd_single_led(555, 25)
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[555]  # doctest: +ELLIPSIS
    1.0000000...
    """

    return SD_SINGLE_LED_METHODS[method](peak_wavelength, fwhm, shape)


def sd_multi_leds_Ohno2005(peak_wavelengths,
                           fwhm,
                           peak_power_ratios=None,
                           shape=SPECTRAL_SHAPE_DEFAULT):
    """
    Returns a multi *LED* spectral distribution of given spectral shape at
    given peak wavelengths and full widths at half maximum according to
    *Ohno (2005)* method.

    The multi *LED* spectral distribution is generated using many single
    *LED* spectral distributions generated with
    :func:`colour.sd_single_led_Ohno2005` definition.

    Parameters
    ----------
    peak_wavelengths : array_like
        Wavelengths the multi *LED* spectral distribution will peak at, i.e.
        the peaks for each generated single *LED* spectral distributions.
    fwhm : array_like
        Full widths at half maximum, i.e. widths of the underlying gaussian
        spectral distributions measured between those points on the *y* axis
        which are half the maximum amplitude.
    peak_power_ratios : array_like, optional
        Peak power ratios for each generated single *LED* spectral
        distributions.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral distribution.

    Returns
    -------
    SpectralDistribution
        Multi *LED* spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape given by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    References
    ----------
    :cite:`Ohno2005`, :cite:`Ohno2008a`

    Examples
    --------
    >>> sd = sd_multi_leds_Ohno2005(
    ...     np.array([457, 530, 615]),
    ...     np.array([20, 30, 20]),
    ...     np.array([0.731, 1.000, 1.660]),
    ... )
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[500]  # doctest: +ELLIPSIS
    0.1295132...
    """

    peak_wavelengths = as_float_array(peak_wavelengths)
    fwhm = np.resize(fwhm, peak_wavelengths.shape)
    if peak_power_ratios is None:
        peak_power_ratios = ones(peak_wavelengths.shape)
    else:
        peak_power_ratios = np.resize(peak_power_ratios,
                                      peak_wavelengths.shape)

    sd = sd_zeros(shape)

    for (peak_wavelength, fwhm_s, peak_power_ratio) in zip(
            peak_wavelengths, fwhm, peak_power_ratios):
        sd += sd_single_led_Ohno2005(peak_wavelength,
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

    sd.name = (
        '{0}nm - {1}FWHM - {2} Peak Power Ratios - LED - Ohno (2005)'.format(
            _format_array(peak_wavelengths),
            _format_array(fwhm),
            _format_array(peak_power_ratios),
        ))

    return sd


SD_MULTI_LEDS_METHODS = CaseInsensitiveMapping({
    'Ohno 2005': sd_multi_leds_Ohno2005,
})
SD_MULTI_LEDS_METHODS.__doc__ = """
Supported multi *LED* spectral distribution computation methods.

SD_MULTI_LEDS_METHODS : CaseInsensitiveMapping
    **{'Ohno 2005'}**
"""


def sd_multi_leds(peak_wavelengths,
                  fwhm,
                  peak_power_ratios=None,
                  shape=SPECTRAL_SHAPE_DEFAULT,
                  method='Ohno 2005'):
    """
    Returns a multi *LED* spectral distribution of given spectral shape at
    given peak wavelengths and full widths at half maximum according to given
    method.

    Parameters
    ----------
    peak_wavelengths : array_like
        Wavelengths the multi *LED* spectral distribution will peak at, i.e.
        the peaks for each generated single *LED* spectral distributions.
    fwhm : array_like
        Full widths at half maximum, i.e. widths of the underlying gaussian
        spectral distributions measured between those points on the *y* axis
        which are half the maximum amplitude.
    peak_power_ratios : array_like, optional
        Peak power ratios for each generated single *LED* spectral
        distributions.
    shape : SpectralShape, optional
        Spectral shape used to create the spectral distribution.
    method : unicode, optional
        **{'Ohno 2005'}**,
        Computation method.

    Returns
    -------
    SpectralDistribution
        Multi *LED* spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape given by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    References
    ----------
    :cite:`Ohno2005`, :cite:`Ohno2008a`

    Examples
    --------
    >>> sd = sd_multi_leds(
    ...     np.array([457, 530, 615]),
    ...     np.array([20, 30, 20]),
    ...     np.array([0.731, 1.000, 1.660]),
    ... )
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[500]  # doctest: +ELLIPSIS
    0.1295132...
    """

    return SD_MULTI_LEDS_METHODS[method](peak_wavelengths, fwhm,
                                         peak_power_ratios, shape)
