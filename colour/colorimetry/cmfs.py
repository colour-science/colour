# -*- coding: utf-8 -*-
"""
Colour Matching Functions
=========================

Defines colour matching functions classes for the datasets from
:mod:`colour.colorimetry.datasets.cmfs` module:

-   :class:`colour.colorimetry.LMS_ConeFundamentals`: Implements support for
    the *Stockman and Sharpe* *LMS* cone fundamentals colour matching
    functions.
-   :class:`colour.colorimetry.RGB_ColourMatchingFunctions`: Implements support
    for the *CIE RGB* colour matching functions.
-   :class:`colour.colorimetry.XYZ_ColourMatchingFunctions`: Implements support
    for the *CIE Standard Observers* *XYZ* colour matching functions.

See Also
--------
`Colour Matching Functions Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/cmfs.ipynb>`_
"""

from __future__ import division, unicode_literals

from colour.colorimetry import MultiSpectralDistributions

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'LMS_ConeFundamentals', 'RGB_ColourMatchingFunctions',
    'XYZ_ColourMatchingFunctions'
]


class LMS_ConeFundamentals(MultiSpectralDistributions):
    """
    Implements support for the Stockman and Sharpe *LMS* cone fundamentals
    colour matching functions.

    Parameters
    ----------
    data : Series or Dataframe or Signal or MultiSignals or \
MultiSpectralDistributions or array_like or dict_like, optional
        Data to be stored in the multi-spectral distributions.
    domain : array_like, optional
        class instances :attr:`colour.continuous.Signal.wavelengths` attribute
        with. If both ``data`` and ``domain`` arguments are defined, the latter
        will be used to initialise the
        Values to initialise the multiple :class:`colour.SpectralDistribution`
        :attr:`colour.continuous.Signal.wavelengths` attribute.
    labels : array_like, optional
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.

    Other Parameters
    ----------------
    name : unicode, optional
       Multi-spectral distributions name.
    interpolator : object, optional
        Interpolator class type to use as interpolating function for the
        :class:`colour.SpectralDistribution` class instances.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function
        of the :class:`colour.SpectralDistribution` class instances.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function for the
        :class:`colour.SpectralDistribution` class instances.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating the extrapolating function
        of the :class:`colour.SpectralDistribution` class instances.
    strict_labels : array_like, optional
        Multi-spectral distributions labels for figures, default to
        :attr:`colour.colorimetry.LMS_ConeFundamentals.labels` attribute value.
    """

    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        super(LMS_ConeFundamentals, self).__init__(
            data,
            domain,
            labels=('l_bar', 'm_bar', 's_bar'),
            strict_labels=('$\\bar{l}$', '$\\bar{m}$', '$\\bar{s}$'),
            **kwargs)


class RGB_ColourMatchingFunctions(MultiSpectralDistributions):
    """
    Implements support for the *CIE RGB* colour matching functions.

    Parameters
    ----------
    data : Series or Dataframe or Signal or MultiSignals or \
MultiSpectralDistributions or array_like or dict_like, optional
        Data to be stored in the multi-spectral distributions.
    domain : array_like, optional
        Values to initialise the multiple :class:`colour.SpectralDistribution`
        class instances :attr:`colour.continuous.Signal.wavelengths` attribute
        with. If both ``data`` and ``domain`` arguments are defined, the latter
        will be used to initialise the
        :attr:`colour.continuous.Signal.wavelengths` attribute.
    labels : array_like, optional
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.

    Other Parameters
    ----------------
    name : unicode, optional
       Multi-spectral distributions name.
    interpolator : object, optional
        Interpolator class type to use as interpolating function for the
        :class:`colour.SpectralDistribution` class instances.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function
        of the :class:`colour.SpectralDistribution` class instances.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function for the
        :class:`colour.SpectralDistribution` class instances.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating the extrapolating function
        of the :class:`colour.SpectralDistribution` class instances.
    strict_labels : array_like, optional
        Multi-spectral distributions labels for figures, default to
        :attr:`colour.colorimetry.RGB_ColourMatchingFunctions.labels` attribute
        value.
    """

    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        super(RGB_ColourMatchingFunctions, self).__init__(
            data,
            domain,
            labels=('r_bar', 'g_bar', 'b_bar'),
            strict_labels=('$\\bar{r}$', '$\\bar{g}$', '$\\bar{b}$'),
            **kwargs)


class XYZ_ColourMatchingFunctions(MultiSpectralDistributions):
    """
    Implements support for the *CIE* Standard Observers *XYZ* colour matching
    functions.

    Parameters
    ----------
    data : Series or Dataframe or Signal or MultiSignals or \
MultiSpectralDistributions or array_like or dict_like, optional
        Data to be stored in the multi-spectral distributions.
    domain : array_like, optional
        Values to initialise the multiple :class:`colour.SpectralDistribution`
        class instances :attr:`colour.continuous.Signal.wavelengths` attribute
        with. If both ``data`` and ``domain`` arguments are defined, the latter
        will be used to initialise the
        :attr:`colour.continuous.Signal.wavelengths` attribute.
    labels : array_like, optional
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.

    Other Parameters
    ----------------
    name : unicode, optional
       Multi-spectral distributions name.
    interpolator : object, optional
        Interpolator class type to use as interpolating function for the
        :class:`colour.SpectralDistribution` class instances.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function
        of the :class:`colour.SpectralDistribution` class instances.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function for the
        :class:`colour.SpectralDistribution` class instances.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating the extrapolating function
        of the :class:`colour.SpectralDistribution` class instances.
    strict_labels : array_like, optional
        Multi-spectral distributions labels for figures, default to
        :attr:`colour.colorimetry.XYZ_ColourMatchingFunctions.labels` attribute
        value.
    """

    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        super(XYZ_ColourMatchingFunctions, self).__init__(
            data,
            domain,
            labels=('x_bar', 'y_bar', 'z_bar'),
            strict_labels=('$\\bar{x}$', '$\\bar{y}$', '$\\bar{z}$'),
            **kwargs)
