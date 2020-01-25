# -*- coding: utf-8 -*-
"""
RGB Displays
============

Defines spectral distributions classes for the datasets from
:mod:`colour.characterisation.datasets.displays` module:

-   :class:`colour.characterisation.RGB_DisplayPrimaries`: Implements support
    for a *RGB* display (such as a *CRT* or *LCD*) primaries multi-spectral
    distributions.

See Also
--------
`Displays Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/characterisation/displays.ipynb>`_
"""

from __future__ import division, unicode_literals

from colour.colorimetry import MultiSpectralDistributions

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['RGB_DisplayPrimaries']


class RGB_DisplayPrimaries(MultiSpectralDistributions):
    """
    Implements support for a *RGB* display (such as a *CRT* or *LCD*)
    primaries multi-spectral distributions.

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
        :attr:`colour.characterisation.RGB_DisplayPrimaries.labels` attribute
        value.
    """

    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        super(RGB_DisplayPrimaries, self).__init__(
            data, domain, labels=('red', 'green', 'blue'), **kwargs)
