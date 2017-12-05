#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RGB Displays
============

Defines spectral power distributions classes for the dataset from
:mod:`colour.characterisation.dataset.displays` module:

-   :class:`RGB_DisplayPrimaries`: Implements support for a *RGB* display (such
    as a *CRT* or *LCD*) primaries tri-spectral power distributions.

See Also
--------
`Displays Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/characterisation/displays.ipynb>`_
"""

from __future__ import division, unicode_literals

from colour.colorimetry import MultiSpectralPowerDistribution

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RGB_DisplayPrimaries']


class RGB_DisplayPrimaries(MultiSpectralPowerDistribution):
    """
    Implements support for a *RGB* display (such as a *CRT* or *LCD*)
    primaries tri-spectral power distributions.

    Parameters
    ----------
    data : Series or Dataframe or Signal or MultiSignal or \
MultiSpectralPowerDistribution or array_like or dict_like, optional
        Data to be stored in the multi-spectral power distribution.
    domain : array_like, optional
        Values to initialise the multiple :class:`SpectralPowerDistribution`
        class instances :attr:`Signal.wavelengths` attribute with. If both
        `data` and `domain` arguments are defined, the latter with be used to
        initialise the :attr:`Signal.wavelengths` attribute.
    labels : array_like, optional
        Names to use for the :class:`SpectralPowerDistribution` class
        instances.

    Other Parameters
    ----------------
    name : unicode, optional
       Multi-spectral power distribution name.
    interpolator : object, optional
        Interpolator class type to use as interpolating function for the
        :class:`SpectralPowerDistribution` class instances.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function
        of the :class:`SpectralPowerDistribution` class instances.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function for the
        :class:`SpectralPowerDistribution` class instances.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating the extrapolating function
        of the :class:`SpectralPowerDistribution` class instances.
    strict_labels : array_like, optional
        Multi-spectral power distribution labels for figures, default to
        :attr:`RGB_DisplayPrimaries.labels` attribute value.
    """

    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        super(RGB_DisplayPrimaries, self).__init__(
            data, domain, labels=('red', 'green', 'blue'), **kwargs)
