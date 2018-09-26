# -*- coding: utf-8 -*-
"""
Spectral Generation
===================

Defines various classes performing spectral generation:

See Also
--------
`Spectrum Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/generation.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.colorimetry import (DEFAULT_SPECTRAL_SHAPE,
                                SpectralPowerDistribution)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['constant_spd', 'zeros_spd', 'ones_spd']


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
