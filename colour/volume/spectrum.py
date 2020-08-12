# -*- coding: utf-8 -*-
"""
Visible Spectrum Volume Computations
====================================

Defines objects related to visible spectrum volume computations.

References
----------
-   :cite:`Lindbloom2015` : Lindbloom, B. (2015). About the Lab Gamut.
    Retrieved August 20, 2018, from
    http://www.brucelindbloom.com/LabGamutDisplayHelp.html
-   :cite:`Mansencal2018` : Mansencal, T. (2018). How is the visible gamut
    bounded? Retrieved August 19, 2018, from
    https://stackoverflow.com/a/48396021/931625
"""

from __future__ import division, unicode_literals

import numpy as np
import six

from colour.colorimetry import (MSDS_CMFS, msds_to_XYZ, SpectralShape, sd_ones)
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.volume import is_within_mesh_volume
from colour.utilities import zeros

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'SPECTRAL_SHAPE_OUTER_SURFACE_XYZ', 'generate_pulse_waves',
    'XYZ_outer_surface', 'is_within_visible_spectrum'
]

SPECTRAL_SHAPE_OUTER_SURFACE_XYZ = SpectralShape(360, 780, 5)
"""
Default spectral shape according to *ASTM E308-15* practise shape but using an
interval of 5.

SPECTRAL_SHAPE_OUTER_SURFACE_XYZ : SpectralShape
"""

_CACHE_OUTER_SURFACE_XYZ = {}
_CACHE_OUTER_SURFACE_XYZ_POINTS = {}


def generate_pulse_waves(bins):
    """
    Generates the pulse waves of given number of bins necessary to totally
    stimulate the colour matching functions.

    Assuming 5 bins, a first set of SPDs would be as follows::

        1 0 0 0 0
        0 1 0 0 0
        0 0 1 0 0
        0 0 0 1 0
        0 0 0 0 1

    The second one::

        1 1 0 0 0
        0 1 1 0 0
        0 0 1 1 0
        0 0 0 1 1
        1 0 0 0 1

    The third:

        1 1 1 0 0
        0 1 1 1 0
        0 0 1 1 1
        1 0 0 1 1
        1 1 0 0 1

    Etc...

    Parameters
    ----------
    bins : int
        Number of bins of the pulse waves.

    Returns
    -------
    ndarray
        Pulse waves.

    References
    ----------
    :cite:`Lindbloom2015`, :cite:`Mansencal2018`

    Examples
    --------
    >>> generate_pulse_waves(5)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  1.],
           [ 1.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  1.,  1.],
           [ 1.,  0.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  1.,  1.],
           [ 1.,  0.,  0.,  1.,  1.],
           [ 1.,  1.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.],
           [ 1.,  0.,  1.,  1.,  1.],
           [ 1.,  1.,  0.,  1.,  1.],
           [ 1.,  1.,  1.,  0.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]])
    """

    square_waves = []
    square_waves_basis = np.tril(
        np.ones((bins, bins), dtype=DEFAULT_FLOAT_DTYPE))[0:-1, :]
    for square_wave_basis in square_waves_basis:
        for i in range(bins):
            square_waves.append(np.roll(square_wave_basis, i))

    return np.vstack([
        zeros(bins),
        np.vstack(square_waves),
        np.ones(bins, dtype=DEFAULT_FLOAT_DTYPE)
    ])


def XYZ_outer_surface(cmfs=MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
                      .copy().align(SPECTRAL_SHAPE_OUTER_SURFACE_XYZ),
                      illuminant=sd_ones(SPECTRAL_SHAPE_OUTER_SURFACE_XYZ),
                      **kwargs):
    """
    Generates the *CIE XYZ* colourspace outer surface for given colour matching
    functions using multi-spectral conversion of pulse waves to *CIE XYZ*
    tristimulus values.

    Parameters
    ----------
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.msds_to_XYZ`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    ndarray
        Outer surface *CIE XYZ* tristimulus values.

    References
    ----------
    :cite:`Lindbloom2015`, :cite:`Mansencal2018`

    Examples
    --------
    >>> from colour.colorimetry import SPECTRAL_SHAPE_DEFAULT
    >>> shape = SpectralShape(
    ...     SPECTRAL_SHAPE_DEFAULT.start, SPECTRAL_SHAPE_DEFAULT.end, 84)
    >>> cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    >>> XYZ_outer_surface(cmfs.copy().align(shape))  # doctest: +ELLIPSIS
    array([[  0.0000000...e+00,   0.0000000...e+00,   0.0000000...e+00],
           [  9.6361381...e-05,   2.9056776...e-06,   4.4961226...e-04],
           [  2.5910529...e-01,   2.1031298...e-02,   1.3207468...e+00],
           [  1.0561021...e-01,   6.2038243...e-01,   3.5423571...e-02],
           [  7.2647980...e-01,   3.5460869...e-01,   2.1005149...e-04],
           [  1.0971874...e-02,   3.9635453...e-03,   0.0000000...e+00],
           [  3.0792572...e-05,   1.1119762...e-05,   0.0000000...e+00],
           [  2.5920165...e-01,   2.1034203...e-02,   1.3211965...e+00],
           [  3.6471551...e-01,   6.4141373...e-01,   1.3561704...e+00],
           [  8.3209002...e-01,   9.7499113...e-01,   3.5633622...e-02],
           [  7.3745167...e-01,   3.5857224...e-01,   2.1005149...e-04],
           [  1.1002667...e-02,   3.9746651...e-03,   0.0000000...e+00],
           [  1.2715395...e-04,   1.4025439...e-05,   4.4961226...e-04],
           [  3.6481187...e-01,   6.4141663...e-01,   1.3566200...e+00],
           [  1.0911953...e+00,   9.9602242...e-01,   1.3563805...e+00],
           [  8.4306189...e-01,   9.7895467...e-01,   3.5633622...e-02],
           [  7.3748247...e-01,   3.5858336...e-01,   2.1005149...e-04],
           [  1.1099028...e-02,   3.9775708...e-03,   4.4961226...e-04],
           [  2.5923244...e-01,   2.1045323...e-02,   1.3211965...e+00],
           [  1.0912916...e+00,   9.9602533...e-01,   1.3568301...e+00],
           [  1.1021671...e+00,   9.9998597...e-01,   1.3563805...e+00],
           [  8.4309268...e-01,   9.7896579...e-01,   3.5633622...e-02],
           [  7.3757883...e-01,   3.5858626...e-01,   6.5966375...e-04],
           [  2.7020432...e-01,   2.5008868...e-02,   1.3211965...e+00],
           [  3.6484266...e-01,   6.4142775...e-01,   1.3566200...e+00],
           [  1.1022635...e+00,   9.9998888...e-01,   1.3568301...e+00],
           [  1.1021979...e+00,   9.9999709...e-01,   1.3563805...e+00],
           [  8.4318905...e-01,   9.7896870...e-01,   3.6083235...e-02],
           [  9.9668412...e-01,   3.7961756...e-01,   1.3214065...e+00],
           [  3.7581454...e-01,   6.4539130...e-01,   1.3566200...e+00],
           [  1.0913224...e+00,   9.9603645...e-01,   1.3568301...e+00],
           [  1.1022943...e+00,   1.0000000...e+00,   1.3568301...e+00]])
    """

    settings = {'method': 'Integration', 'shape': cmfs.shape}
    settings.update(kwargs)

    key = (hash(cmfs), hash(illuminant), six.text_type(settings))
    XYZ = _CACHE_OUTER_SURFACE_XYZ.get(key)

    if XYZ is None:
        pulse_waves = generate_pulse_waves(len(cmfs.wavelengths))
        XYZ = msds_to_XYZ(pulse_waves, cmfs, illuminant, **settings) / 100

        _CACHE_OUTER_SURFACE_XYZ[key] = XYZ

    return XYZ


def is_within_visible_spectrum(
        XYZ,
        cmfs=MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy().align(SPECTRAL_SHAPE_OUTER_SURFACE_XYZ),
        illuminant=sd_ones(SPECTRAL_SHAPE_OUTER_SURFACE_XYZ),
        tolerance=None,
        **kwargs):
    """
    Returns if given *CIE XYZ* tristimulus values are within visible spectrum
    volume / given colour matching functions volume.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.
    tolerance : numeric, optional
        Tolerance allowed in the inside-triangle check.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.msds_to_XYZ`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    bool
        Is within visible spectrum.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> is_within_visible_spectrum(np.array([0.3205, 0.4131, 0.51]))
    array(True, dtype=bool)
    >>> a = np.array([[0.3205, 0.4131, 0.51],
    ...               [-0.0005, 0.0031, 0.001]])
    >>> is_within_visible_spectrum(a)
    array([ True, False], dtype=bool)
    """

    key = (hash(cmfs), hash(illuminant), six.text_type(kwargs))
    vertices = _CACHE_OUTER_SURFACE_XYZ_POINTS.get(key)

    if vertices is None:
        _CACHE_OUTER_SURFACE_XYZ_POINTS[key] = vertices = (XYZ_outer_surface(
            cmfs, illuminant, **kwargs))

    return is_within_mesh_volume(XYZ, vertices, tolerance)
