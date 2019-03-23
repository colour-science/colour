# -*- coding: utf-8 -*-
"""
Visible Spectrum Volume Computations
====================================

Defines objects related to visible spectrum volume computations.

See Also
--------
`Spectrum Volume Computations Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/volume/spectrum.ipynb>`_

References
----------
-   :cite:`Lindbloom2015` :Lindbloom, B. (2015). About the Lab Gamut.
    Retrieved August 20, 2018, from
    http://www.brucelindbloom.com/LabGamutDisplayHelp.html
-   :cite:`Mansencal2018` :Mansencal, T. (2018). How is the visible gamut
    bounded? Retrieved August 19, 2018, from https://stackoverflow.com/a/\
48396021/931625
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import NearestNeighbourInterpolator
from colour.colorimetry import (
    DEFAULT_SPECTRAL_SHAPE, STANDARD_OBSERVERS_CMFS,
    multi_sds_to_XYZ_integration, SpectralShape, sd_ones)
from colour.volume import is_within_mesh_volume

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'generate_pulse_waves', 'XYZ_outer_surface', 'is_within_visible_spectrum'
]

_XYZ_OUTER_SURFACE_CACHE = {}
_XYZ_OUTER_SURFACE_POINTS_CACHE = {}


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
    square_waves_basis = np.tril(np.ones((bins, bins)))[0:-1, :]
    for square_wave_basis in square_waves_basis:
        for i in range(bins):
            square_waves.append(np.roll(square_wave_basis, i))

    return np.vstack([np.zeros(bins), np.vstack(square_waves), np.ones(bins)])


def XYZ_outer_surface(
        interval=10,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'],
        illuminant=sd_ones(STANDARD_OBSERVERS_CMFS[
            'CIE 1931 2 Degree Standard Observer'].shape)):
    """
    Generates the *CIE XYZ* colourspace outer surface for given colour matching
    functions using multi-spectral conversion of pulse waves to *CIE XYZ*
    tristimulus values.

    Parameters
    ----------
    interval : int, optional
        Wavelength :math:`\\lambda_{i}` range interval used to compute the
        pulse waves.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.

    Returns
    -------
    ndarray
        Outer surface *CIE XYZ* tristimulus values.

    References
    ----------
    :cite:`Lindbloom2015`, :cite:`Mansencal2018`

    Examples
    --------
    >>> XYZ_outer_surface(84)  # doctest: +ELLIPSIS
    array([[  0.0000000...e+00,   0.0000000...e+00,   0.0000000...e+00],
           [  1.4766924...e-03,   4.1530347...e-05,   6.9884362...e-03],
           [  1.6281275...e-01,   3.7114387...e-02,   9.0151471...e-01],
           [  1.8650894...e-01,   5.6617464...e-01,   9.1355179...e-02],
           [  6.1555347...e-01,   3.8427775...e-01,   4.7422070...e-04],
           [  3.3622045...e-02,   1.2354556...e-02,   0.0000000...e+00],
           [  1.0279500...e-04,   3.7121158...e-05,   0.0000000...e+00],
           [  1.6428945...e-01,   3.7155917...e-02,   9.0850314...e-01],
           [  3.4932169...e-01,   6.0328903...e-01,   9.9286989...e-01],
           [  8.0206241...e-01,   9.5045240...e-01,   9.1829399...e-02],
           [  6.4917552...e-01,   3.9663231...e-01,   4.7422070...e-04],
           [  3.3724840...e-02,   1.2391678...e-02,   0.0000000...e+00],
           [  1.5794874...e-03,   7.8651505...e-05,   6.9884362...e-03],
           [  3.5079839...e-01,   6.0333056...e-01,   9.9985832...e-01],
           [  9.6487517...e-01,   9.8756679...e-01,   9.9334411...e-01],
           [  8.3568446...e-01,   9.6280696...e-01,   9.1829399...e-02],
           [  6.4927831...e-01,   3.9666943...e-01,   4.7422070...e-04],
           [  3.5201532...e-02,   1.2433208...e-02,   6.9884362...e-03],
           [  1.6439224...e-01,   3.7193038...e-02,   9.0850314...e-01],
           [  9.6635186...e-01,   9.8760832...e-01,   1.0003325...e+00],
           [  9.9849722...e-01,   9.9992134...e-01,   9.9334411...e-01],
           [  8.3578726...e-01,   9.6284408...e-01,   9.1829399...e-02],
           [  6.5075501...e-01,   3.9671096...e-01,   7.4626569...e-03],
           [  1.9801429...e-01,   4.9547595...e-02,   9.0850314...e-01],
           [  3.5090118...e-01,   6.0336768...e-01,   9.9985832...e-01],
           [  9.9997391...e-01,   9.9996287...e-01,   1.0003325...e+00],
           [  9.9860001...e-01,   9.9995847...e-01,   9.9334411...e-01],
           [  8.3726395...e-01,   9.6288561...e-01,   9.8817836...e-02],
           [  8.1356776...e-01,   4.3382535...e-01,   9.0897737...e-01],
           [  3.8452323...e-01,   6.1572224...e-01,   9.9985832...e-01],
           [  9.6645466...e-01,   9.8764544...e-01,   1.0003325...e+00],
           [  1.0000767...e+00,   1.0000000...e+00,   1.0003325...e+00]])

    """

    key = (interval, hash(cmfs), hash(illuminant))
    XYZ = _XYZ_OUTER_SURFACE_CACHE.get(key)
    if XYZ is None:
        wavelengths = SpectralShape(DEFAULT_SPECTRAL_SHAPE.start,
                                    DEFAULT_SPECTRAL_SHAPE.end,
                                    interval).range()
        values = []
        domain = DEFAULT_SPECTRAL_SHAPE.range()
        for wave in generate_pulse_waves(len(wavelengths)):
            values.append(
                NearestNeighbourInterpolator(wavelengths, wave)(domain))

        XYZ = multi_sds_to_XYZ_integration(values, DEFAULT_SPECTRAL_SHAPE,
                                           cmfs, illuminant)

        XYZ = XYZ / np.max(XYZ[-1, 1])

        _XYZ_OUTER_SURFACE_CACHE[key] = XYZ

    return XYZ


def is_within_visible_spectrum(
        XYZ,
        interval=10,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'],
        illuminant=sd_ones(STANDARD_OBSERVERS_CMFS[
            'CIE 1931 2 Degree Standard Observer'].shape),
        tolerance=None):
    """
    Returns if given *CIE XYZ* tristimulus values are within visible spectrum
    volume / given colour matching functions volume.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    interval : int, optional
        Wavelength :math:`\\lambda_{i}` range interval used to compute the
        pulse waves for the *CIE XYZ* colourspace outer surface.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.
    tolerance : numeric, optional
        Tolerance allowed in the inside-triangle check.

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

    key = (interval, hash(cmfs), hash(illuminant))
    vertices = _XYZ_OUTER_SURFACE_POINTS_CACHE.get(key)
    if vertices is None:
        _XYZ_OUTER_SURFACE_POINTS_CACHE[key] = vertices = (XYZ_outer_surface(
            interval,
            STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'],
            illuminant))

    return is_within_mesh_volume(XYZ, vertices, tolerance)
