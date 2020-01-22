# -*- coding: utf-8 -*-
"""
Dominant Wavelength and Purity
==============================

Defines objects to compute the *dominant wavelength* and *purity* of a colour
and related quantities:

-   :func:`colour.dominant_wavelength`
-   :func:`colour.complementary_wavelength`
-   :func:`colour.excitation_purity`
-   :func:`colour.colorimetric_purity`

See Also
--------
`Dominant Wavelength and Purity Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/dominant_wavelength.ipynb>`_

References
----------
-   :cite:`CIETC1-482004o` : CIE TC 1-48. (2004). 9.1 Dominant wavelength and
    purity. In CIE 015:2004 Colorimetry, 3rd Edition (pp. 32-33).
    ISBN:978-3-901-90633-6
-   :cite:`Erdogana` : Erdogan, T. (n.d.). How to Calculate Luminosity,
    Dominant Wavelength, and Excitation Purity. Retrieved from
    http://www.semrock.com/Data/Sites/1/semrockpdfs/\
whitepaper_howtocalculateluminositywavelengthandpurity.pdf
"""

from __future__ import division, unicode_literals

import numpy as np
import scipy.spatial.distance

from colour.algebra import (euclidean_distance, extend_line_segment,
                            intersect_line_segments)
from colour.colorimetry import CMFS
from colour.models import XYZ_to_xy
from colour.utilities import as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'closest_spectral_locus_wavelength', 'dominant_wavelength',
    'complementary_wavelength', 'excitation_purity', 'colorimetric_purity'
]


def closest_spectral_locus_wavelength(xy, xy_n, xy_s, inverse=False):
    """
    Returns the coordinates and closest spectral locus wavelength index to the
    point where the line defined by the given achromatic stimulus :math:`xy_n`
    to colour stimulus :math:`xy_n` *CIE xy* chromaticity coordinates
    intersects the spectral locus.

    Parameters
    ----------
    xy : array_like
        Colour stimulus *CIE xy* chromaticity coordinates.
    xy_n : array_like
        Achromatic stimulus *CIE xy* chromaticity coordinates.
    xy_s : array_like
        Spectral locus *CIE xy* chromaticity coordinates.
    inverse : bool, optional
        The intersection will be computed using the colour stimulus :math:`xy`
        to achromatic stimulus :math:`xy_n` inverse direction.

    Returns
    -------
    tuple
        Closest wavelength index, intersection point *CIE xy* chromaticity
        coordinates.

    Raises
    ------
    ValueError
        If no closest spectral locus wavelength index and coordinates found.

    Examples
    --------
    >>> xy = np.array([0.54369557, 0.32107944])
    >>> xy_n = np.array([0.31270000, 0.32900000])
    >>> xy_s = XYZ_to_xy(CMFS['CIE 1931 2 Degree Standard Observer'].values)
    >>> ix, intersect = closest_spectral_locus_wavelength(xy, xy_n, xy_s)
    >>> print(ix) #
    256
    >>> print(intersect) # doctest: +ELLIPSIS
    [ 0.6835474...  0.3162840...]
    """

    xy = as_float_array(xy)
    xy_n = np.resize(xy_n, xy.shape)
    xy_s = as_float_array(xy_s)

    xy_e = (extend_line_segment(xy, xy_n)
            if inverse else extend_line_segment(xy_n, xy))

    # Closing horse-shoe shape to handle line of purples intersections.
    xy_s = np.vstack([xy_s, xy_s[0, :]])

    xy_wl = intersect_line_segments(
        np.concatenate((xy_n, xy_e), -1),
        np.hstack([xy_s, np.roll(xy_s, 1, axis=0)])).xy
    xy_wl = xy_wl[~np.isnan(xy_wl).any(axis=-1)]
    if not len(xy_wl):
        raise ValueError(
            'No closest spectral locus wavelength index and coordinates found '
            'for "{0}" colour stimulus and "{1}" achromatic stimulus "xy" '
            'chromaticity coordinates!'.format(xy, xy_n))

    i_wl = np.argmin(scipy.spatial.distance.cdist(xy_wl, xy_s), axis=-1)

    i_wl = np.reshape(i_wl, xy.shape[0:-1])
    xy_wl = np.reshape(xy_wl, xy.shape)

    return i_wl, xy_wl


def dominant_wavelength(xy,
                        xy_n,
                        cmfs=CMFS['CIE 1931 2 Degree Standard Observer'],
                        inverse=False):
    """
    Returns the *dominant wavelength* :math:`\\lambda_d` for given colour
    stimulus :math:`xy` and the related :math:`xy_wl` first and :math:`xy_{cw}`
    second intersection coordinates with the spectral locus.

    In the eventuality where the :math:`xy_wl` first intersection coordinates
    are on the line of purples, the *complementary wavelength* will be
    computed in lieu.

    The *complementary wavelength* is indicated by a negative sign
    and the :math:`xy_{cw}` second intersection coordinates which are set by
    default to the same value than :math:`xy_wl` first intersection coordinates
    will be set to the *complementary dominant wavelength* intersection
    coordinates with the spectral locus.

    Parameters
    ----------
    xy : array_like
        Colour stimulus *CIE xy* chromaticity coordinates.
    xy_n : array_like
        Achromatic stimulus *CIE xy* chromaticity coordinates.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.
    inverse : bool, optional
        Inverse the computation direction to retrieve the
        *complementary wavelength*.

    Returns
    -------
    tuple
        *Dominant wavelength*, first intersection point *CIE xy* chromaticity
        coordinates, second intersection point *CIE xy* chromaticity
        coordinates.

    References
    ----------
    :cite:`CIETC1-482004o`, :cite:`Erdogana`

    Examples
    --------
    *Dominant wavelength* computation:

    >>> from pprint import pprint
    >>> xy = np.array([0.54369557, 0.32107944])
    >>> xy_n = np.array([0.31270000, 0.32900000])
    >>> cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
    >>> pprint(dominant_wavelength(xy, xy_n, cmfs))  # doctest: +ELLIPSIS
    (array(616...),
     array([ 0.6835474...,  0.3162840...]),
     array([ 0.6835474...,  0.3162840...]))

    *Complementary dominant wavelength* is returned if the first intersection
    is located on the line of purples:

    >>> xy = np.array([0.37605506, 0.24452225])
    >>> pprint(dominant_wavelength(xy, xy_n, cmfs))  # doctest: +ELLIPSIS
    (array(-509.0),
     array([ 0.4572314...,  0.1362814...]),
     array([ 0.0104096...,  0.7320745...]))
    """

    xy = as_float_array(xy)
    xy_n = np.resize(xy_n, xy.shape)

    xy_s = XYZ_to_xy(cmfs.values)

    i_wl, xy_wl = closest_spectral_locus_wavelength(xy, xy_n, xy_s, inverse)
    xy_cwl = xy_wl
    wl = cmfs.wavelengths[i_wl]

    xy_e = (extend_line_segment(xy, xy_n)
            if inverse else extend_line_segment(xy_n, xy))
    intersect = intersect_line_segments(
        np.concatenate((xy_n, xy_e), -1), np.hstack([xy_s[0],
                                                     xy_s[-1]])).intersect
    intersect = np.reshape(intersect, wl.shape)

    i_wl_r, xy_cwl_r = closest_spectral_locus_wavelength(
        xy, xy_n, xy_s, not inverse)
    wl_r = -cmfs.wavelengths[i_wl_r]

    wl = np.where(intersect, wl_r, wl)
    xy_cwl = np.where(intersect[..., np.newaxis], xy_cwl_r, xy_cwl)

    return wl, np.squeeze(xy_wl), np.squeeze(xy_cwl)


def complementary_wavelength(xy,
                             xy_n,
                             cmfs=CMFS['CIE 1931 2 Degree Standard Observer']):
    """
    Returns the *complementary wavelength* :math:`\\lambda_c` for given colour
    stimulus :math:`xy` and the related :math:`xy_wl` first and :math:`xy_{cw}`
    second intersection coordinates with the spectral locus.

    In the eventuality where the :math:`xy_wl` first intersection coordinates
    are on the line of purples, the *dominant wavelength* will be computed in
    lieu.

    The *dominant wavelength* is indicated by a negative sign and the
    :math:`xy_{cw}` second intersection coordinates which are set by default to
    the same value than :math:`xy_wl` first intersection coordinates will be
    set to the *dominant wavelength* intersection coordinates with the spectral
    locus.

    Parameters
    ----------
    xy : array_like
        Colour stimulus *CIE xy* chromaticity coordinates.
    xy_n : array_like
        Achromatic stimulus *CIE xy* chromaticity coordinates.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.

    Returns
    -------
    tuple
        *Complementary wavelength*, first intersection point *CIE xy*
        chromaticity coordinates, second intersection point *CIE xy*
        chromaticity coordinates.

    References
    ----------
    :cite:`CIETC1-482004o`, :cite:`Erdogana`

    Examples
    --------
    *Complementary wavelength* computation:

    >>> from pprint import pprint
    >>> xy = np.array([0.37605506, 0.24452225])
    >>> xy_n = np.array([0.31270000, 0.32900000])
    >>> cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
    >>> pprint(complementary_wavelength(xy, xy_n, cmfs))  # doctest: +ELLIPSIS
    (array(509.0),
     array([ 0.0104096...,  0.7320745...]),
     array([ 0.0104096...,  0.7320745...]))

    *Dominant wavelength* is returned if the first intersection is located on
    the line of purples:

    >>> xy = np.array([0.54369557, 0.32107944])
    >>> pprint(complementary_wavelength(xy, xy_n, cmfs))  # doctest: +ELLIPSIS
    (array(492.0),
     array([ 0.0364795 ,  0.3384712...]),
     array([ 0.0364795 ,  0.3384712...]))
    """

    return dominant_wavelength(xy, xy_n, cmfs, True)


def excitation_purity(xy,
                      xy_n,
                      cmfs=CMFS['CIE 1931 2 Degree Standard Observer']):
    """
    Returns the *excitation purity* :math:`P_e` for given colour stimulus
    :math:`xy`.

    Parameters
    ----------
    xy : array_like
        Colour stimulus *CIE xy* chromaticity coordinates.
    xy_n : array_like
        Achromatic stimulus *CIE xy* chromaticity coordinates.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.

    Returns
    -------
    numeric or array_like
        *Excitation purity* :math:`P_e`.

    References
    ----------
    :cite:`CIETC1-482004o`, :cite:`Erdogana`

    Examples
    --------
    >>> xy = np.array([0.54369557, 0.32107944])
    >>> xy_n = np.array([0.31270000, 0.32900000])
    >>> cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
    >>> excitation_purity(xy, xy_n, cmfs)  # doctest: +ELLIPSIS
    0.6228856...
    """

    _wl, xy_wl, _xy_cwl = dominant_wavelength(xy, xy_n, cmfs)

    P_e = euclidean_distance(xy_n, xy) / euclidean_distance(xy_n, xy_wl)

    return P_e


def colorimetric_purity(xy,
                        xy_n,
                        cmfs=CMFS['CIE 1931 2 Degree Standard Observer']):
    """
    Returns the *colorimetric purity* :math:`P_c` for given colour stimulus
    :math:`xy`.

    Parameters
    ----------
    xy : array_like
        Colour stimulus *CIE xy* chromaticity coordinates.
    xy_n : array_like
        Achromatic stimulus *CIE xy* chromaticity coordinates.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.

    Returns
    -------
    numeric or array_like
        *Colorimetric purity* :math:`P_c`.

    References
    ----------
    :cite:`CIETC1-482004o`, :cite:`Erdogana`

    Examples
    --------
    >>> xy = np.array([0.54369557, 0.32107944])
    >>> xy_n = np.array([0.31270000, 0.32900000])
    >>> cmfs = CMFS['CIE 1931 2 Degree Standard Observer']
    >>> colorimetric_purity(xy, xy_n, cmfs)  # doctest: +ELLIPSIS
    0.6135828...
    """

    xy = as_float_array(xy)

    _wl, xy_wl, _xy_cwl = dominant_wavelength(xy, xy_n, cmfs)
    P_e = excitation_purity(xy, xy_n, cmfs)

    P_c = P_e * xy_wl[..., 1] / xy[..., 1]

    return P_c
