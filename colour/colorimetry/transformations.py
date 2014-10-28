#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour Matching Functions Transformations
=========================================

Defines various educational objects for colour matching functions
transformations:

-   :func:`RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs`
-   :func:`RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs`
-   :func:`RGB_10_degree_cmfs_to_LMS_10_degree_cmfs`
-   :func:`LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs`
-   :func:`LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs`

See Also
--------
`Colour Matching Functions IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/colorimetry/cmfs.ipynb>`_  # noqa
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import LMS_CMFS, RGB_CMFS, PHOTOPIC_LEFS

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs',
           'RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs',
           'RGB_10_degree_cmfs_to_LMS_10_degree_cmfs',
           'LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs',
           'LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs']


def RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wavelength):
    """
    Converts *Wright & Guild 1931 2 Degree RGB CMFs* colour matching functions
    into the *CIE 1931 2 Degree Standard Observer* colour matching functions.

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray, (3,)
        *CIE 1931 2 Degree Standard Observer* spectral tristimulus values.

    Raises
    ------
    KeyError
        If wavelength :math:`\lambda` is not available in the colour matching
        functions.

    See Also
    --------
    :attr:`colour.colorimetry.dataset.cmfs.RGB_CMFS`

    Notes
    -----
    -   Data for the *CIE 1931 2 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    .. [1]  Wyszecki, G., & Stiles, W. S. (2000). Table 1(3.3.3). In Color
            Science: Concepts and Methods, Quantitative Data and Formulae
            (pp. 138–139). Wiley. ISBN:978-0471399186

    Examples
    --------
    >>> RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0113577...,  0.004102  ,  0.        ])
    """

    cmfs = RGB_CMFS.get('Wright & Guild 1931 2 Degree RGB CMFs')
    r_bar, g_bar, b_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(
        wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, g_bar, b_bar):
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            wavelength, cmfs.name, cmfs.shape))

    r = r_bar / (r_bar + g_bar + b_bar)
    g = g_bar / (r_bar + g_bar + b_bar)
    b = b_bar / (r_bar + g_bar + b_bar)

    x = ((0.49000 * r + 0.31000 * g + 0.20000 * b) /
         (0.66697 * r + 1.13240 * g + 1.20063 * b))
    y = ((0.17697 * r + 0.81240 * g + 0.01063 * b) /
         (0.66697 * r + 1.13240 * g + 1.20063 * b))
    z = ((0.00000 * r + 0.01000 * g + 0.99000 * b) /
         (0.66697 * r + 1.13240 * g + 1.20063 * b))

    V = PHOTOPIC_LEFS.get('CIE 1924 Photopic Standard Observer').clone()
    V.align(cmfs.shape)
    L = V.get(wavelength)

    x_bar = x / y * L
    y_bar = L
    z_bar = z / y * L

    return np.array([x_bar, y_bar, z_bar])


def RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wavelength):
    """
    Converts *Stiles & Burch 1959 10 Degree RGB CMFs* colour matching
    functions into the *CIE 1964 10 Degree Standard Observer* colour matching
    functions.

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray, (3,)
        *CIE 1964 10 Degree Standard Observer* spectral tristimulus values.

    Raises
    ------
    KeyError
        If wavelength :math:`\lambda` is not available in the colour matching
        functions.

    See Also
    --------
    :attr:`colour.colorimetry.dataset.cmfs.RGB_CMFS`

    Notes
    -----
    -   Data for the *CIE 1964 10 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    .. [2]  Wyszecki, G., & Stiles, W. S. (2000). The CIE 1964 Standard
            Observer. In Color Science: Concepts and Methods, Quantitative
            Data and Formulae (p. 141). Wiley. ISBN:978-0471399186

    Examples
    --------
    >>> RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([  9.6432150...e-03,   3.7526317...e-03,  -4.1078830...e-06])
    """

    cmfs = RGB_CMFS.get('Stiles & Burch 1959 10 Degree RGB CMFs')
    r_bar, g_bar, b_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(
        wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, g_bar, b_bar):
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            wavelength, cmfs.name, cmfs.shape))

    x_bar = 0.341080 * r_bar + 0.189145 * g_bar + 0.387529 * b_bar
    y_bar = 0.139058 * r_bar + 0.837460 * g_bar + 0.073316 * b_bar
    z_bar = 0.000000 * r_bar + 0.039553 * g_bar + 2.026200 * b_bar

    return np.array([x_bar, y_bar, z_bar])


def RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(wavelength):
    """
    Converts *Stiles & Burch 1959 10 Degree RGB CMFs* colour matching
    functions into the *Stockman & Sharpe 10 Degree Cone Fundamentals*
    spectral sensitivity functions.

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray, (3,)
        *Stockman & Sharpe 10 Degree Cone Fundamentals* spectral tristimulus
        values.

    Raises
    ------
    KeyError
        If wavelength :math:`\lambda` is not available in the colour matching
        functions.

    Notes
    -----
    -   Data for the *Stockman & Sharpe 10 Degree Cone Fundamentals* already
        exists, this definition is intended for educational purpose.

    References
    ----------
    .. [3]  CIE TC 1-36. (2006). CIE 170-1:2006 Fundamental Chromaticity
            Diagram with Physiological Axes - Part 1 (pp. 1–56).
            ISBN:978-3-901-90646-6

    Examples
    --------
    >>> RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0052860...,  0.0003252...,  0.        ])
    """

    cmfs = RGB_CMFS.get('Stiles & Burch 1959 10 Degree RGB CMFs')
    r_bar, g_bar, z_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(
        wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, g_bar, z_bar):
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            wavelength, cmfs.name, cmfs.shape))

    l_bar = 0.192325269 * r_bar + 0.749548882 * g_bar + 0.0675726702 * z_bar
    g_bar = 0.0192290085 * r_bar + 0.940908496 * g_bar + 0.113830196 * z_bar
    z_bar = (0.0105107859 * g_bar + 0.991427669 * z_bar
             if wavelength <= 505 else 0)

    return np.array([l_bar, g_bar, z_bar])


def LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(wavelength):
    """
    Converts *Stockman & Sharpe 2 Degree Cone Fundamentals* colour matching
    functions into the *CIE 2012 2 Degree Standard Observer* colour matching
    functions.

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray, (3,)
        *CIE 2012 2 Degree Standard Observer* spectral tristimulus values.

    Raises
    ------
    KeyError
        If wavelength :math:`\lambda` is not available in the colour matching
        functions.

    Notes
    -----
    -   Data for the *CIE 2012 2 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    .. [4]  CVRL. (n.d.). CIE (2012) 2-deg XYZ “physiologically-relevant”
            colour matching functions. Retrieved June 25, 2014, from
            http://www.cvrl.org/database/text/cienewxyz/cie2012xyz2.htm

    Examples
    --------
    >>> LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0109677...,  0.0041959...,  0.        ])
    """

    cmfs = LMS_CMFS.get('Stockman & Sharpe 2 Degree Cone Fundamentals')
    l_bar, m_bar, s_bar = cmfs.l_bar.get(wavelength), cmfs.m_bar.get(
        wavelength), cmfs.s_bar.get(wavelength)
    if None in (l_bar, m_bar, s_bar):
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            wavelength, cmfs.name, cmfs.shape))

    x_bar = 1.94735469 * l_bar - 1.41445123 * m_bar + 0.36476327 * s_bar
    y_bar = 0.68990272 * l_bar + 0.34832189 * m_bar
    z_bar = 1.93485343 * s_bar

    return np.array([x_bar, y_bar, z_bar])


def LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(wavelength):
    """
    Converts *Stockman & Sharpe 10 Degree Cone Fundamentals* colour matching
    functions into the *CIE 2012 10 Degree Standard Observer* colour matching
    functions.

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray, (3,)
        *CIE 2012 10 Degree Standard Observer* spectral tristimulus values.

    Raises
    ------
    KeyError
        If wavelength :math:`\lambda` is not available in the colour matching
        functions.

    Notes
    -----
    -   Data for the *CIE 2012 10 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    .. [5]  CVRL. (n.d.). CIE (2012) 10-deg XYZ “physiologically-relevant”
            colour matching functions. Retrieved June 25, 2014, from
            http://www.cvrl.org/database/text/cienewxyz/cie2012xyz10.htm

    Examples
    --------
    >>> LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0098162...,  0.0037761...,  0.        ])
    """

    cmfs = LMS_CMFS.get('Stockman & Sharpe 10 Degree Cone Fundamentals')
    l_bar, m_bar, s_bar = cmfs.l_bar.get(wavelength), cmfs.m_bar.get(
        wavelength), cmfs.s_bar.get(wavelength)
    if None in (l_bar, m_bar, s_bar):
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            wavelength, cmfs.name, cmfs.shape))

    x_bar = 1.93986443 * l_bar - 1.34664359 * m_bar + 0.43044935 * s_bar
    y_bar = 0.69283932 * l_bar + 0.34967567 * m_bar
    z_bar = 2.14687945 * s_bar

    return np.array([x_bar, y_bar, z_bar])
