# -*- coding: utf-8 -*-
"""
Colour Matching Functions Transformations
=========================================

Defines various educational objects for colour matching functions
transformations:

-   :func:`colour.colorimetry.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs`
-   :func:`colour.colorimetry.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs`
-   :func:`colour.colorimetry.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs`
-   :func:`colour.colorimetry.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs`
-   :func:`colour.colorimetry.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs`

References
----------
-   :cite:`CIETC1-362006a` : CIE TC 1-36. (2006). CIE 170-1:2006 Fundamental
    Chromaticity Diagram with Physiological Axes - Part 1. Commission
    Internationale de l'Eclairage. ISBN:978-3-901906-46-6
-   :cite:`CVRLp` : CVRL. (n.d.). CIE (2012) 10-deg XYZ
    "physiologically-relevant" colour matching functions. Retrieved June 25,
    2014, from http://www.cvrl.org/database/text/cienewxyz/cie2012xyz10.htm
-   :cite:`CVRLv` : CVRL. (n.d.). CIE (2012) 2-deg XYZ
    "physiologically-relevant" colour matching functions. Retrieved June 25,
    2014, from http://www.cvrl.org/database/text/cienewxyz/cie2012xyz2.htm
-   :cite:`Wyszecki2000be` : Wyszecki, Günther, & Stiles, W. S. (2000). The
    CIE 1964 Standard Observer. In Color Science: Concepts and Methods,
    Quantitative Data and Formulae (p. 141). Wiley. ISBN:978-0-471-39918-6
-   :cite:`Wyszecki2000bg` : Wyszecki, Günther, & Stiles, W. S. (2000). Table
    1(3.3.3). In Color Science: Concepts and Methods, Quantitative Data and
    Formulae (pp. 138-139). Wiley. ISBN:978-0-471-39918-6
"""

from __future__ import annotations

import numpy as np

from colour.algebra import vector_dot
from colour.colorimetry import (
    MSDS_CMFS_LMS,
    MSDS_CMFS_RGB,
    SDS_LEFS_PHOTOPIC,
    reshape_sd,
)
from colour.hints import FloatingOrArrayLike, NDArray
from colour.utilities import tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs',
    'RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs',
    'RGB_10_degree_cmfs_to_LMS_10_degree_cmfs',
    'LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs',
    'LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs',
]


def RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(
        wavelength: FloatingOrArrayLike) -> NDArray:
    """
    Converts *Wright & Guild 1931 2 Degree RGB CMFs* colour matching functions
    into the *CIE 1931 2 Degree Standard Observer* colour matching functions.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in nm.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE 1931 2 Degree Standard Observer* spectral tristimulus values.

    Notes
    -----
    -   Data for the *CIE 1931 2 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    :cite:`Wyszecki2000bg`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0113577...,  0.004102  ,  0.        ])
    """

    cmfs = MSDS_CMFS_RGB['Wright & Guild 1931 2 Degree RGB CMFs']

    rgb_bar = cmfs[wavelength]

    rgb = rgb_bar / np.sum(rgb_bar)

    M1 = np.array([
        [0.49000, 0.31000, 0.20000],
        [0.17697, 0.81240, 0.01063],
        [0.00000, 0.01000, 0.99000],
    ])

    M2 = np.array([
        [0.66697, 1.13240, 1.20063],
        [0.66697, 1.13240, 1.20063],
        [0.66697, 1.13240, 1.20063],
    ])

    xyz = vector_dot(M1, rgb)
    xyz /= vector_dot(M2, rgb)

    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    V = reshape_sd(SDS_LEFS_PHOTOPIC['CIE 1924 Photopic Standard Observer'],
                   cmfs.shape)
    L = V[wavelength]

    x_bar = x / y * L
    y_bar = L
    z_bar = z / y * L

    xyz_bar = tstack([x_bar, y_bar, z_bar])

    return xyz_bar


def RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(
        wavelength: FloatingOrArrayLike) -> NDArray:
    """
    Converts *Stiles & Burch 1959 10 Degree RGB CMFs* colour matching
    functions into the *CIE 1964 10 Degree Standard Observer* colour matching
    functions.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in nm.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE 1964 10 Degree Standard Observer* spectral tristimulus values.

    Notes
    -----
    -   Data for the *CIE 1964 10 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    :cite:`Wyszecki2000be`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0096432...,  0.0037526..., -0.0000041...])
    """

    cmfs = MSDS_CMFS_RGB['Stiles & Burch 1959 10 Degree RGB CMFs']

    rgb_bar = cmfs[wavelength]

    M = np.array([
        [0.341080, 0.189145, 0.387529],
        [0.139058, 0.837460, 0.073316],
        [0.000000, 0.039553, 2.026200],
    ])

    xyz_bar = vector_dot(M, rgb_bar)

    return xyz_bar


def RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(
        wavelength: FloatingOrArrayLike) -> NDArray:
    """
    Converts *Stiles & Burch 1959 10 Degree RGB CMFs* colour matching
    functions into the *Stockman & Sharpe 10 Degree Cone Fundamentals*
    spectral sensitivity functions.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in nm.

    Returns
    -------
    :class:`numpy.ndarray`
        *Stockman & Sharpe 10 Degree Cone Fundamentals* spectral tristimulus
        values.

    Notes
    -----
    -   Data for the *Stockman & Sharpe 10 Degree Cone Fundamentals* already
        exists, this definition is intended for educational purpose.

    References
    ----------
    :cite:`CIETC1-362006a`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0052860...,  0.0003252...,  0.        ])
    """

    cmfs = MSDS_CMFS_RGB['Stiles & Burch 1959 10 Degree RGB CMFs']

    rgb_bar = cmfs[wavelength]

    M = np.array([
        [0.1923252690, 0.749548882, 0.0675726702],
        [0.0192290085, 0.940908496, 0.113830196],
        [0.0000000000, 0.0105107859, 0.991427669],
    ])

    lms_bar = vector_dot(M, rgb_bar)
    lms_bar[..., -1][np.asarray(np.asarray(wavelength) > 505)] = 0

    return lms_bar


def LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(
        wavelength: FloatingOrArrayLike) -> NDArray:
    """
    Converts *Stockman & Sharpe 2 Degree Cone Fundamentals* colour matching
    functions into the *CIE 2012 2 Degree Standard Observer* colour matching
    functions.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in nm.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE 2012 2 Degree Standard Observer* spectral tristimulus values.

    Notes
    -----
    -   Data for the *CIE 2012 2 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    :cite:`CVRLv`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0109677...,  0.0041959...,  0.        ])
    """

    cmfs = MSDS_CMFS_LMS['Stockman & Sharpe 2 Degree Cone Fundamentals']

    lms_bar = cmfs[wavelength]

    M = np.array([
        [1.94735469, -1.41445123, 0.36476327],
        [0.68990272, 0.34832189, 0.00000000],
        [0.00000000, 0.00000000, 1.93485343],
    ])

    xyz_bar = vector_dot(M, lms_bar)

    return xyz_bar


def LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(
        wavelength: FloatingOrArrayLike) -> NDArray:
    """
    Converts *Stockman & Sharpe 10 Degree Cone Fundamentals* colour matching
    functions into the *CIE 2012 10 Degree Standard Observer* colour matching
    functions.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in nm.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE 2012 10 Degree Standard Observer* spectral tristimulus values.

    Notes
    -----
    -   Data for the *CIE 2012 10 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    :cite:`CVRLp`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0098162...,  0.0037761...,  0.        ])
    """

    cmfs = MSDS_CMFS_LMS['Stockman & Sharpe 10 Degree Cone Fundamentals']

    lms_bar = cmfs[wavelength]

    M = np.array([
        [1.93986443, -1.34664359, 0.43044935],
        [0.69283932, 0.34967567, 0.00000000],
        [0.00000000, 0.00000000, 2.14687945],
    ])

    xyz_bar = vector_dot(M, lms_bar)

    return xyz_bar
