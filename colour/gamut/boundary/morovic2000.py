# -*- coding: utf-8 -*-
"""
Gamut Boundary Descriptor (GDB) - Morovic and Luo (2000)
========================================================

Defines the *Morovic and Luo (2000)* *Gamut Boundary Descriptor (GDB)*
computation objects:

-   :func:`colour.gamut.area_boundary_descriptor_Morovic2000`
-   :func:`colour.gamut.volume_boundary_descriptor_Morovic2000`
-   :func:`colour.gamut.gamut_boundary_descriptor_Morovic2000`

See Also
--------
`Gamut Boundary Descriptor Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/gamut/boundary.ipynb>`_

References
----------
-   :cite:`Morovic2000` : Morovič, J., & Luo, M. R. (2000). Calculating medium
    and image gamut boundaries for gamut mapping. Color Research and
    Application, 25(6), 394–401.
    https://doi.org/10.1002/1520-6378(200012)25:63.0.CO;2-Y
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import cartesian_to_polar, cartesian_to_spherical
from colour.gamut.boundary import (close_gamut_boundary_descriptor,
                                   fill_gamut_boundary_descriptor)
from colour.utilities import (as_int_array, as_float_array, filter_kwargs,
                              tsplit)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'area_boundary_descriptor_Morovic2000',
    'volume_boundary_descriptor_Morovic2000',
    'gamut_boundary_descriptor_Morovic2000'
]


def area_boundary_descriptor_Morovic2000(xy, m=16, n=16):
    xy = as_float_array(xy)

    r_alpha = cartesian_to_polar(np.reshape(xy, [-1, 2]))
    r, alpha = tsplit(r_alpha)

    GBD = np.full([m, n, 2], np.nan)

    alpha_i = (alpha + np.pi) / (2 * np.pi) * n
    alpha_i = as_int_array(np.clip(np.floor(alpha_i), 0, n - 1))

    for i in np.arange(m):
        for j in np.arange(n):
            i_j = np.argwhere(alpha_i == j)

            if i_j.size == 0:
                continue

            GBD[i, j] = r_alpha[i_j[np.argmax(r[i_j])]]

    return GBD


def volume_boundary_descriptor_Morovic2000(Jab,
                                           E=np.array([50, 0, 0]),
                                           m=16,
                                           n=16):
    """
    Computes the *Gamut Boundary Descriptor (GDB)* for given :math:`Jab` array
    according to *Morovic and Luo (2000)* method.

    Parameters
    ----------
    Jab : array_like
        :math:`Jab` array to calculate the *GDB* of, where :math:`J` represents
        a quantity akin to *Lightness* and :math:`a` and :math:`b` are the
        related opponent colour dimensions.
    E : array_like, optional
        Estimated center of the gamut of :math:`Jab` array, it is subtracted
        to the :math:`Jab` array so that its values are converted meaningfully
        to spherical coordinates.
    m : int, optional
        Inclination or elevation sectors count spanning the [0, 180] degrees
        range.
    n : int, optional
        Azimuth sectors count spanning the [-180, 180] degrees range.

    Returns
    -------
    ndarray
        *GDB* for given :math:`Jab` array

    References
    ----------
    :cite:`Morovic2000`

    Examples
    --------
    """

    Jab = as_float_array(Jab)
    E = as_float_array(E)

    r_theta_alpha = cartesian_to_spherical(
        np.roll(np.reshape(Jab, [-1, 3]) - E, 2, -1))
    r, theta, alpha = tsplit(r_theta_alpha)

    GBD = np.full([m, n, 3], np.nan)

    theta_i = theta / np.pi * m
    theta_i = as_int_array(np.clip(np.floor(theta_i), 0, m - 1))

    alpha_i = (alpha + np.pi) / (2 * np.pi) * n
    alpha_i = as_int_array(np.clip(np.floor(alpha_i), 0, n - 1))

    for i in np.arange(m):
        for j in np.arange(n):
            i_j = np.intersect1d(
                np.argwhere(theta_i == i), np.argwhere(alpha_i == j))

            if i_j.size == 0:
                continue

            GBD[i, j] = r_theta_alpha[i_j[np.argmax(r[i_j])]]

    # Naive non-vectorised implementation kept for reference.
    # :math:`r_m` is used to keep track of the maximum :math:`r` value.
    # r_m = np.full([m, n, 1], np.nan)
    # for i, r_theta_alpha_i in enumerate(r_theta_alpha):
    #     p_i, a_i = theta_i[i], alpha_i[i]
    #     r_i_j = r_m[p_i, a_i]
    #
    #     if r[i] > r_i_j or np.isnan(r_i_j):
    #         GBD[p_i, a_i] = r_theta_alpha_i
    #         r_m[p_i, a_i] = r[i]

    return GBD


def gamut_boundary_descriptor_Morovic2000(
        Jab_ij,
        E=np.array([50, 0, 0]),
        m=16,
        n=16,
        close_callable=close_gamut_boundary_descriptor,
        close_callable_kwargs=None,
        fill_callable=fill_gamut_boundary_descriptor,
        fill_callable_kwargs=None):
    """
    Computes the *Gamut Boundary Descriptor (GDB)* for given :math:`Jab\lor ij`
    array according to *Morovic and Luo (2000)* method.

    Parameters
    ----------
    Jab_ij : array_like
        :math:`Jab\\lor ij` array to calculate the *GDB* of.
        :math:`Jab\\lor ij` array can be either 3-dimensional for a volume or
        a solid or 2-dimensional for an area.
    E : array_like, optional
        Estimated center of the gamut of :math:`Jab\\lor ij` array. It is
        subtracted to the :math:`Jab\\lor ij` array if 3-dimensional so that
        its values are converted meaningfully to spherical coordinates.
    m : int, optional
        Inclination or elevation sectors count spanning the [0, 180] degrees
        range.
    n : int, optional
        Azimuth sectors count spanning the [-180, 180] degrees range.
    close_callable : callable, optional
        Callable used to close the *GDB* poles.
    close_callable_kwargs : dict_like, optional
        Arguments to use when calling the the *GDB* poles closing function.
    fill_callable : callable, optional
        Callable used to fill the *GDB* holes, i.e. NaNs in the *GDB* matrix.
    fill_callable_kwargs
        Arguments to use when calling the the *GDB* holes filling function.

    Returns
    -------
    ndarray
        *GDB* for given :math:`Jab` array

    References
    ----------
    :cite:`Morovic2000`

    Examples
    --------
    # Computing the *GDB* for a perfect sphere.

    >>> from colour.algebra import spherical_to_cartesian
    >>> from colour.utilities import tstack
    >>> m = n = 9
    >>> theta = np.tile(np.radians(np.linspace(0, 180, m)), (m, 1))
    >>> theta = np.squeeze(np.transpose(theta).reshape(-1, 1))
    >>> phi = np.tile(
    ...     np.radians(np.linspace(-180, 180, m)) + np.radians(360 / 8 / 4), m)
    >>> rho = np.ones(m * m) * 50
    >>> rho_theta_phi = tstack([rho, theta, phi])
    >>> Jab = np.roll(spherical_to_cartesian(rho_theta_phi), 1, -1)
    >>> Jab += [50, 0, 0]
    >>> gamut_boundary_descriptor_Morovic2000(Jab, m=8, n=8)[3, ...]
    """

    Jab_ij = as_float_array(Jab_ij)

    if close_callable_kwargs is None:
        close_callable_kwargs = {}

    if fill_callable_kwargs is None:
        fill_callable_kwargs = {}

    if Jab_ij.shape[-1] == 3:
        GBD = volume_boundary_descriptor_Morovic2000(Jab_ij, E, m, n)
    else:
        GBD = area_boundary_descriptor_Morovic2000(Jab_ij, m, n)

    if close_callable is not None and Jab_ij.shape[-1] == 3:
        GBD = close_gamut_boundary_descriptor(
            GBD,
            **filter_kwargs(close_gamut_boundary_descriptor,
                            **close_callable_kwargs))

    if fill_callable is not None:
        GBD = fill_gamut_boundary_descriptor(
            GBD,
            **filter_kwargs(fill_gamut_boundary_descriptor,
                            **fill_callable_kwargs))

    return GBD
