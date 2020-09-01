# -*- coding: utf-8 -*-
"""
Mesh Volume Computation Helpers
===============================

Defines helpers objects related to volume computations.
"""

from __future__ import division, unicode_literals

import colour.ndarray as np
from scipy.spatial import Delaunay

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['is_within_mesh_volume']


def is_within_mesh_volume(points, mesh, tolerance=None):
    """
    Returns if given points are within given mesh volume using Delaunay
    triangulation.

    Parameters
    ----------
    points : array_like
        Points to check if they are within ``mesh`` volume.
    mesh : array_like
        Points of the volume used to generate the Delaunay triangulation.
    tolerance : numeric, optional
        Tolerance allowed in the inside-triangle check.

    Returns
    -------
    bool
        Is within mesh volume.

    Examples
    --------
    >>> mesh = np.array(
    ...     [[-1.0, -1.0, 1.0],
    ...       [1.0, -1.0, 1.0],
    ...       [1.0, -1.0, -1.0],
    ...       [-1.0, -1.0, -1.0],
    ...       [0.0, 1.0, 0.0]]
    ... )
    >>> is_within_mesh_volume(np.array([0.0005, 0.0031, 0.0010]), mesh)
    array(True, dtype=bool)
    >>> a = np.array([[0.0005, 0.0031, 0.0010],
    ...               [0.3205, 0.4131, 0.5100]])
    >>> is_within_mesh_volume(a, mesh)
    array([ True, False], dtype=bool)
    """
    cupy = False
    if np.__name__ == 'cupy':
        cupy = True
        points = np.asnumpy(points)
        mesh = np.asnumpy(mesh)
        np.set_ndimensional_array_backend('numpy')

    triangulation = Delaunay(mesh)

    simplex = triangulation.find_simplex(points, tol=tolerance)

    if cupy is True:
        np.set_ndimensional_array_backend('cupy')
        simplex = np.array(simplex)

    simplex = np.where(simplex >= 0, True, False)

    return simplex
