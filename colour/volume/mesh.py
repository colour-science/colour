"""
Mesh Volume Computation Helpers
===============================

Defines the helpers objects related to volume computations.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from colour.hints import ArrayLike, Floating, NDArray, Optional

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "is_within_mesh_volume",
]


def is_within_mesh_volume(
    points: ArrayLike, mesh: ArrayLike, tolerance: Optional[Floating] = None
) -> NDArray:
    """
    Return whether given points are within given mesh volume using Delaunay
    triangulation.

    Parameters
    ----------
    points
        Points to check if they are within ``mesh`` volume.
    mesh
        Points of the volume used to generate the Delaunay triangulation.
    tolerance
        Tolerance allowed in the inside-triangle check.

    Returns
    -------
    :class:`numpy.ndarray`
        Whether given points are within given mesh volume.

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

    triangulation = Delaunay(mesh)

    simplex = triangulation.find_simplex(points, tol=tolerance)
    simplex = np.where(simplex >= 0, True, False)

    return simplex
