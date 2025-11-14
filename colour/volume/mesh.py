"""
Mesh Volume Computation Helpers
===============================

Define helper objects for computing volumes of three-dimensional meshes
and polyhedra using Delaunay triangulation and related computational
geometry methods.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import EPSILON
from colour.utilities import as_float_array, required

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "is_within_mesh_volume",
]


@required("SciPy")
def is_within_mesh_volume(
    points: ArrayLike, mesh: ArrayLike, tolerance: float = 100 * EPSILON
) -> NDArrayFloat:
    """
    Determine whether the specified points are within the volume defined by a mesh
    using Delaunay triangulation.

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
        Boolean array indicating whether specified points are within
        specified mesh volume.

    Examples
    --------
    >>> mesh = np.array(
    ...     [
    ...         [-1.0, -1.0, 1.0],
    ...         [1.0, -1.0, 1.0],
    ...         [1.0, -1.0, -1.0],
    ...         [-1.0, -1.0, -1.0],
    ...         [0.0, 1.0, 0.0],
    ...     ]
    ... )
    >>> is_within_mesh_volume(np.array([0.0005, 0.0031, 0.0010]), mesh)
    array(True, dtype=bool)
    >>> a = np.array([[0.0005, 0.0031, 0.0010], [0.3205, 0.4131, 0.5100]])
    >>> is_within_mesh_volume(a, mesh)
    array([ True, False], dtype=bool)
    """

    from scipy.spatial import Delaunay  # noqa: PLC0415

    triangulation = Delaunay(as_float_array(mesh))

    simplex = triangulation.find_simplex(as_float_array(points), tol=tolerance)

    return np.where(simplex >= 0, True, False)
