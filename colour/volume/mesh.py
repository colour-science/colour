"""
Mesh Volume Computation Helpers
===============================

Define helper objects for computing volumes of three-dimensional meshes
and polyhedra using Delaunay triangulation and related computational
geometry methods.
"""

from __future__ import annotations

import typing

from colour.constants import EPSILON
from colour.utilities import (
    CACHE_REGISTRY,
    array_namespace,
    as_ndarray,
    int_digest,
    is_caching_enabled,
    required,
    xp_as_array,
)

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

_CACHE_DELAUNAY: dict = CACHE_REGISTRY.register_cache(f"{__name__}._CACHE_DELAUNAY")


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
    >>> import numpy as np
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
    array(True)
    >>> a = np.array([[0.0005, 0.0031, 0.0010], [0.3205, 0.4131, 0.5100]])
    >>> is_within_mesh_volume(a, mesh)
    array([ True, False])
    """

    from scipy.spatial import Delaunay  # noqa: PLC0415

    xp = array_namespace(points)

    mesh_np = as_ndarray(mesh)
    # ``Delaunay`` triangulation is a *scipy* host-only operation so the
    # mesh is materialised to *NumPy* and content-hashed for the cache;
    # ``id(mesh)`` would thrash across array copies that share content
    # and could be reused after garbage collection.
    cache_key = (int_digest(mesh_np.tobytes()), mesh_np.shape, mesh_np.dtype.str)
    triangulation = _CACHE_DELAUNAY.get(cache_key) if is_caching_enabled() else None

    if triangulation is None:
        triangulation = Delaunay(mesh_np)

        if is_caching_enabled():
            _CACHE_DELAUNAY[cache_key] = triangulation

    simplex = triangulation.find_simplex(as_ndarray(points), tol=tolerance)

    return xp_as_array(simplex >= 0, xp=xp, like=points)
