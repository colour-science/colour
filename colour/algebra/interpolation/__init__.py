"""
Interpolation
=============

Provide classes and functions for interpolating variables in colour science
computations.

This module implements various interpolation methods for one-dimensional
functions and multi-dimensional table-based interpolation. These methods
support spectral data processing, colour transformations, and general
numerical interpolation tasks in colour science applications.

-   :class:`colour.KernelInterpolator`: 1-D function generic interpolation
    with arbitrary kernel.
-   :class:`colour.NearestNeighbourInterpolator`: 1-D function
    nearest-neighbour interpolation.
-   :class:`colour.LinearInterpolator`: 1-D function linear interpolation.
-   :class:`colour.SpragueInterpolator`: 1-D function fifth-order polynomial
    interpolation using *Sprague (1880)* method.
-   :class:`colour.CubicSplineInterpolator`: 1-D function cubic spline
    interpolation.
-   :class:`colour.PchipInterpolator`: 1-D function piecewise cube Hermite
    interpolation.
-   :class:`colour.NullInterpolator`: 1-D function null interpolation.
-   :func:`colour.lagrange_coefficients`: Compute *Lagrange Coefficients*.
-   :func:`colour.algebra.table_interpolation_trilinear`: Perform trilinear
    interpolation with table.
-   :func:`colour.algebra.table_interpolation_tetrahedral`: Perform
    tetrahedral interpolation with table.
-   :attr:`colour.TABLE_INTERPOLATION_METHODS`: Supported table interpolation
    methods.
-   :func:`colour.table_interpolation`: Perform interpolation with table using
    specified method.

References
----------
-   :cite:`Bourkeb` : Bourke, P. (n.d.). Trilinear Interpolation. Retrieved
    January 13, 2018, from http://paulbourke.net/miscellaneous/interpolation/
-   :cite:`Burger2009b` : Burger, W., & Burge, M. J. (2009). Principles of
    Digital Image Processing. Springer London. doi:10.1007/978-1-84800-195-4
-   :cite:`CIETC1-382005f` : CIE TC 1-38. (2005). 9.2.4 Method of
    interpolation for uniformly spaced independent variable. In CIE 167:2005
    Recommended Practice for Tabulating Spectral Data for Use in Colour
    Computations (pp. 1-27). ISBN:978-3-901906-41-1
-   :cite:`CIETC1-382005h` : CIE TC 1-38. (2005). Table V. Values of the
    c-coefficients of Equ.s 6 and 7. In CIE 167:2005 Recommended Practice for
    Tabulating Spectral Data for Use in Colour Computations (p. 19).
    ISBN:978-3-901906-41-1
-   :cite:`Fairman1985b` : Fairman, H. S. (1985). The calculation of weight
    factors for tristimulus integration. Color Research & Application, 10(4),
    199-203. doi:10.1002/col.5080100407
-   :cite:`Kirk2006` : Kirk, R. (2006). Truelight Software Library 2.0.
    Retrieved July 8, 2017, from
    https://www.filmlight.ltd.uk/pdf/whitepapers/FL-TL-TN-0057-SoftwareLib.pdf
-   :cite:`Westland2012h` : Westland, S., Ripamonti, C., & Cheung, V. (2012).
    Interpolation Methods. In Computational Colour Science Using MATLAB (2nd
    ed., pp. 29-37). ISBN:978-0-470-66569-5
-   :cite:`Wikipedia2003a` : Wikipedia. (2003). Lagrange polynomial -
    Definition. Retrieved January 20, 2016, from
    https://en.wikipedia.org/wiki/Lagrange_polynomial#Definition
-   :cite:`Wikipedia2005b` : Wikipedia. (2005). Lanczos resampling. Retrieved
    October 14, 2017, from https://en.wikipedia.org/wiki/Lanczos_resampling
"""

from __future__ import annotations

import sys
import typing
from functools import reduce
from unittest.mock import MagicMock

import numpy as np

from colour.utilities.requirements import is_scipy_installed
from colour.utilities.verbose import usage_warning

if not is_scipy_installed():  # pragma: no cover
    try:
        is_scipy_installed(raise_exception=True)
    except ImportError as error:
        usage_warning(str(error))

    mock = MagicMock()
    mock.__name__ = ""

    for module in (
        "scipy",
        "scipy.interpolate",
    ):
        sys.modules[module] = mock

from colour.constants import (
    DTYPE_INT_DEFAULT,
)

if typing.TYPE_CHECKING:
    from colour.hints import (
        ArrayLike,
        Literal,
    )

from colour.hints import NDArrayFloat, NDArrayReal, cast
from colour.utilities import (
    CanonicalMapping,
    array_namespace,
    validate_method,
    xp_as_float_array,
    xp_as_int_array,
    xp_astype,
    xp_reshape,
    xp_select,
)

from ._dispatch import (
    CubicSplineInterpolator,
    KernelInterpolator,
    LinearInterpolator,
    NearestNeighbourInterpolator,
    NullInterpolator,
    PchipInterpolator,
    SpragueInterpolator,
)
from .backends._kernels import (
    kernel_cardinal_spline,
    kernel_lanczos,
    kernel_linear,
    kernel_nearest_neighbour,
    kernel_sinc,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"


__all__ = [
    "kernel_nearest_neighbour",
    "kernel_linear",
    "kernel_sinc",
    "kernel_lanczos",
    "kernel_cardinal_spline",
    "KernelInterpolator",
    "NearestNeighbourInterpolator",
    "LinearInterpolator",
    "SpragueInterpolator",
    "CubicSplineInterpolator",
    "PchipInterpolator",
    "NullInterpolator",
    "lagrange_coefficients",
    "table_interpolation_trilinear",
    "table_interpolation_tetrahedral",
    "TABLE_INTERPOLATION_METHODS",
    "table_interpolation",
    "linear_interpolation_index_and_factor",
]


def lagrange_coefficients(r: float, n: int = 4) -> NDArrayFloat:
    """
    Compute *Lagrange coefficients* at specified point :math:`r` for
    polynomial interpolation of degree :math:`n`.

    Parameters
    ----------
    r
        Point at which to compute the *Lagrange coefficients*.
    n
        Degree of the polynomial interpolation. The number of coefficients
        returned will be :math:`n + 1`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of *Lagrange coefficients* computed at point :math:`r`.

    References
    ----------
    :cite:`Fairman1985b`, :cite:`Wikipedia2003a`

    Examples
    --------
    >>> lagrange_coefficients(0.1)
    array([ 0.8265,  0.2755, -0.1305,  0.0285])
    """

    r_i = np.arange(n)
    L_n = []
    for j in range(len(r_i)):
        basis = [(r - r_i[i]) / (r_i[j] - r_i[i]) for i in range(len(r_i)) if i != j]
        L_n.append(reduce(lambda x, y: x * y, basis))

    return np.asarray(L_n)


def table_interpolation_trilinear(V_xyz: ArrayLike, table: ArrayLike) -> NDArrayFloat:
    """
    Perform trilinear interpolation of the specified :math:`V_{xyz}` values using
    the specified interpolation table.

    Parameters
    ----------
    V_xyz
        :math:`V_{xyz}` values to interpolate.
    table
        4-Dimensional (NxNxNx3) interpolation table.

    Returns
    -------
    :class:`numpy.ndarray`
        Interpolated :math:`V_{xyz}` values.

    References
    ----------
    :cite:`Bourkeb`

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     os.path.dirname(__file__),
    ...     "..",
    ...     "..",
    ...     "io",
    ...     "luts",
    ...     "tests",
    ...     "resources",
    ...     "iridas_cube",
    ...     "Colour_Correct.cube",
    ... )
    >>> LUT = colour.read_LUT(path)
    >>> table = LUT.table
    >>> prng = np.random.RandomState(4)
    >>> V_xyz = colour.algebra.random_triplet_generator(3, random_state=prng)
    >>> print(V_xyz)  # doctest: +ELLIPSIS
    [[0.9670298... 0.7148159... 0.9762744...]
     [0.5472322... 0.6977288... 0.0062302...]
     [0.9726843... 0.2160895... 0.2529823...]]
    >>> table_interpolation_trilinear(V_xyz, table)  # doctest: +ELLIPSIS
    array([[1.0120664..., 0.7539146..., 1.0228540...],
           [0.5075794..., 0.6479459..., 0.1066404...],
           [1.0976519..., 0.1785998..., 0.2299897...]])
    """

    V_xyz = cast("NDArrayFloat", V_xyz)

    xp = array_namespace(V_xyz)

    original_shape = V_xyz.shape

    V_xyz = cast("NDArrayFloat", xp_reshape(xp.clip(V_xyz, 0, 1), (-1, 3), xp=xp))

    # Index computation
    table = xp_as_float_array(table, xp=xp, like=V_xyz)
    i_m = xp_as_int_array(table.shape[:-1], xp=xp, like=V_xyz) - 1
    V_xyz_s = V_xyz * i_m

    i_f = xp_astype(V_xyz_s, DTYPE_INT_DEFAULT, xp=xp)
    i_f = xp.clip(i_f, 0, i_m)
    i_c = xp.minimum(i_f + 1, i_m)

    # Relative coordinates (fractional part)
    frac = V_xyz_s - i_f

    # Extract indices for direct lookup
    fx, fy, fz = i_f[:, 0], i_f[:, 1], i_f[:, 2]
    cx, cy, cz = i_c[:, 0], i_c[:, 1], i_c[:, 2]

    # Extract fractional coordinates
    dx, dy, dz = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]
    dx1, dy1, dz1 = 1.0 - dx, 1.0 - dy, 1.0 - dz

    # Direct vertex lookups (8 corners of cube)
    v000 = table[fx, fy, fz]
    v001 = table[fx, fy, cz]
    v010 = table[fx, cy, fz]
    v011 = table[fx, cy, cz]
    v100 = table[cx, fy, fz]
    v101 = table[cx, fy, cz]
    v110 = table[cx, cy, fz]
    v111 = table[cx, cy, cz]

    # Trilinear interpolation (vectorized)
    result = (
        v000 * (dx1 * dy1 * dz1)
        + v001 * (dx1 * dy1 * dz)
        + v010 * (dx1 * dy * dz1)
        + v011 * (dx1 * dy * dz)
        + v100 * (dx * dy1 * dz1)
        + v101 * (dx * dy1 * dz)
        + v110 * (dx * dy * dz1)
        + v111 * (dx * dy * dz)
    )

    return xp_reshape(result, original_shape, xp=xp)


def table_interpolation_tetrahedral(V_xyz: ArrayLike, table: ArrayLike) -> NDArrayFloat:
    """
    Perform tetrahedral interpolation of the specified :math:`V_{xyz}` values
    using the specified 4-dimensional interpolation table.

    Parameters
    ----------
    V_xyz
        :math:`V_{xyz}` values to interpolate.
    table
        4-Dimensional (NxNxNx3) interpolation table.

    Returns
    -------
    :class:`numpy.ndarray`
        Interpolated :math:`V_{xyz}` values.

    References
    ----------
    :cite:`Kirk2006`

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     os.path.dirname(__file__),
    ...     "..",
    ...     "..",
    ...     "io",
    ...     "luts",
    ...     "tests",
    ...     "resources",
    ...     "iridas_cube",
    ...     "Colour_Correct.cube",
    ... )
    >>> LUT = colour.read_LUT(path)
    >>> table = LUT.table
    >>> prng = np.random.RandomState(4)
    >>> V_xyz = colour.algebra.random_triplet_generator(3, random_state=prng)
    >>> print(V_xyz)  # doctest: +ELLIPSIS
    [[0.9670298... 0.7148159... 0.9762744...]
     [0.5472322... 0.6977288... 0.0062302...]
     [0.9726843... 0.2160895... 0.2529823...]]
    >>> table_interpolation_tetrahedral(V_xyz, table)  # doctest: +ELLIPSIS
    array([[1.0196197..., 0.7674062..., 1.0311751...],
           [0.5105603..., 0.6466722..., 0.1077296...],
           [1.1178206..., 0.1762039..., 0.2209534...]])
    """

    V_xyz = cast("NDArrayFloat", V_xyz)

    xp = array_namespace(V_xyz)

    original_shape = V_xyz.shape

    V_xyz = cast("NDArrayFloat", xp_reshape(xp.clip(V_xyz, 0, 1), (-1, 3), xp=xp))

    # Index computation
    table = xp_as_float_array(table, xp=xp, like=V_xyz)
    i_m = xp_as_int_array(table.shape[:-1], xp=xp, like=V_xyz) - 1
    V_xyz_s = V_xyz * i_m

    i_f = xp_astype(V_xyz_s, DTYPE_INT_DEFAULT, xp=xp)
    i_f = xp.clip(i_f, 0, i_m)
    i_c = xp.minimum(i_f + 1, i_m)

    # Relative coordinates
    r = V_xyz_s - i_f
    x, y, z = r[:, 0], r[:, 1], r[:, 2]

    # Extract indices for direct lookup
    fx, fy, fz = i_f[:, 0], i_f[:, 1], i_f[:, 2]
    cx, cy, cz = i_c[:, 0], i_c[:, 1], i_c[:, 2]

    # Look up 8 corner vertices
    V000 = table[fx, fy, fz]
    V001 = table[fx, fy, cz]
    V010 = table[fx, cy, fz]
    V011 = table[fx, cy, cz]
    V100 = table[cx, fy, fz]
    V101 = table[cx, fy, cz]
    V110 = table[cx, cy, fz]
    V111 = table[cx, cy, cz]

    # Expand dimensions for broadcasting
    x = x[:, None]
    y = y[:, None]
    z = z[:, None]

    # Tetrahedral interpolation - select tetrahedron based on position
    xyz_o = xp_select(
        [
            xp.logical_and(x > y, y > z),
            xp.logical_and(x > z, z >= y),
            xp.logical_and(z >= x, x > y),
            xp.logical_and(y >= x, x > z),
            xp.logical_and(y >= z, z >= x),
            xp.logical_and(z > y, y >= x),
        ],
        [
            (1 - x) * V000 + (x - y) * V100 + (y - z) * V110 + z * V111,
            (1 - x) * V000 + (x - z) * V100 + (z - y) * V101 + y * V111,
            (1 - z) * V000 + (z - x) * V001 + (x - y) * V101 + y * V111,
            (1 - y) * V000 + (y - x) * V010 + (x - z) * V110 + z * V111,
            (1 - y) * V000 + (y - z) * V010 + (z - x) * V011 + x * V111,
            (1 - z) * V000 + (z - y) * V001 + (y - x) * V011 + x * V111,
        ],
        xp=xp,
    )

    return xp_reshape(xyz_o, original_shape, xp=xp)


TABLE_INTERPOLATION_METHODS = CanonicalMapping(
    {
        "Trilinear": table_interpolation_trilinear,
        "Tetrahedral": table_interpolation_tetrahedral,
    }
)
TABLE_INTERPOLATION_METHODS.__doc__ = """
Supported table interpolation methods.

References
----------
:cite:`Bourkeb`, :cite:`Kirk2006`
"""


def table_interpolation(
    V_xyz: ArrayLike,
    table: ArrayLike,
    method: Literal["Trilinear", "Tetrahedral"] | str = "Trilinear",
) -> NDArrayFloat:
    """
    Perform interpolation of the specified :math:`V_{xyz}` values using a
    4-dimensional interpolation table.

    Interpolate the input :math:`V_{xyz}` values through either trilinear
    or tetrahedral interpolation methods using the specified lookup table.

    Parameters
    ----------
    V_xyz
        :math:`V_{xyz}` values to interpolate, where each row represents
        a three-dimensional coordinate within the interpolation table's
        domain.
    table
        4-dimensional (NxNxNx3) interpolation table defining the mapping
        from input coordinates to output values.
    method
        Interpolation method to use. Either "Trilinear" for trilinear
        interpolation or "Tetrahedral" for tetrahedral interpolation.

    Returns
    -------
    :class:`numpy.ndarray`
        Interpolated :math:`V_{xyz}` values with the same shape as the
        input array.

    References
    ----------
    :cite:`Bourkeb`, :cite:`Kirk2006`

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     os.path.dirname(__file__),
    ...     "..",
    ...     "..",
    ...     "io",
    ...     "luts",
    ...     "tests",
    ...     "resources",
    ...     "iridas_cube",
    ...     "Colour_Correct.cube",
    ... )
    >>> LUT = colour.read_LUT(path)
    >>> table = LUT.table
    >>> prng = np.random.RandomState(4)
    >>> V_xyz = colour.algebra.random_triplet_generator(3, random_state=prng)
    >>> print(V_xyz)  # doctest: +ELLIPSIS
    [[0.9670298... 0.7148159... 0.9762744...]
     [0.5472322... 0.6977288... 0.0062302...]
     [0.9726843... 0.2160895... 0.2529823...]]
    >>> table_interpolation(V_xyz, table)  # doctest: +ELLIPSIS
    array([[1.0120664..., 0.7539146..., 1.0228540...],
           [0.5075794..., 0.6479459..., 0.1066404...],
           [1.0976519..., 0.1785998..., 0.2299897...]])
    >>> table_interpolation(V_xyz, table, method="Tetrahedral")
    ... # doctest: +ELLIPSIS
    array([[1.0196197..., 0.7674062..., 1.0311751...],
           [0.5105603..., 0.6466722..., 0.1077296...],
           [1.1178206..., 0.1762039..., 0.2209534...]])
    """

    method = validate_method(method, tuple(TABLE_INTERPOLATION_METHODS))

    return TABLE_INTERPOLATION_METHODS[method](V_xyz, table)


def linear_interpolation_index_and_factor(
    value: ArrayLike, break_points: ArrayLike
) -> tuple[NDArrayReal, NDArrayFloat]:
    """
    Compute the bin index and fractional position for piecewise linear
    interpolation of *value* within sorted *break_points*.

    For each element in *value*, the returned ``index`` identifies the
    interval ``[break_points[index], break_points[index + 1])`` that
    contains the value, and ``factor`` gives the normalised position
    within that interval (0 at the left edge, 1 at the right).

    Values outside the range of *break_points* are clamped.

    Parameters
    ----------
    value
        Query value(s), scalar or array.
    break_points
        Sorted array of break points defining the piecewise linear
        intervals.

    Returns
    -------
    :class:`tuple`
        Tuple of ``(index, factor)`` arrays with the same shape as
        *value*.

    Examples
    --------
    >>> break_points = np.array([0.0, 1.0, 2.0, 3.0])
    >>> linear_interpolation_index_and_factor(1.5, break_points)
    ... # doctest: +ELLIPSIS
    (array(1...), array(0.5))
    >>> linear_interpolation_index_and_factor(  # doctest: +ELLIPSIS
    ...     np.array([0.0, 0.5, 2.5, 3.0]), break_points
    ... )
    (array([0, 0, 2, 3]...), array([0. , 0.5, 0.5, 0. ]))
    """

    xp = array_namespace(value, break_points)

    value = xp_as_float_array(value, xp=xp, like=break_points)
    break_points = xp_as_float_array(break_points, xp=xp, like=value)

    clamped = xp.clip(value, break_points[0], break_points[-1])

    # Upper bound search starting from break_points[1].
    next_idx = (
        xp_reshape(
            xp.searchsorted(
                break_points[1:],
                xp_reshape(clamped, (-1,), xp=xp),
                side="right",
            ),
            clamped.shape,
            xp=xp,
        )
        + 1
    )

    at_end = next_idx >= len(break_points)
    index = xp.where(at_end, len(break_points) - 1, next_idx - 1)

    safe_next = xp.clip(next_idx, max=len(break_points) - 1)
    denominator = break_points[safe_next] - break_points[index]
    factor = xp.where(
        at_end | (denominator == 0),
        0.0,
        (clamped - break_points[index]) / xp.where(denominator == 0, 1.0, denominator),
    )

    return xp_astype(index, DTYPE_INT_DEFAULT, xp=xp), factor
