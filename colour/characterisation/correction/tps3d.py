"""
TPS-3D Colour Correction
========================

Define the *TPS-3D* (Thin-Plate Spline in RGB space) colour correction objects:

-   :func:`colour.characterisation.tps3d_kernel_bookstein`
-   :func:`colour.characterisation.tps3d_kernel_polyharmonic_3d`
-   :func:`colour.characterisation.pairwise_distances_euclidean`
-   :func:`colour.characterisation.tps3d_parameters`
-   :func:`colour.characterisation.apply_tps3d`
-   :func:`colour.characterisation.colour_correction_TPS3D`

References
----------
-   :cite:`Menesatti2012` : Menesatti, P., Angelini, C., Pallottino, F.,
    Antonucci, F., Aguzzi, J., & Costa, C. (2012). RGB Color Calibration for
    Quantitative Image Analysis: The “3D Thin-Plate Spline” Warping Approach.
    Sensors, 12(6), 7063-7079. doi:10.3390/s120607063
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Literal, NDArrayFloat

from colour.utilities import as_float_array, validate_method

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "tps3d_kernel_bookstein",
    "tps3d_kernel_polyharmonic_3d",
    "pairwise_distances_euclidean",
    "tps3d_parameters",
    "apply_tps3d",
    "colour_correction_TPS3D",
]


def tps3d_kernel_bookstein(r: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute the Bookstein TPS kernel: phi(r) = r^2 log(r^2).

    This kernel is commonly used in Thin-Plate Spline interpolation and is
    equivalent to 2 * r^2 * log(r).

    Parameters
    ----------
    r
        Euclidean distances in the (R, G, B) space.
    eps
        Small epsilon value for numerical stability.

    Returns
    -------
    :class:`numpy.ndarray`
        Kernel values.

    Notes
    -----
    -   We use r^2 log(r^2) for numerical stability and to avoid extra factors.
    -   r is Euclidean distance in the (R, G, B) space.

    References
    ----------
    Thin plate spline radial basis kernel: phi(r) = r^2 log r. See e.g. Wikipedia.

    Examples
    --------
    >>> r = np.array([0.0, 0.5, 1.0])
    >>> tps3d_kernel_bookstein(r)  # doctest: +ELLIPSIS
    array([-2.7631021...e-11, -3.4657359...e-01,  0.0000000...e+00])
    """

    r2 = np.maximum(r * r, eps)

    return r2 * np.log(r2)


def tps3d_kernel_polyharmonic_3d(r: np.ndarray) -> np.ndarray:
    """
    Compute the polyharmonic spline kernel for 3D with m=2.

    This kernel is proportional to r and represents the theoretical 3D
    biharmonic fundamental solution form used in polyharmonic splines.

    Parameters
    ----------
    r
        Euclidean distances.

    Returns
    -------
    :class:`numpy.ndarray`
        Kernel values (equal to r).

    Notes
    -----
    -   This is the *theoretical* 3D biharmonic fundamental solution form used
        in polyharmonic splines.
    -   Some "TPS-3D" implementations still use the 2D thin-plate kernel
        (Bookstein) but compute distances in 3D.
    -   Default remains Bookstein to match common TPS usage.

    Examples
    --------
    >>> r = np.array([0.0, 0.5, 1.0])
    >>> tps3d_kernel_polyharmonic_3d(r)
    array([0. , 0.5, 1. ])
    """

    return r


def pairwise_distances_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between two arrays.

    Parameters
    ----------
    A
        First array with shape (M, 3).
    B
        Second array with shape (N, 3).

    Returns
    -------
    :class:`numpy.ndarray`
        Pairwise distance matrix with shape (M, N).

    Examples
    --------
    >>> A = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> B = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> pairwise_distances_euclidean(A, B)  # doctest: +ELLIPSIS
    array([[0.        , 1.        , 1.        ],
           [1.        , 1.41421356, 1.41421356]])
    """

    # (M,1,3) - (1,N,3) -> (M,N,3)
    D = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(D * D, axis=-1))


def tps3d_parameters(
    source_points: ArrayLike,
    destination_points: ArrayLike,
    *,
    smoothing: float = 0.0,
    kernel: Literal["Bookstein", "Polyharmonic 3D"] | str = "Bookstein",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit TPS-3D parameters that warp RGB source_points -> destination_points.

    Parameters
    ----------
    source_points
        (N,3) measured RGB points (e.g., detected ColorChecker swatches).
    destination_points
        (N,3) reference RGB points (e.g., ideal ColorChecker swatches).
    smoothing
        Non-negative regularization added to diag(K) to improve conditioning
        (useful with noise / near-collinear points).
    kernel
        Kernel choice:
        - "Bookstein": phi(r) = r^2 log(r^2)  (classic TPS kernel)
        - "Polyharmonic 3D": phi(r) = r       (3D polyharmonic option)

    Returns
    -------
    :class:`tuple`
        Tuple of (W, A, ctrl):

        -   W : (N,3) non-linear weights
        -   A : (4,3) affine coefficients for [1, R, G, B]
        -   ctrl : (N,3) control points (source_points), returned for reuse

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> source = rng.random((10, 3))
    >>> dest = source * 0.9 + 0.05
    >>> W, A, ctrl = tps3d_parameters(source, dest, smoothing=1e-10)
    >>> W.shape
    (10, 3)
    >>> A.shape
    (4, 3)
    """

    ctrl = as_float_array(source_points)
    dest = as_float_array(destination_points)

    if ctrl.ndim != 2 or ctrl.shape[1] != 3:
        message = '"source_points" must be an (N, 3) array!'
        raise ValueError(message)

    if dest.shape != ctrl.shape:
        message = '"destination_points" must have the same shape as "source_points"!'
        raise ValueError(message)

    N = ctrl.shape[0]
    if N < 4:
        message = "TPS-3D requires at least 4 control points!"
        raise ValueError(message)

    kernel = validate_method(kernel, ("Bookstein", "Polyharmonic 3D"))

    # P: (N,4) -> [1, R, G, B]
    P = np.hstack([np.ones((N, 1)), ctrl])

    # K: (N,N) from pairwise distances
    r = pairwise_distances_euclidean(ctrl, ctrl)
    if kernel == "Bookstein":
        K = tps3d_kernel_bookstein(r)
        np.fill_diagonal(K, 0.0)
    else:
        K = tps3d_kernel_polyharmonic_3d(r)
        np.fill_diagonal(K, 0.0)

    if smoothing < 0:
        message = '"smoothing" must be >= 0!'
        raise ValueError(message)

    if smoothing > 0:
        K = K + np.eye(N) * smoothing

    Z = np.zeros((4, 4))
    L = np.block([[K, P], [P.T, Z]])

    V = np.vstack([dest, np.zeros((4, 3))])

    # Solve L * params = V
    # Use solve when possible; fallback to lstsq for robustness.
    try:
        params = np.linalg.solve(L, V)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(L, V, rcond=None)[0]

    W = params[:N, :]
    A = params[N:, :]

    return W, A, ctrl


def apply_tps3d(
    RGB: ArrayLike,
    W: np.ndarray,
    A: np.ndarray,
    ctrl: np.ndarray,
    *,
    kernel: Literal["Bookstein", "Polyharmonic 3D"] | str = "Bookstein",
    clip: bool = True,
    chunk_size: int = 250_000,
) -> NDArrayFloat:
    """
    Apply pre-fitted TPS-3D to an arbitrary RGB array (... , 3).

    Parameters
    ----------
    RGB
        RGB array to warp. Can be (M,3) or (H,W,3) etc.
    W
        TPS non-linear weights from :func:`tps3d_parameters`.
    A
        TPS affine coefficients from :func:`tps3d_parameters`.
    ctrl
        Control points from :func:`tps3d_parameters`.
    kernel
        Same kernel used during fitting.
    clip
        Whether to clip output to [0, 1].
    chunk_size
        Process pixels in chunks to avoid huge (M,N) temporary arrays for images.

    Returns
    -------
    :class:`numpy.ndarray`
        Warped RGB array with same shape as input.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> source = rng.random((10, 3))
    >>> dest = source * 0.9 + 0.05
    >>> W, A, ctrl = tps3d_parameters(source, dest, smoothing=1e-10)
    >>> RGB = rng.random((5, 5, 3))
    >>> result = apply_tps3d(RGB, W, A, ctrl)
    >>> result.shape
    (5, 5, 3)
    """

    kernel = validate_method(kernel, ("Bookstein", "Polyharmonic 3D"))

    RGB = as_float_array(RGB)
    shape = RGB.shape

    if shape[-1] != 3:
        message = '"RGB" last dimension must be 3!'
        raise ValueError(message)

    pixels = RGB.reshape((-1, 3))
    M = pixels.shape[0]

    out = np.empty_like(pixels)

    # Precompute affine input [1, R, G, B]
    # Do it chunked to keep memory stable.
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        X = pixels[start:end]

        P_all = np.hstack([np.ones((X.shape[0], 1)), X])  # (m,4)
        r = pairwise_distances_euclidean(X, ctrl)  # (m,N)

        if kernel == "Bookstein":
            U = tps3d_kernel_bookstein(r)
        else:
            U = tps3d_kernel_polyharmonic_3d(r)

        out[start:end] = U @ W + P_all @ A

    if clip:
        out = np.clip(out, 0.0, 1.0)

    return out.reshape(shape)


def colour_correction_TPS3D(
    RGB: ArrayLike,
    M_T: ArrayLike,
    M_R: ArrayLike,
    *,
    smoothing: float = 0.0,
    kernel: Literal["Bookstein", "Polyharmonic 3D"] | str = "Bookstein",
    clip: bool = True,
    chunk_size: int = 250_000,
) -> NDArrayFloat:
    """
    Perform colour correction using TPS-3D warping in RGB space.

    Parameters
    ----------
    RGB
        RGB array to colour correct (... , 3).
    M_T
        Source control points (N,3): measured RGBs (e.g., extracted swatches).
    M_R
        Destination control points (N,3): reference RGBs (e.g., ideal swatches).
    smoothing
        Regularization added to diag(K) for stability.
    kernel
        "Bookstein" (classic TPS) or "Polyharmonic 3D".
    clip
        Clip output to [0, 1].
    chunk_size
        Chunk size for large images.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour corrected RGB array.

    References
    ----------
    :cite:`Menesatti2012`

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> M_T = rng.random((24, 3))
    >>> M_R = M_T * 0.9 + 0.05
    >>> RGB = rng.random((10, 10, 3))
    >>> result = colour_correction_TPS3D(RGB, M_T, M_R, smoothing=1e-10)
    >>> result.shape
    (10, 10, 3)
    """

    W, A, ctrl = tps3d_parameters(M_T, M_R, smoothing=smoothing, kernel=kernel)

    return apply_tps3d(RGB, W, A, ctrl, kernel=kernel, clip=clip, chunk_size=chunk_size)
