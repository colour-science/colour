"""
Unit tests for the TPS-3D colour correction method.
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from colour.characterisation.correction import (
    apply_tps3d,
    colour_correction_TPS3D,
    tps3d_parameters,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import as_ndarray, xp_as_array


def test_tps3d_maps_control_points(xp: ModuleType) -> None:
    """Control points should map exactly to their targets."""
    rng = np.random.default_rng(42)
    M_T_np = rng.random((24, 3))
    M_R_np = np.clip(M_T_np * 0.85 + 0.05, 0, 1)

    M_T = xp_as_array(M_T_np, xp=xp)
    M_R = xp_as_array(M_R_np, xp=xp)

    W, A, ctrl = tps3d_parameters(M_T, M_R, smoothing=1e-10)
    mapped = apply_tps3d(M_T, W, A, ctrl, clip=False, chunk_size=1024)

    assert np.max(np.abs(as_ndarray(mapped) - M_R_np)) < TOLERANCE_ABSOLUTE_TESTS


def test_tps3d_identity_is_identity(xp: ModuleType) -> None:
    """Identity mapping should leave data unchanged."""
    rng = np.random.default_rng(123)
    M_T = xp_as_array(rng.random((24, 3)), xp=xp)
    W, A, ctrl = tps3d_parameters(M_T, M_T, smoothing=1e-12)

    img_np = rng.random((16, 17, 3))
    img = xp_as_array(img_np, xp=xp)
    out = apply_tps3d(img, W, A, ctrl, clip=False, chunk_size=1024)

    assert np.max(np.abs(as_ndarray(out) - img_np)) < TOLERANCE_ABSOLUTE_TESTS


def test_colour_correction_tps3d_shape(xp: ModuleType) -> None:
    """Colour correction should preserve the input shape."""
    rng = np.random.default_rng(7)
    M_T = xp_as_array(rng.random((24, 3)), xp=xp)
    M_R = xp_as_array(rng.random((24, 3)), xp=xp)
    img = xp_as_array(rng.random((10, 11, 3)), xp=xp)

    out = colour_correction_TPS3D(img, M_T, M_R, smoothing=1e-8, chunk_size=1024)
    assert as_ndarray(out).shape == (10, 11, 3)
