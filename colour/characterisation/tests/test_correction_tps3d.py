"""
Unit tests for the TPS-3D colour correction method.
"""

from __future__ import annotations

import typing
from contextlib import nullcontext

import numpy as np

if typing.TYPE_CHECKING:
    import pytest
    from colour.hints import ModuleType

from colour.characterisation.correction import (
    apply_tps3d,
    colour_correction_TPS3D,
    tps3d_parameters,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import as_ndarray, xp_as_array, xp_assert_close


def test_tps3d_maps_control_points(
    xp: ModuleType, request: pytest.FixtureRequest
) -> None:
    """Control points should map exactly to their targets."""
    rng = np.random.default_rng(42)
    M_T_np = rng.random((24, 3))
    M_R_np = np.clip(M_T_np * 0.85 + 0.05, 0, 1)

    M_T = xp_as_array(M_T_np, xp=xp)
    M_R = xp_as_array(M_R_np, xp=xp)

    W, A, ctrl = tps3d_parameters(M_T, M_R, smoothing=1e-10)
    mapped = apply_tps3d(M_T, W, A, ctrl, clip=False, chunk_size=1024)

    assert np.max(np.abs(as_ndarray(mapped) - M_R_np)) < TOLERANCE_ABSOLUTE_TESTS

    parameters = tps3d_parameters(M_T, M_R, smoothing=1e-10, kernel="Polyharmonic 3D")
    xp_assert_close(
        apply_tps3d(M_T, *parameters, kernel="Polyharmonic 3D", clip=False),
        M_R,
        atol=TOLERANCE_ABSOLUTE_TESTS,
    )

    for mixed_source, mixed_destination in ((M_T_np, M_R), (M_T, M_R_np)):
        with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
            mixed_W, mixed_A, mixed_ctrl = tps3d_parameters(
                mixed_source, mixed_destination, smoothing=1e-10
            )
            mapped = apply_tps3d(M_T_np, mixed_W, mixed_A, mixed_ctrl, clip=False)
        assert np.max(np.abs(as_ndarray(mapped) - M_R_np)) < TOLERANCE_ABSOLUTE_TESTS
        if hasattr(M_T, "device"):
            assert mapped.device == M_T.device

    # Rank-deficient controls retain the least-squares fallback on every backend.
    duplicate_source = xp_as_array(np.zeros((4, 3)), xp=xp)
    constant_destination = xp_as_array(np.full((4, 3), 0.2), xp=xp)
    parameters = tps3d_parameters(duplicate_source, constant_destination)
    xp_assert_close(
        apply_tps3d([[0.1, 0.2, 0.3]], *parameters, clip=False),
        [[0.2, 0.2, 0.2]],
        atol=TOLERANCE_ABSOLUTE_TESTS,
    )

    if xp.__name__ != "numpy":
        autodiff = request.getfixturevalue("autodiff")
        source = np.random.default_rng(4).random((8, 3))
        destination = np.random.default_rng(5).random((8, 3))
        for controls, vary_source in ((source, True), (destination, False)):
            autodiff(
                lambda values, vary_source=vary_source: apply_tps3d(
                    [[0.1, 0.2, 0.3]],
                    *tps3d_parameters(
                        values if vary_source else source,
                        destination if vary_source else values,
                        smoothing=0.001,
                    ),
                    clip=False,
                ),
                controls,
            )


def test_tps3d_identity_is_identity(xp: ModuleType) -> None:
    """Identity mapping should leave data unchanged."""
    rng = np.random.default_rng(123)
    M_T = xp_as_array(rng.random((24, 3)), xp=xp)
    W, A, ctrl = tps3d_parameters(M_T, M_T, smoothing=1e-12)

    empty = xp_as_array(np.empty((0, 3)), xp=xp)
    assert apply_tps3d(empty, W, A, ctrl).shape == (0, 3)

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
