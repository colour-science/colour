import numpy as np

from colour.characterisation.correction import (
    tps3d_parameters,
    apply_tps3d,
    colour_correction_TPS3D,
)


def test_tps3d_maps_control_points():
    # 24 pontos "tipo ColorChecker", mas aleatórios fixos.
    rng = np.random.default_rng(42)
    M_T = rng.random((24, 3))
    M_R = np.clip(M_T * 0.85 + 0.05, 0, 1)  # transformação simples

    W, A, ctrl = tps3d_parameters(M_T, M_R, smoothing=1e-10)
    mapped = apply_tps3d(M_T, W, A, ctrl, clip=False, chunk_size=1024)

    # Deve bater bem nos pontos de controle
    assert np.max(np.abs(mapped - M_R)) < 1e-6


def test_tps3d_identity_is_identity():
    rng = np.random.default_rng(123)
    M_T = rng.random((24, 3))
    W, A, ctrl = tps3d_parameters(M_T, M_T, smoothing=1e-12)

    img = rng.random((16, 17, 3))
    out = apply_tps3d(img, W, A, ctrl, clip=False, chunk_size=1024)

    assert np.max(np.abs(out - img)) < 1e-6


def test_colour_correction_tps3d_shape():
    rng = np.random.default_rng(7)
    M_T = rng.random((24, 3))
    M_R = rng.random((24, 3))
    img = rng.random((10, 11, 3))

    out = colour_correction_TPS3D(img, M_T, M_R, smoothing=1e-8, chunk_size=1024)
    assert out.shape == img.shape
