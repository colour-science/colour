"""
Define the unit tests for the
:mod:`colour.characterisation.correction.tps3d` module.
"""

from __future__ import annotations

import numpy as np

from colour.characterisation.correction import (
    apply_tps3d,
    colour_correction_TPS3D,
)
from colour.characterisation.correction.tps3d import (
    pairwise_distances_euclidean,
    tps3d_kernel_bookstein,
    tps3d_kernel_polyharmonic_3d,
    tps3d_parameters,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestTps3dKernelBookstein",
    "TestTps3dKernelPolyharmonic3d",
    "TestPairwiseDistancesEuclidean",
    "TestTps3dParameters",
    "TestApplyTps3d",
    "TestColourCorrectionTPS3D",
]


class TestTps3dKernelBookstein:
    """
    Define :func:`colour.characterisation.correction.tps3d.\
tps3d_kernel_bookstein` definition unit tests methods.
    """

    def test_tps3d_kernel_bookstein(self) -> None:
        """
        Test :func:`colour.characterisation.correction.tps3d.\
tps3d_kernel_bookstein` definition.
        """

        r = np.array([0.0, 0.5, 1.0, 2.0])
        result = tps3d_kernel_bookstein(r)

        # r^2 * log(r^2) for r=0 gives -inf*0 -> we use eps
        # r=0.5: 0.25 * log(0.25) = 0.25 * (-1.386) = -0.3466
        # r=1.0: 1 * log(1) = 0
        # r=2.0: 4 * log(4) = 4 * 1.386 = 5.545
        assert result.shape == (4,)
        np.testing.assert_allclose(result[2], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[3], 4.0 * np.log(4.0), atol=1e-10)


class TestTps3dKernelPolyharmonic3d:
    """
    Define :func:`colour.characterisation.correction.tps3d.\
tps3d_kernel_polyharmonic_3d` definition unit tests methods.
    """

    def test_tps3d_kernel_polyharmonic_3d(self) -> None:
        """
        Test :func:`colour.characterisation.correction.tps3d.\
tps3d_kernel_polyharmonic_3d` definition.
        """

        r = np.array([0.0, 0.5, 1.0, 2.0])
        result = tps3d_kernel_polyharmonic_3d(r)

        # phi(r) = r
        np.testing.assert_array_equal(result, r)


class TestPairwiseDistancesEuclidean:
    """
    Define :func:`colour.characterisation.correction.tps3d.\
pairwise_distances_euclidean` definition unit tests methods.
    """

    def test_pairwise_distances_euclidean(self) -> None:
        """
        Test :func:`colour.characterisation.correction.tps3d.\
pairwise_distances_euclidean` definition.
        """

        A = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        B = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        result = pairwise_distances_euclidean(A, B)

        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[0, 1], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[0, 2], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[1, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[1, 1], np.sqrt(2.0), atol=1e-10)
        np.testing.assert_allclose(result[1, 2], np.sqrt(2.0), atol=1e-10)


class TestTps3dParameters:
    """
    Define :func:`colour.characterisation.correction.tps3d.\
tps3d_parameters` definition unit tests methods.
    """

    def test_tps3d_parameters(self) -> None:
        """
        Test :func:`colour.characterisation.correction.tps3d.\
tps3d_parameters` definition.
        """

        rng = np.random.default_rng(42)
        source = rng.random((10, 3))
        dest = source * 0.9 + 0.05

        W, A, ctrl = tps3d_parameters(source, dest, smoothing=1e-10)

        assert W.shape == (10, 3)
        assert A.shape == (4, 3)
        assert ctrl.shape == (10, 3)
        np.testing.assert_array_equal(ctrl, source)


class TestApplyTps3d:
    """
    Define :func:`colour.characterisation.correction.tps3d.\
apply_tps3d` definition unit tests methods.
    """

    def test_apply_tps3d(self) -> None:
        """
        Test :func:`colour.characterisation.correction.tps3d.\
apply_tps3d` definition.
        """

        rng = np.random.default_rng(42)
        source = rng.random((10, 3))
        dest = source * 0.9 + 0.05

        W, A, ctrl = tps3d_parameters(source, dest, smoothing=1e-10)

        RGB = rng.random((5, 5, 3))
        result = apply_tps3d(RGB, W, A, ctrl)

        assert result.shape == (5, 5, 3)

    def test_apply_tps3d_control_points(self) -> None:
        """
        Test :func:`colour.characterisation.correction.tps3d.\
apply_tps3d` definition control point mapping.
        """

        rng = np.random.default_rng(42)
        M_T = rng.random((24, 3))
        M_R = np.clip(M_T * 0.85 + 0.05, 0, 1)

        W, A, ctrl = tps3d_parameters(M_T, M_R, smoothing=1e-10)
        mapped = apply_tps3d(M_T, W, A, ctrl, clip=False, chunk_size=1024)

        # Control points should map exactly to their targets
        assert np.max(np.abs(mapped - M_R)) < 1e-6

    def test_apply_tps3d_identity(self) -> None:
        """
        Test :func:`colour.characterisation.correction.tps3d.\
apply_tps3d` definition identity mapping.
        """

        rng = np.random.default_rng(123)
        M_T = rng.random((24, 3))
        W, A, ctrl = tps3d_parameters(M_T, M_T, smoothing=1e-12)

        img = rng.random((16, 17, 3))
        out = apply_tps3d(img, W, A, ctrl, clip=False, chunk_size=1024)

        # Identity mapping should leave data unchanged
        assert np.max(np.abs(out - img)) < 1e-6


class TestColourCorrectionTPS3D:
    """
    Define :func:`colour.characterisation.correction.tps3d.\
colour_correction_TPS3D` definition unit tests methods.
    """

    def test_colour_correction_TPS3D(self) -> None:
        """
        Test :func:`colour.characterisation.correction.tps3d.\
colour_correction_TPS3D` definition.
        """

        rng = np.random.default_rng(7)
        M_T = rng.random((24, 3))
        M_R = rng.random((24, 3))
        img = rng.random((10, 11, 3))

        out = colour_correction_TPS3D(img, M_T, M_R, smoothing=1e-8, chunk_size=1024)

        # Should preserve the input shape
        assert out.shape == img.shape

    def test_colour_correction_TPS3D_polyharmonic(self) -> None:
        """
        Test :func:`colour.characterisation.correction.tps3d.\
colour_correction_TPS3D` definition with polyharmonic kernel.
        """

        rng = np.random.default_rng(42)
        M_T = rng.random((24, 3))
        M_R = M_T * 0.9 + 0.05
        img = rng.random((8, 8, 3))

        out = colour_correction_TPS3D(
            img, M_T, M_R, smoothing=1e-8, kernel="Polyharmonic 3D", chunk_size=1024
        )

        assert out.shape == img.shape
