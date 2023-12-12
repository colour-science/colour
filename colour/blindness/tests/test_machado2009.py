# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.blindness.machado2009` module."""

import unittest

import numpy as np

from colour.blindness import (
    CVD_MATRICES_MACHADO2010,
    matrix_anomalous_trichromacy_Machado2009,
    matrix_cvd_Machado2009,
    msds_cmfs_anomalous_trichromacy_Machado2009,
)
from colour.characterisation import MSDS_DISPLAY_PRIMARIES
from colour.colorimetry import MSDS_CMFS_LMS
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestMsdsCmfsAnomalousTrichromacyMachado2009",
    "TestMatrixAnomalousTrichromacyMachado2009",
    "TestMatrixCvdMachado2009",
]


class TestMsdsCmfsAnomalousTrichromacyMachado2009(unittest.TestCase):
    """
    Define :func:`colour.blindness.machado2009.\
msds_cmfs_anomalous_trichromacy_Machado2009` definition unit tests methods.
    """

    def test_msds_cmfs_anomalous_trichromacy_Machado2009(self):
        """
        Test :func:`colour.blindness.machado2009.\
msds_cmfs_anomalous_trichromacy_Machado2009` definition.
        """

        cmfs = MSDS_CMFS_LMS.get("Smith & Pokorny 1975 Normal Trichromats")
        np.testing.assert_allclose(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                np.array(
                    [0, 0, 0],
                ),
            )[450],
            cmfs[450],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                np.array(
                    [1, 0, 0],
                ),
            )[450],
            np.array([0.03631700, 0.06350000, 0.91000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                np.array(
                    [0, 1, 0],
                ),
            )[450],
            np.array([0.03430000, 0.06178404, 0.91000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                np.array(
                    [0, 0, 1],
                ),
            )[450],
            np.array([0.03430000, 0.06350000, 0.92270240]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                np.array(
                    [10, 0, 0],
                ),
            )[450],
            np.array([0.05447001, 0.06350000, 0.91000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                np.array(
                    [0, 10, 0],
                ),
            )[450],
            np.array([0.03430000, 0.04634036, 0.91000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                np.array(
                    [0, 0, 10],
                ),
            )[450],
            np.array([0.03430000, 0.06350000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestMatrixAnomalousTrichromacyMachado2009(unittest.TestCase):
    """
    Define :func:`colour.blindness.machado2009.\
matrix_anomalous_trichromacy_Machado2009` definition unit tests methods.
    """

    def test_matrix_anomalous_trichromacy_Machado2009(self):
        """
        Test :func:`colour.blindness.machado2009.\
matrix_anomalous_trichromacy_Machado2009` definition.
        """

        cmfs = MSDS_CMFS_LMS.get("Smith & Pokorny 1975 Normal Trichromats")
        primaries = MSDS_DISPLAY_PRIMARIES["Typical CRT Brainard 1997"]
        np.testing.assert_allclose(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, np.array([0, 0, 0])
            ),
            np.identity(3),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, np.array([2, 0, 0])
            ),
            CVD_MATRICES_MACHADO2010.get("Protanomaly").get(0.1),
            atol=0.0001,
        )

        np.testing.assert_allclose(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, np.array([10, 0, 0])
            ),
            CVD_MATRICES_MACHADO2010.get("Protanomaly").get(0.5),
            atol=0.0001,
        )

        np.testing.assert_allclose(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, np.array([20, 0, 0])
            ),
            CVD_MATRICES_MACHADO2010.get("Protanomaly").get(1.0),
            atol=0.0001,
        )

        np.testing.assert_allclose(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, np.array([0, 2, 0])
            ),
            CVD_MATRICES_MACHADO2010.get("Deuteranomaly").get(0.1),
            atol=0.0001,
        )

        np.testing.assert_allclose(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, np.array([0, 10, 0])
            ),
            CVD_MATRICES_MACHADO2010.get("Deuteranomaly").get(0.5),
            atol=0.0001,
        )

        np.testing.assert_allclose(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, np.array([0, 20, 0])
            ),
            CVD_MATRICES_MACHADO2010.get("Deuteranomaly").get(1.0),
            atol=0.0001,
        )

        np.testing.assert_allclose(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, np.array([0, 0, 5.00056688094503])
            ),
            CVD_MATRICES_MACHADO2010.get("Tritanomaly").get(0.1),
            atol=0.0001,
        )

        np.testing.assert_allclose(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, np.array([0, 0, 29.002939088780934])
            ),
            CVD_MATRICES_MACHADO2010.get("Tritanomaly").get(0.5),
            atol=0.0001,
        )

        np.testing.assert_allclose(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, np.array([0, 0, 59.00590434857581])
            ),
            CVD_MATRICES_MACHADO2010.get("Tritanomaly").get(1.0),
            atol=0.001,
        )


class TestMatrixCvdMachado2009(unittest.TestCase):
    """
    Define :func:`colour.blindness.machado2009.matrix_cvd_Machado2009`
    definition unit tests methods.
    """

    def test_matrix_cvd_Machado2009(self):
        """
        Test :func:`colour.blindness.machado2009.matrix_cvd_Machado2009`
        definition.
        """

        np.testing.assert_allclose(
            matrix_cvd_Machado2009("Protanomaly", 0.0),
            np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            matrix_cvd_Machado2009("Deuteranomaly", 0.1),
            np.array(
                [
                    [0.86643500, 0.17770400, -0.04413900],
                    [0.04956700, 0.93906300, 0.01137000],
                    [-0.00345300, 0.00723300, 0.99622000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            matrix_cvd_Machado2009("Tritanomaly", 1.0),
            np.array(
                [
                    [1.25552800, -0.07674900, -0.17877900],
                    [-0.07841100, 0.93080900, 0.14760200],
                    [0.00473300, 0.69136700, 0.30390000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            matrix_cvd_Machado2009("Tritanomaly", 0.55),
            np.array(
                [
                    [1.06088700, -0.01504350, -0.04584350],
                    [-0.01895750, 0.96774750, 0.05121150],
                    [0.00317700, 0.27513700, 0.72168600],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_matrix_cvd_Machado2009(self):
        """
        Test :func:`colour.blindness.machado2009.matrix_cvd_Machado2009`
        definition nan support.
        """

        for case in [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]:
            matrix_cvd_Machado2009("Tritanomaly", case)


if __name__ == "__main__":
    unittest.main()
