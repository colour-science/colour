"""Define the unit tests for the :mod:`colour.graph.common` module."""

from __future__ import annotations

import unittest

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.graph import colourspace_model_to_reference
from colour.models import COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestColourspaceModelToReference",
]


class TestColourspaceModelToReference(unittest.TestCase):
    """
    Define :func:`colour.graph.common.colourspace_model_to_reference`
    definition unit tests methods.
    """

    def test_colourspace_model_to_reference(self) -> None:
        """
        Test :func:`colour.graph.common.colourspace_model_to_reference`
        definition.
        """

        Lab_1 = np.array([0.41527875, 0.52638583, 0.26923179])
        Lab_reference = colourspace_model_to_reference(Lab_1, "CIE Lab")
        np.testing.assert_allclose(
            Lab_reference,
            np.array([41.52787529, 52.63858304, 26.92317922]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Luv_1 = np.array([0.41527875, 0.52638583, 0.26923179])
        Luv_reference = colourspace_model_to_reference(Luv_1, "CIE Luv")
        np.testing.assert_allclose(
            Luv_reference,
            np.array([41.52787529, 52.63858304, 26.92317922]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        CAM02LCD_1 = np.array([0.5, 0.5, 0.5])
        CAM02LCD_reference = colourspace_model_to_reference(CAM02LCD_1, "CAM02LCD")
        np.testing.assert_allclose(
            CAM02LCD_reference,
            np.array([50.0, 50.0, 50.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_1 = np.array([0.20654008, 0.12197225, 0.05136952])
        XYZ_reference = colourspace_model_to_reference(XYZ_1, "CIE XYZ")
        np.testing.assert_allclose(
            XYZ_reference,
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB_1 = np.array([0.5, 0.3, 0.8])
        RGB_reference = colourspace_model_to_reference(RGB_1, "RGB")
        np.testing.assert_allclose(
            RGB_reference,
            np.array([0.5, 0.3, 0.8]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        value_1 = np.array([0.5, 0.5, 0.5])
        value_reference = colourspace_model_to_reference(value_1, "Invalid Model")
        np.testing.assert_allclose(
            value_reference,
            value_1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        for model in COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE:
            with self.subTest(model=model):
                np.testing.assert_allclose(
                    colourspace_model_to_reference(value_1, model),
                    value_1
                    * COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[model],
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                    err_msg=f"Mismatch for model: {model}",
                )


if __name__ == "__main__":
    unittest.main()
