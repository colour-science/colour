"""Define the unit tests for the :mod:`colour.difference.cam02_ucs` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.difference import delta_E_CAM02LCD, delta_E_CAM02SCD, delta_E_CAM02UCS
from colour.difference.cam02_ucs import delta_E_Luo2006
from colour.models.cam02_ucs import COEFFICIENTS_UCS_LUO2006
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestDelta_E_Luo2006",
]


class TestDelta_E_Luo2006:
    """
    Define :func:`colour.difference.cam02_ucs.delta_E_Luo2006` definition unit
    tests methods.
    """

    def test_delta_E_Luo2006(self, xp: ModuleType) -> None:
        """Test :func:`colour.difference.cam02_ucs.delta_E_Luo2006` definition."""

        xp_assert_close(
            delta_E_Luo2006(
                xp_as_array([54.90433134, -0.08450395, -0.06854831], xp=xp),
                xp_as_array([54.80352754, -3.96940084, -13.57591013], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            ),
            14.055546437777583,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xp.stack(
                list(
                    delta_E_Luo2006(
                        xp_as_array([54.90433134, -0.08450395, -0.06854831], xp=xp),
                        xp_as_array([54.80352754, -3.96940084, -13.57591013], xp=xp),
                        COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
                        additional_data=True,
                    ).values
                )
            ),
            [
                14.055546437777583,
                0.1309140259740277,
                3.8848968900000003,
                13.50736182,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            delta_E_Luo2006(
                xp_as_array([54.90433134, -0.08450395, -0.06854831], xp=xp),
                xp_as_array([54.80352754, -3.96940084, -13.57591013], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            ),
            as_ndarray(
                delta_E_CAM02LCD(
                    xp_as_array([54.90433134, -0.08450395, -0.06854831], xp=xp),
                    xp_as_array([54.80352754, -3.96940084, -13.57591013], xp=xp),
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            delta_E_Luo2006(
                xp_as_array([54.90433134, -0.08450395, -0.06854831], xp=xp),
                xp_as_array([54.80352754, -3.96940084, -13.57591013], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-SCD"],
            ),
            as_ndarray(
                delta_E_CAM02SCD(
                    xp_as_array([54.90433134, -0.08450395, -0.06854831], xp=xp),
                    xp_as_array([54.80352754, -3.96940084, -13.57591013], xp=xp),
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            delta_E_Luo2006(
                xp_as_array([54.90433134, -0.08450395, -0.06854831], xp=xp),
                xp_as_array([54.80352754, -3.96940084, -13.57591013], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-UCS"],
            ),
            as_ndarray(
                delta_E_CAM02UCS(
                    xp_as_array([54.90433134, -0.08450395, -0.06854831], xp=xp),
                    xp_as_array([54.80352754, -3.96940084, -13.57591013], xp=xp),
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_delta_E_Luo2006(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.difference.cam02_ucs.delta_E_Luo2006` definition
        n-dimensional arrays support.
        """

        Jpapbp_1 = xp_as_array([54.90433134, -0.08450395, -0.06854831], xp=xp)
        Jpapbp_2 = xp_as_array([54.80352754, -3.96940084, -13.57591013], xp=xp)
        delta_E_p = delta_E_Luo2006(
            Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]
        )
        additional_data = delta_E_Luo2006(
            Jpapbp_1,
            Jpapbp_2,
            COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            additional_data=True,
        )

        Jpapbp_1 = xp.tile(xp_as_array(Jpapbp_1, xp=xp), (6, 1))
        Jpapbp_2 = xp.tile(xp_as_array(Jpapbp_2, xp=xp), (6, 1))
        delta_E_p = xp.tile(xp_as_array(delta_E_p, xp=xp), (6,))
        xp_assert_close(
            delta_E_Luo2006(Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]),
            delta_E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            xp.stack(
                list(
                    delta_E_Luo2006(
                        Jpapbp_1,
                        Jpapbp_2,
                        COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
                        additional_data=True,
                    ).values
                )
            ),
            xp.stack(
                [
                    xp.tile(xp_as_array(val, xp=xp), (6,))
                    for val in additional_data.values
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Jpapbp_1 = xp_reshape(xp_as_array(Jpapbp_1, xp=xp), (2, 3, 3), xp=xp)
        Jpapbp_2 = xp_reshape(xp_as_array(Jpapbp_2, xp=xp), (2, 3, 3), xp=xp)
        delta_E_p = xp_reshape(xp_as_array(delta_E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            delta_E_Luo2006(Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]),
            delta_E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            xp.stack(
                list(
                    delta_E_Luo2006(
                        Jpapbp_1,
                        Jpapbp_2,
                        COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
                        additional_data=True,
                    ).values
                )
            ),
            xp.stack(
                [
                    xp.full((2, 3), float(as_ndarray(val)))
                    for val in additional_data.values
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_delta_E_Luo2006(self) -> None:
        """
        Test :func:`colour.difference.cam02_ucs.delta_E_Luo2006`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        delta_E_Luo2006(cases, cases, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
        delta_E_Luo2006(
            cases,
            cases,
            COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            additional_data=True,
        )
