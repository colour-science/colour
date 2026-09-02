"""Define the unit tests for the :mod:`colour.adaptation.cmccat2000."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.adaptation.cmccat2000 import (
    chromatic_adaptation_CMCCAT2000,
    chromatic_adaptation_forward_CMCCAT2000,
    chromatic_adaptation_inverse_CMCCAT2000,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
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
    "TestChromaticAdaptationForwardCMCCAT2000",
    "TestChromaticAdaptationInverseCMCCAT2000",
    "TestChromaticAdaptationCMCCAT2000",
]


class TestChromaticAdaptationForwardCMCCAT2000:
    """
    Define :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_forward_CMCCAT2000` definition unit tests methods.
    """

    def test_chromatic_adaptation_forward_CMCCAT2000(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_forward_CMCCAT2000` definition.
        """

        xp_assert_close(
            chromatic_adaptation_forward_CMCCAT2000(
                xp_as_array([22.48, 22.74, 8.54], xp=xp),
                xp_as_array([111.15, 100.00, 35.20], xp=xp),
                xp_as_array([94.81, 100.00, 107.30], xp=xp),
                200,
                200,
            ),
            [19.52698326, 23.06833960, 24.97175229],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_forward_CMCCAT2000(
                xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100,
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp) * 100,
                xp_as_array([1.09846607, 1.00000000, 0.35582280], xp=xp) * 100,
                100,
                100,
            ),
            [17.90511171, 22.75299363, 3.79837384],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_forward_CMCCAT2000(
                xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp) * 100,
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp) * 100,
                xp_as_array([0.99144661, 1.00000000, 0.67315942], xp=xp) * 100,
                100,
                100,
            ),
            [6.76564344, 5.86585763, 18.40577315],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_chromatic_adaptation_forward_CMCCAT2000(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_forward_CMCCAT2000` definition n-dimensional arrays
        support.
        """

        XYZ = xp_as_array([22.48, 22.74, 8.54], xp=xp)
        XYZ_w = xp_as_array([111.15, 100.00, 35.20], xp=xp)
        XYZ_wr = xp_as_array([94.81, 100.00, 107.30], xp=xp)
        L_A1 = 200
        L_A2 = 200
        XYZ_c = as_ndarray(
            chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2)
        )

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        XYZ_c = xp.tile(xp_as_array(XYZ_c, xp=xp), (6, 1))
        xp_assert_close(
            chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2),
            XYZ_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        XYZ_wr = xp.tile(xp_as_array(XYZ_wr, xp=xp), (6, 1))
        L_A1 = xp.tile(xp_as_array(L_A1, xp=xp), (6,))
        L_A2 = xp.tile(xp_as_array(L_A2, xp=xp), (6,))
        xp_assert_close(
            chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2),
            XYZ_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        XYZ_wr = xp_reshape(xp_as_array(XYZ_wr, xp=xp), (2, 3, 3), xp=xp)
        L_A1 = xp_reshape(xp_as_array(L_A1, xp=xp), (2, 3), xp=xp)
        L_A2 = xp_reshape(xp_as_array(L_A2, xp=xp), (2, 3), xp=xp)
        XYZ_c = xp_reshape(xp_as_array(XYZ_c, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2),
            XYZ_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_chromatic_adaptation_CMCCAT2000(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_forward_CMCCAT2000` definition domain and range scale
        support.
        """

        XYZ = xp_as_array([22.48, 22.74, 8.54], xp=xp)
        XYZ_w = xp_as_array([111.15, 100.00, 35.20], xp=xp)
        XYZ_wr = xp_as_array([94.81, 100.00, 107.30], xp=xp)
        L_A1 = 200
        L_A2 = 200
        XYZ_c = as_ndarray(
            chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2)
        )

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    chromatic_adaptation_forward_CMCCAT2000(
                        XYZ * factor,
                        XYZ_w * factor,
                        XYZ_wr * factor,
                        L_A1,
                        L_A2,
                    ),
                    XYZ_c * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_forward_CMCCAT2000(self) -> None:
        """
        Test :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_forward_CMCCAT2000` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        chromatic_adaptation_forward_CMCCAT2000(
            cases, cases, cases, cases[..., 0], cases[..., 0]
        )


class TestChromaticAdaptationInverseCMCCAT2000:
    """
    Define :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_inverse_CMCCAT2000` definition unit tests methods.
    """

    def test_chromatic_adaptation_inverse_CMCCAT2000(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_inverse_CMCCAT2000` definition.
        """

        xp_assert_close(
            chromatic_adaptation_inverse_CMCCAT2000(
                xp_as_array([19.52698326, 23.06833960, 24.97175229], xp=xp),
                xp_as_array([111.15, 100.00, 35.20], xp=xp),
                xp_as_array([94.81, 100.00, 107.30], xp=xp),
                200,
                200,
            ),
            [22.48, 22.74, 8.54],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_inverse_CMCCAT2000(
                xp_as_array([17.90511171, 22.75299363, 3.79837384], xp=xp),
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp) * 100,
                xp_as_array([1.09846607, 1.00000000, 0.35582280], xp=xp) * 100,
                100,
                100,
            ),
            xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_inverse_CMCCAT2000(
                xp_as_array([6.76564344, 5.86585763, 18.40577315], xp=xp),
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp) * 100,
                xp_as_array([0.99144661, 1.00000000, 0.67315942], xp=xp) * 100,
                100,
                100,
            ),
            xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_chromatic_adaptation_inverse_CMCCAT2000(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_inverse_CMCCAT2000` definition n-dimensional arrays
        support.
        """

        XYZ_c = xp_as_array([19.52698326, 23.06833960, 24.97175229], xp=xp)
        XYZ_w = xp_as_array([111.15, 100.00, 35.20], xp=xp)
        XYZ_wr = xp_as_array([94.81, 100.00, 107.30], xp=xp)
        L_A1 = 200
        L_A2 = 200
        XYZ = as_ndarray(
            chromatic_adaptation_inverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr, L_A1, L_A2)
        )

        XYZ_c = xp.tile(xp_as_array(XYZ_c, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            chromatic_adaptation_inverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr, L_A1, L_A2),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        XYZ_wr = xp.tile(xp_as_array(XYZ_wr, xp=xp), (6, 1))
        L_A1 = xp.tile(xp_as_array(L_A1, xp=xp), (6,))
        L_A2 = xp.tile(xp_as_array(L_A2, xp=xp), (6,))
        xp_assert_close(
            chromatic_adaptation_inverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr, L_A1, L_A2),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_c = xp_reshape(xp_as_array(XYZ_c, xp=xp), (2, 3, 3), xp=xp)
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        XYZ_wr = xp_reshape(xp_as_array(XYZ_wr, xp=xp), (2, 3, 3), xp=xp)
        L_A1 = xp_reshape(xp_as_array(L_A1, xp=xp), (2, 3), xp=xp)
        L_A2 = xp_reshape(xp_as_array(L_A2, xp=xp), (2, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            chromatic_adaptation_inverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr, L_A1, L_A2),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_chromatic_adaptation_CMCCAT2000(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_inverse_CMCCAT2000` definition domain and range scale
        support.
        """

        XYZ_c = xp_as_array([19.52698326, 23.06833960, 24.97175229], xp=xp)
        XYZ_w = xp_as_array([111.15, 100.00, 35.20], xp=xp)
        XYZ_wr = xp_as_array([94.81, 100.00, 107.30], xp=xp)
        L_A1 = 200
        L_A2 = 200
        XYZ = as_ndarray(
            chromatic_adaptation_inverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr, L_A1, L_A2)
        )

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    chromatic_adaptation_inverse_CMCCAT2000(
                        XYZ_c * factor,
                        XYZ_w * factor,
                        XYZ_wr * factor,
                        L_A1,
                        L_A2,
                    ),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_inverse_CMCCAT2000(self) -> None:
        """
        Test :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_inverse_CMCCAT2000` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        chromatic_adaptation_inverse_CMCCAT2000(
            cases, cases, cases, cases[..., 0], cases[..., 0]
        )


class TestChromaticAdaptationCMCCAT2000:
    """
    Define :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_CMCCAT2000` wrapper definition unit tests methods.
    """

    def test_chromatic_adaptation_CMCCAT2000(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_CMCCAT2000` wrapper definition.
        """

        xp_assert_close(
            chromatic_adaptation_CMCCAT2000(
                xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100,
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp) * 100,
                xp_as_array([1.09846607, 1.00000000, 0.35582280], xp=xp) * 100,
                100,
                100,
            ),
            [17.90511171, 22.75299363, 3.79837384],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_CMCCAT2000(
                xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100,
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp) * 100,
                xp_as_array([1.09846607, 1.00000000, 0.35582280], xp=xp) * 100,
                100,
                100,
                direction="Forward",
            ),
            [17.90511171, 22.75299363, 3.79837384],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_CMCCAT2000(
                xp_as_array([17.90511171, 22.75299363, 3.79837384], xp=xp),
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp) * 100,
                xp_as_array([1.09846607, 1.00000000, 0.35582280], xp=xp) * 100,
                100,
                100,
                direction="Inverse",
            ),
            xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
