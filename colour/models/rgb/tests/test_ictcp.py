"""Define the unit tests for the :mod:`colour.models.rgb.ictcp` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb import ICtCp_to_RGB, ICtCp_to_XYZ, RGB_to_ICtCp, XYZ_to_ICtCp
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
    "TestRGB_to_ICtCp",
    "TestICtCp_to_RGB",
    "TestXYZ_to_ICtCp",
    "TestICtCp_to_XYZ",
]


class TestRGB_to_ICtCp:
    """
    Define :func:`colour.models.rgb.ictcp.TestRGB_to_ICtCp` definition unit
    tests methods.
    """

    def test_RGB_to_ICtCp(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ictcp.RGB_to_ICtCp` definition."""

        xp_assert_close(
            RGB_to_ICtCp(xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)),
            [0.07351364, 0.00475253, 0.09351596],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_ICtCp(
                xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp), L_p=4000
            ),
            [0.10516931, 0.00514031, 0.12318730],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_ICtCp(
                xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp), L_p=1000
            ),
            [0.17079612, 0.00485580, 0.17431356],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_ICtCp(
                xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp),
                method="ITU-R BT.2100-1 PQ",
            ),
            [0.07351364, 0.00475253, 0.09351596],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_ICtCp(
                xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp),
                method="ITU-R BT.2100-2 PQ",
            ),
            [0.07351364, 0.00475253, 0.09351596],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_ICtCp(
                xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp),
                method="ITU-R BT.2100-1 HLG",
            ),
            [0.62567899, -0.03622422, 0.67786522],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_ICtCp(
                xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp),
                method="ITU-R BT.2100-2 HLG",
            ),
            [0.62567899, -0.01984490, 0.35911259],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_ICtCp(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.RGB_to_ICtCp` definition
        n-dimensional support.
        """

        RGB = xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)
        ICtCp = as_ndarray(RGB_to_ICtCp(RGB))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (6, 1))
        ICtCp = xp.tile(xp_as_array(ICtCp, xp=xp), (6, 1))
        xp_assert_close(RGB_to_ICtCp(RGB), ICtCp, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (2, 3, 3), xp=xp)
        ICtCp = xp_reshape(xp_as_array(ICtCp, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(RGB_to_ICtCp(RGB), ICtCp, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_RGB_to_ICtCp(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.RGB_to_ICtCp` definition domain
        and range scale support.
        """

        RGB = xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)
        ICtCp = as_ndarray(RGB_to_ICtCp(RGB))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    RGB_to_ICtCp(RGB * factor),
                    ICtCp * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_ICtCp(self) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.RGB_to_ICtCp` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_ICtCp(cases)


class TestICtCp_to_RGB:
    """
    Define :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition unit tests
    methods.
    """

    def test_ICtCp_to_RGB(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition."""

        xp_assert_close(
            ICtCp_to_RGB(xp_as_array([0.07351364, 0.00475253, 0.09351596], xp=xp)),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_RGB(
                xp_as_array([0.10516931, 0.00514031, 0.12318730], xp=xp), L_p=4000
            ),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_RGB(
                xp_as_array([0.17079612, 0.00485580, 0.17431356], xp=xp), L_p=1000
            ),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_RGB(
                xp_as_array([0.07351364, 0.00475253, 0.09351596], xp=xp),
                method="ITU-R BT.2100-1 PQ",
            ),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_RGB(
                xp_as_array([0.07351364, 0.00475253, 0.09351596], xp=xp),
                method="ITU-R BT.2100-2 PQ",
            ),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_RGB(
                xp_as_array([0.62567899, -0.03622422, 0.67786522], xp=xp),
                method="ITU-R BT.2100-1 HLG",
            ),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_RGB(
                xp_as_array([0.62567899, -0.01984490, 0.35911259], xp=xp),
                method="ITU-R BT.2100-2 HLG",
            ),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ICtCp_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition
        n-dimensional support.
        """

        ICtCp = xp_as_array([0.07351364, 0.00475253, 0.09351596], xp=xp)
        RGB = as_ndarray(ICtCp_to_RGB(ICtCp))

        ICtCp = xp.tile(xp_as_array(ICtCp, xp=xp), (6, 1))
        RGB = xp.tile(xp_as_array(RGB, xp=xp), (6, 1))
        xp_assert_close(ICtCp_to_RGB(ICtCp), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        ICtCp = xp_reshape(xp_as_array(ICtCp, xp=xp), (2, 3, 3), xp=xp)
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(ICtCp_to_RGB(ICtCp), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_ICtCp_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition domain
        and range scale support.
        """

        ICtCp = xp_as_array([0.07351364, 0.00475253, 0.09351596], xp=xp)
        RGB = as_ndarray(ICtCp_to_RGB(ICtCp))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ICtCp_to_RGB(ICtCp * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ICtCp_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        ICtCp_to_RGB(cases)


class TestXYZ_to_ICtCp:
    """
    Define :func:`colour.models.rgb.ictcp.TestXYZ_to_ICtCp` definition unit
    tests methods.
    """

    def test_XYZ_to_ICtCp(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ictcp.XYZ_to_ICtCp` definition."""

        xp_assert_close(
            XYZ_to_ICtCp(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            [0.06858097, -0.00283842, 0.06020983],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                [0.34570, 0.35850],
            ),
            [0.06792437, 0.00452089, 0.05514480],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                [0.34570, 0.35850],
                chromatic_adaptation_transform="Bradford",
            ),
            [0.06783951, 0.00476111, 0.05523093],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp), L_p=4000
            ),
            [0.09871102, -0.00447247, 0.07984812],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp), L_p=1000
            ),
            [0.16173872, -0.00792543, 0.11409458],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                method="ITU-R BT.2100-1 PQ",
            ),
            [0.06858097, -0.00283842, 0.06020983],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                method="ITU-R BT.2100-2 PQ",
            ),
            [0.06858097, -0.00283842, 0.06020983],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                method="ITU-R BT.2100-1 HLG",
            ),
            [0.59242792, -0.06824263, 0.47421473],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                method="ITU-R BT.2100-2 HLG",
            ),
            [0.59242792, -0.03740730, 0.25122675],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_ICtCp(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.XYZ_to_ICtCp` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        ICtCp = as_ndarray(XYZ_to_ICtCp(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        ICtCp = xp.tile(xp_as_array(ICtCp, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_ICtCp(XYZ), ICtCp, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        ICtCp = xp_reshape(xp_as_array(ICtCp, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_ICtCp(XYZ), ICtCp, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_ICtCp(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.XYZ_to_ICtCp` definition domain
        and range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        ICtCp = as_ndarray(XYZ_to_ICtCp(XYZ))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_ICtCp(XYZ * factor),
                    ICtCp * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_ICtCp(self) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.XYZ_to_ICtCp` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_ICtCp(cases)


class TestICtCp_to_XYZ:
    """
    Define :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition unit tests
    methods.
    """

    def test_ICtCp_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition."""

        xp_assert_close(
            ICtCp_to_XYZ(xp_as_array([0.06858097, -0.00283842, 0.06020983], xp=xp)),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.06792437, 0.00452089, 0.05514480], xp=xp),
                [0.34570, 0.35850],
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.06783951, 0.00476111, 0.05523093], xp=xp),
                [0.34570, 0.35850],
                chromatic_adaptation_transform="Bradford",
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.09871102, -0.00447247, 0.07984812], xp=xp), L_p=4000
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.16173872, -0.00792543, 0.11409458], xp=xp), L_p=1000
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.06858097, -0.00283842, 0.06020983], xp=xp),
                method="ITU-R BT.2100-1 PQ",
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.06858097, -0.00283842, 0.06020983], xp=xp),
                method="ITU-R BT.2100-2 PQ",
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.59242792, -0.06824263, 0.47421473], xp=xp),
                method="ITU-R BT.2100-1 HLG",
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.59242792, -0.03740730, 0.25122675], xp=xp),
                method="ITU-R BT.2100-2 HLG",
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ICtCp_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition
        n-dimensional support.
        """

        ICtCp = xp_as_array([0.06858097, -0.00283842, 0.06020983], xp=xp)
        XYZ = as_ndarray(ICtCp_to_XYZ(ICtCp))

        ICtCp = xp.tile(xp_as_array(ICtCp, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(ICtCp_to_XYZ(ICtCp), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        ICtCp = xp_reshape(xp_as_array(ICtCp, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(ICtCp_to_XYZ(ICtCp), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_ICtCp_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition domain
        and range scale support.
        """

        ICtCp = xp_as_array([0.06858097, -0.00283842, 0.06020983], xp=xp)
        XYZ = as_ndarray(ICtCp_to_XYZ(ICtCp))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ICtCp_to_XYZ(ICtCp * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ICtCp_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        ICtCp_to_XYZ(cases)
