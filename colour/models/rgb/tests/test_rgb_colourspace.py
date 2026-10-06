"""
Define the unit tests for the :mod:`colour.models.rgb.rgb_colourspace` module.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import re
import textwrap
from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    RGB_COLOURSPACE_ACES2065_1,
    RGB_COLOURSPACES,
    RGB_Colourspace,
    RGB_COLOURSPACE_sRGB,
    RGB_to_RGB,
    RGB_to_XYZ,
    XYZ_to_RGB,
    chromatically_adapted_primaries,
    eotf_inverse_sRGB,
    eotf_sRGB,
    linear_function,
    matrix_RGB_to_RGB,
    normalised_primary_matrix,
)
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_assert_equal,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestRGB_Colourspace",
    "TestXYZ_to_RGB",
    "TestRGB_to_XYZ",
    "TestMatrix_RGB_to_RGB",
    "TestRGB_to_RGB",
]


class TestRGB_Colourspace:
    """
    Define :class:`colour.colour.models.RGB_Colourspace` class unit
    tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
        whitepoint = np.array([0.32168, 0.33767])
        matrix_RGB_to_XYZ = np.identity(3)
        matrix_XYZ_to_RGB = np.identity(3)
        self._colourspace = RGB_Colourspace(
            "RGB Colourspace",
            p,
            whitepoint,
            "ACES",
            matrix_RGB_to_XYZ,
            matrix_XYZ_to_RGB,
            linear_function,
            linear_function,
        )

    def test_required_attributes(self) -> None:
        """Test the presence of required attributes."""

        required_attributes = (
            "name",
            "primaries",
            "whitepoint",
            "whitepoint_name",
            "matrix_RGB_to_XYZ",
            "matrix_XYZ_to_RGB",
            "cctf_encoding",
            "cctf_decoding",
            "use_derived_matrix_RGB_to_XYZ",
            "use_derived_matrix_XYZ_to_RGB",
        )

        for attribute in required_attributes:
            assert attribute in dir(RGB_Colourspace)

    def test_required_methods(self) -> None:
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "__str__",
            "__repr__",
            "use_derived_transformation_matrices",
            "chromatically_adapt",
            "copy",
        )

        for method in required_methods:
            assert method in dir(RGB_Colourspace)

    def test__str__(self) -> None:
        """
        Test :meth:`colour.models.rgb.rgb_colourspace.RGB_Colourspace.__str__`
        method.
        """

        assert re.sub(" at 0x\\w+>", "", str(self._colourspace)) == (
            textwrap.dedent(
                """
    RGB Colourspace
    ---------------

    Primaries          : [[ 7.347e-01  2.653e-01]
                          [ 0.000e+00  1.000e+00]
                          [ 1.000e-04 -7.700e-02]]
    Whitepoint         : [0.32168 0.33767]
    Whitepoint Name    : ACES
    Encoding CCTF      : <function linear_function
    Decoding CCTF      : <function linear_function
    NPM                : [[1. 0. 0.]
                          [0. 1. 0.]
                          [0. 0. 1.]]
    NPM -1             : [[1. 0. 0.]
                          [0. 1. 0.]
                          [0. 0. 1.]]
    Derived NPM        : [[ 9.52552396e-01  0.00000000e+00  9.36786317e-05]
                          [ 3.43966450e-01  7.28166097e-01 -7.21325464e-02]
                          [ 0.00000000e+00  0.00000000e+00  1.00882518e+00]]
    Derived NPM -1     : [[ 1.04981102e+00  0.00000000e+00 -9.74845406e-05]
                          [-4.95903023e-01  1.37331305e+00  9.82400361e-02]
                          [ 0.00000000e+00  0.00000000e+00  9.91252018e-01]]
    Use Derived NPM    : False
    Use Derived NPM -1 : False
                """
            ).strip()
        )

    def test__repr__(self) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.RGB_Colourspace.\
__repr__` method.
        """

        assert re.sub(" at 0x\\w+>", "", repr(self._colourspace)) == (
            textwrap.dedent(
                """
        RGB_Colourspace('RGB Colourspace',
                        [[ 7.347e-01,  2.653e-01],
                         [ 0.000e+00,  1.000e+00],
                         [ 1.000e-04, -7.700e-02]],
                        [0.32168, 0.33767],
                        'ACES',
                        [[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]],
                        [[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]],
                        linear_function,
                        linear_function,
                        False,
                        False)
                """
            ).strip()
        )

    def test_use_derived_transformation_matrices(self) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.RGB_Colourspace.\
use_derived_transformation_matrices` method.
        """

        xp_assert_equal(self._colourspace.matrix_RGB_to_XYZ, np.identity(3))
        xp_assert_equal(self._colourspace.matrix_XYZ_to_RGB, np.identity(3))

        self._colourspace.use_derived_transformation_matrices()

        xp_assert_close(
            self._colourspace.matrix_RGB_to_XYZ,
            [
                [0.95255240, 0.00000000, 0.00009368],
                [0.34396645, 0.72816610, -0.07213255],
                [0.00000000, 0.00000000, 1.00882518],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            self._colourspace.matrix_XYZ_to_RGB,
            [
                [1.04981102, 0.00000000, -0.00009748],
                [-0.49590302, 1.37331305, 0.09824004],
                [0.00000000, 0.00000000, 0.99125202],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        self._colourspace.use_derived_matrix_RGB_to_XYZ = False
        xp_assert_equal(self._colourspace.matrix_RGB_to_XYZ, np.identity(3))
        self._colourspace.use_derived_matrix_XYZ_to_RGB = False
        xp_assert_equal(self._colourspace.matrix_XYZ_to_RGB, np.identity(3))

    def test_chromatically_adapt(self) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.RGB_Colourspace.\
chromatically_adapt` method.
        """

        whitepoint_t = np.array([0.31270, 0.32900])
        colourspace = self._colourspace.chromatically_adapt(
            whitepoint_t, "D50", "Bradford"
        )

        xp_assert_close(
            colourspace.primaries,
            [
                [0.73485524, 0.26422533],
                [-0.00617091, 1.01131496],
                [0.01596756, -0.06423550],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            colourspace.whitepoint, whitepoint_t, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        assert colourspace.whitepoint_name == "D50"

        xp_assert_close(
            colourspace.primaries,
            chromatically_adapted_primaries(
                self._colourspace.primaries,
                self._colourspace.whitepoint,
                whitepoint_t,
                "Bradford",
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            colourspace.matrix_RGB_to_XYZ,
            normalised_primary_matrix(colourspace.primaries, colourspace.whitepoint),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            colourspace.matrix_XYZ_to_RGB,
            np.linalg.inv(
                normalised_primary_matrix(colourspace.primaries, colourspace.whitepoint)
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_copy(self) -> None:
        """
        Test :meth:`colour.models.rgb.rgb_colourspace.RGB_Colourspace.copy`
        method.
        """

        assert self._colourspace.copy() is not self


class TestXYZ_to_RGB:
    """
    Define :func:`colour.models.rgb.rgb_colourspace.XYZ_to_RGB` definition
    unit tests methods.
    """

    def test_XYZ_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.XYZ_to_RGB`
        definition.
        """

        xp_assert_close(
            XYZ_to_RGB(
                xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp),
                RGB_COLOURSPACE_sRGB,
                [0.34570, 0.35850],
                "Bradford",
                True,
            ),
            [0.70556403, 0.19112904, 0.22341005],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_RGB(
                xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp),
                RGB_COLOURSPACE_sRGB,
                apply_cctf_encoding=True,
            ),
            [0.72794351, 0.18184112, 0.17951801],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_RGB(
                xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp),
                RGB_COLOURSPACE_ACES2065_1,
                [0.34570, 0.35850],
            ),
            [0.21959099, 0.06985815, 0.04703704],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_RGB(
                xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp),
                "sRGB",
                [0.34570, 0.35850],
                "Bradford",
                True,
            ),
            as_ndarray(
                XYZ_to_RGB(
                    xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp),
                    RGB_COLOURSPACE_sRGB,
                    [0.34570, 0.35850],
                    "Bradford",
                    True,
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # TODO: Remove tests when dropping deprecated signature support.
        xp_assert_close(
            XYZ_to_RGB(
                xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp),
                [0.34570, 0.35850],  # pyright: ignore
                [0.31270, 0.32900],
                xp_as_array(  # pyright: ignore
                    [
                        [3.24062548, -1.53720797, -0.49862860],
                        [-0.96893071, 1.87575606, 0.04151752],
                        [0.05571012, -0.20402105, 1.05699594],
                    ],
                    xp=xp,
                ),
                "Bradford",  # pyright: ignore
                eotf_inverse_sRGB,
            ),
            [0.70556599, 0.19109268, 0.22340812],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_RGB(
                xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp),
                [0.34570, 0.35850],  # pyright: ignore
                [0.31270, 0.32900],
                xp_as_array(  # pyright: ignore
                    [
                        [3.24062548, -1.53720797, -0.49862860],
                        [-0.96893071, 1.87575606, 0.04151752],
                        [0.05571012, -0.20402105, 1.05699594],
                    ],
                    xp=xp,
                ),
                None,  # pyright: ignore
                eotf_inverse_sRGB,
            ),
            [0.72794579, 0.18180021, 0.17951580],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_RGB(
                xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp),
                [0.34570, 0.35850],  # pyright: ignore
                [0.32168, 0.33767],
                xp_as_array(  # pyright: ignore
                    [
                        [1.04981102, 0.00000000, -0.00009748],
                        [-0.49590302, 1.37331305, 0.09824004],
                        [0.00000000, 0.00000000, 0.99125202],
                    ],
                    xp=xp,
                ),
            ),
            [0.21959099, 0.06985815, 0.04703704],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_RGB(
                xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp),
                [0.34570, 0.35850],  # pyright: ignore
                [0.31270, 0.32900, 1.00000],
                xp_as_array(  # pyright: ignore
                    [
                        [3.24062548, -1.53720797, -0.49862860],
                        [-0.96893071, 1.87575606, 0.04151752],
                        [0.05571012, -0.20402105, 1.05699594],
                    ],
                    xp=xp,
                ),
            ),
            [0.45620801, 0.03079991, 0.04091883],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.XYZ_to_RGB` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp)
        W_R = xp_as_array([0.34570, 0.35850], xp=xp)
        RGB = as_ndarray(XYZ_to_RGB(XYZ, "sRGB", W_R, "Bradford", True))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        RGB = xp.tile(xp_as_array(RGB, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_RGB(XYZ, "sRGB", W_R, "Bradford", True),
            RGB,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        W_R = xp.tile(xp_as_array(W_R, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_RGB(XYZ, "sRGB", W_R, "Bradford", True),
            RGB,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        W_R = xp_reshape(xp_as_array(W_R, xp=xp), (2, 3, 2), xp=xp)
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            XYZ_to_RGB(XYZ, "sRGB", W_R, "Bradford", True),
            RGB,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.XYZ_to_RGB` definition
        domain and range scale support.
        """

        XYZ = xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp)
        W_R = xp_as_array([0.34570, 0.35850], xp=xp)
        RGB = as_ndarray(XYZ_to_RGB(XYZ, "sRGB", W_R))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_RGB(XYZ * factor, "sRGB", W_R),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.XYZ_to_RGB` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_RGB(cases, "sRGB", cases[..., 0:2])


class TestRGB_to_XYZ:
    """
    Define :func:`colour.models.rgb.rgb_colourspace.RGB_to_XYZ` definition
    unit tests methods.
    """

    def test_RGB_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.RGB_to_XYZ`
        definition.
        """

        xp_assert_close(
            RGB_to_XYZ(
                xp_as_array([0.70556403, 0.19112904, 0.22341005], xp=xp),
                RGB_COLOURSPACE_sRGB,
                [0.34570, 0.35850],
                "Bradford",
                True,
            ),
            [0.21639121, 0.12570714, 0.03847642],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_XYZ(
                xp_as_array([0.72794351, 0.18184112, 0.17951801], xp=xp),
                RGB_COLOURSPACE_sRGB,
                apply_cctf_decoding=True,
            ),
            [0.21639100, 0.12570754, 0.03847682],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_XYZ(
                xp_as_array([0.21959099, 0.06985815, 0.04703704], xp=xp),
                RGB_COLOURSPACE_ACES2065_1,
                [0.34570, 0.35850],
            ),
            [0.21638819, 0.12570000, 0.03847493],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_XYZ(
                xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp),
                "sRGB",
                [0.34570, 0.35850],
                "Bradford",
                True,
            ),
            as_ndarray(
                RGB_to_XYZ(
                    xp_as_array([0.21638819, 0.12570000, 0.03847493], xp=xp),
                    RGB_COLOURSPACE_sRGB,
                    [0.34570, 0.35850],
                    "Bradford",
                    True,
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # TODO: Remove tests when dropping deprecated signature support.
        xp_assert_close(
            RGB_to_XYZ(
                xp_as_array([0.70556599, 0.19109268, 0.22340812], xp=xp),
                [0.31270, 0.32900],  # pyright: ignore
                [0.34570, 0.35850],
                xp_as_array(  # pyright: ignore
                    [
                        [0.41240000, 0.35760000, 0.18050000],
                        [0.21260000, 0.71520000, 0.07220000],
                        [0.01930000, 0.11920000, 0.95050000],
                    ],
                    xp=xp,
                ),
                "Bradford",  # pyright: ignore
                eotf_sRGB,
            ),
            [0.21638819, 0.12570000, 0.03847493],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_XYZ(
                xp_as_array([0.72794579, 0.18180021, 0.17951580], xp=xp),
                [0.31270, 0.32900],  # pyright: ignore
                [0.34570, 0.35850],
                xp_as_array(  # pyright: ignore
                    [
                        [0.41240000, 0.35760000, 0.18050000],
                        [0.21260000, 0.71520000, 0.07220000],
                        [0.01930000, 0.11920000, 0.95050000],
                    ],
                    xp=xp,
                ),
                None,  # pyright: ignore
                eotf_sRGB,
            ),
            [0.21638819, 0.12570000, 0.03847493],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_XYZ(
                xp_as_array([0.21959099, 0.06985815, 0.04703704], xp=xp),
                [0.32168, 0.33767],  # pyright: ignore
                [0.34570, 0.35850],
                xp_as_array(  # pyright: ignore
                    [
                        [0.95255240, 0.00000000, 0.00009368],
                        [0.34396645, 0.72816610, -0.07213255],
                        [0.00000000, 0.00000000, 1.00882518],
                    ],
                    xp=xp,
                ),
            ),
            [0.21638819, 0.12570000, 0.03847493],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_XYZ(
                xp_as_array([0.45620801, 0.03079991, 0.04091883], xp=xp),
                [0.31270, 0.32900, 1.00000],  # pyright: ignore
                [0.34570, 0.35850],
                xp_as_array(  # pyright: ignore
                    [
                        [0.41240000, 0.35760000, 0.18050000],
                        [0.21260000, 0.71520000, 0.07220000],
                        [0.01930000, 0.11920000, 0.95050000],
                    ],
                    xp=xp,
                ),
            ),
            [0.21638819, 0.12570000, 0.03847493],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.RGB_to_XYZ` definition
        n-dimensional support.
        """

        RGB = xp_as_array([0.70556599, 0.19109268, 0.22340812], xp=xp)
        W_R = xp_as_array([0.31270, 0.32900], xp=xp)
        XYZ = as_ndarray(RGB_to_XYZ(RGB, "sRGB", W_R, "Bradford", True))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            RGB_to_XYZ(RGB, "sRGB", W_R, "Bradford", True),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        W_R = xp.tile(xp_as_array(W_R, xp=xp), (6, 1))
        xp_assert_close(
            RGB_to_XYZ(RGB, "sRGB", W_R, "Bradford", True),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (2, 3, 3), xp=xp)
        W_R = xp_reshape(xp_as_array(W_R, xp=xp), (2, 3, 2), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            RGB_to_XYZ(RGB, "sRGB", W_R, "Bradford", True),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.RGB_to_XYZ` definition
        domain and range scale support.
        """

        RGB = xp_as_array([0.45620801, 0.03079991, 0.04091883], xp=xp)
        W_R = xp_as_array([0.31270, 0.32900], xp=xp)
        XYZ = as_ndarray(RGB_to_XYZ(RGB, "sRGB", W_R))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    RGB_to_XYZ(RGB * factor, "sRGB", W_R),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.RGB_to_XYZ` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_XYZ(cases, "sRGB", cases[..., 0:2])


class TestMatrix_RGB_to_RGB:
    """
    Define :func:`colour.models.rgb.rgb_colourspace.matrix_RGB_to_RGB`
    definition unit tests methods.
    """

    def test_matrix_RGB_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.matrix_RGB_to_RGB`
        definition.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES["ACES2065-1"]
        aces_cg_colourspace = RGB_COLOURSPACES["ACEScg"]
        sRGB_colourspace = RGB_COLOURSPACES["sRGB"]

        xp_assert_close(
            matrix_RGB_to_RGB(aces_2065_1_colourspace, sRGB_colourspace),
            [
                [2.52164943, -1.13688855, -0.38491759],
                [-0.27521355, 1.36970515, -0.09439245],
                [-0.01592501, -0.14780637, 1.16380582],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_RGB_to_RGB(sRGB_colourspace, aces_2065_1_colourspace),
            [
                [0.43958564, 0.38392940, 0.17653274],
                [0.08953957, 0.81474984, 0.09568361],
                [0.01738718, 0.10873911, 0.87382059],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_RGB_to_RGB(aces_2065_1_colourspace, aces_cg_colourspace, "Bradford"),
            [
                [1.45143932, -0.23651075, -0.21492857],
                [-0.07655377, 1.17622970, -0.09967593],
                [0.00831615, -0.00603245, 0.99771630],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_RGB_to_RGB(aces_2065_1_colourspace, sRGB_colourspace, "Bradford"),
            [
                [2.52140089, -1.13399575, -0.38756186],
                [-0.27621406, 1.37259557, -0.09628236],
                [-0.01532020, -0.15299256, 1.16838720],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_RGB_to_RGB(aces_2065_1_colourspace, sRGB_colourspace, None),
            [
                [2.55809607, -1.11933692, -0.39181451],
                [-0.27771575, 1.36589396, -0.09353075],
                [-0.01711199, -0.14854588, 1.08104848],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_RGB_to_RGB(aces_2065_1_colourspace, sRGB_colourspace),
            matrix_RGB_to_RGB("ACES2065-1", "sRGB"),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestRGB_to_RGB:
    """
    Define :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition
    unit tests methods.
    """

    def test_RGB_to_RGB(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition."""

        aces_2065_1_colourspace = RGB_COLOURSPACES["ACES2065-1"]
        sRGB_colourspace = RGB_COLOURSPACES["sRGB"]

        xp_assert_close(
            RGB_to_RGB(
                xp_as_array([0.21931722, 0.06950287, 0.04694832], xp=xp),
                aces_2065_1_colourspace,
                sRGB_colourspace,
            ),
            [0.45595289, 0.03040780, 0.04087313],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_RGB(
                xp_as_array([0.45595571, 0.03039702, 0.04087245], xp=xp),
                sRGB_colourspace,
                aces_2065_1_colourspace,
            ),
            [0.21931722, 0.06950287, 0.04694832],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_RGB(
                xp_as_array([0.21931722, 0.06950287, 0.04694832], xp=xp),
                aces_2065_1_colourspace,
                sRGB_colourspace,
                "Bradford",
            ),
            [0.45597530, 0.03030054, 0.04086041],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_RGB(
                xp_as_array([0.21931722, 0.06950287, 0.04694832], xp=xp),
                aces_2065_1_colourspace,
                sRGB_colourspace,
                None,
            ),
            [0.46484236, 0.02963459, 0.03667609],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        aces_cg_colourspace = RGB_COLOURSPACES["ACEScg"]
        aces_cc_colourspace = RGB_COLOURSPACES["ACEScc"]

        xp_assert_close(
            RGB_to_RGB(
                xp_as_array([0.21931722, 0.06950287, 0.04694832], xp=xp),
                aces_cg_colourspace,
                aces_cc_colourspace,
                apply_cctf_decoding=True,
                apply_cctf_encoding=True,
            ),
            [0.42985679, 0.33522924, 0.30292336],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_RGB(
                xp_as_array([0.46956438, 0.48137533, 0.43788601], xp=xp),
                aces_cc_colourspace,
                sRGB_colourspace,
                apply_cctf_decoding=True,
                apply_cctf_encoding=True,
            ),
            [0.60983062, 0.67896356, 0.50435764],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_equal(
            as_ndarray(
                RGB_to_RGB(
                    xp_as_array([0.21931722, 0.06950287, 0.04694832], xp=xp),
                    aces_2065_1_colourspace,
                    RGB_COLOURSPACES["ProPhoto RGB"],
                    apply_cctf_encoding=True,
                    out_int=True,
                )
            ),
            [120, 59, 46],
        )

        xp_assert_close(
            RGB_to_RGB(
                xp_as_array([0.21931722, 0.06950287, 0.04694832], xp=xp),
                aces_2065_1_colourspace,
                sRGB_colourspace,
            ),
            as_ndarray(
                RGB_to_RGB(
                    xp_as_array([0.21931722, 0.06950287, 0.04694832], xp=xp),
                    "ACES2065-1",
                    "sRGB",
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition
        n-dimensional support.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES["ACES2065-1"]
        sRGB_colourspace = RGB_COLOURSPACES["sRGB"]
        RGB_i = xp_as_array([0.21931722, 0.06950287, 0.04694832], xp=xp)
        RGB_o = as_ndarray(RGB_to_RGB(RGB_i, aces_2065_1_colourspace, sRGB_colourspace))

        RGB_i = xp.tile(xp_as_array(RGB_i, xp=xp), (6, 1))
        RGB_o = xp.tile(xp_as_array(RGB_o, xp=xp), (6, 1))
        xp_assert_close(
            RGB_to_RGB(RGB_i, aces_2065_1_colourspace, sRGB_colourspace),
            RGB_o,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB_i = xp_reshape(xp_as_array(RGB_i, xp=xp), (2, 3, 3), xp=xp)
        RGB_o = xp_reshape(xp_as_array(RGB_o, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            RGB_to_RGB(RGB_i, aces_2065_1_colourspace, sRGB_colourspace),
            RGB_o,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition
        domain and range scale support.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES["ACES2065-1"]
        sRGB_colourspace = RGB_COLOURSPACES["sRGB"]
        RGB_i = xp_as_array([0.21931722, 0.06950287, 0.04694832], xp=xp)
        RGB_o = as_ndarray(RGB_to_RGB(RGB_i, aces_2065_1_colourspace, sRGB_colourspace))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    RGB_to_RGB(
                        RGB_i * factor,
                        aces_2065_1_colourspace,
                        sRGB_colourspace,
                    ),
                    RGB_o * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition
        nan support.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES["ACES2065-1"]
        sRGB_colourspace = RGB_COLOURSPACES["sRGB"]

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_RGB(cases, aces_2065_1_colourspace, sRGB_colourspace)
