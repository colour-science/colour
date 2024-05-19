# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.luts.cinespace_csp` module."""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.io import LUT1D, LUT3x1D, read_LUT_Cinespace, write_LUT_Cinespace
from colour.utilities import tstack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_LUTS",
    "TestReadLUTCinespace",
    "TestWriteLUTCinespace",
]

ROOT_LUTS: str = os.path.join(os.path.dirname(__file__), "resources", "cinespace")


class TestReadLUTCinespace:
    """
    Define :func:`colour.io.luts.cinespace_csp.read_LUT_Cinespace` definition
    unit tests methods.
    """

    def test_read_LUT_Cinespace(self):
        """
        Test :func:`colour.io.luts.cinespace_csp.read_LUT_Cinespace`
        definition.
        """

        LUT_1 = read_LUT_Cinespace(os.path.join(ROOT_LUTS, "ACES_Proxy_10_to_ACES.csp"))

        np.testing.assert_allclose(
            LUT_1.table,
            np.array(
                [
                    [4.88300000e-04, 4.88300000e-04, 4.88300000e-04],
                    [7.71400000e-04, 7.71400000e-04, 7.71400000e-04],
                    [1.21900000e-03, 1.21900000e-03, 1.21900000e-03],
                    [1.92600000e-03, 1.92600000e-03, 1.92600000e-03],
                    [3.04400000e-03, 3.04400000e-03, 3.04400000e-03],
                    [4.80900000e-03, 4.80900000e-03, 4.80900000e-03],
                    [7.59900000e-03, 7.59900000e-03, 7.59900000e-03],
                    [1.20100000e-02, 1.20100000e-02, 1.20100000e-02],
                    [1.89700000e-02, 1.89700000e-02, 1.89700000e-02],
                    [2.99800000e-02, 2.99800000e-02, 2.99800000e-02],
                    [4.73700000e-02, 4.73700000e-02, 4.73700000e-02],
                    [7.48400000e-02, 7.48400000e-02, 7.48400000e-02],
                    [1.18300000e-01, 1.18300000e-01, 1.18300000e-01],
                    [1.86900000e-01, 1.86900000e-01, 1.86900000e-01],
                    [2.95200000e-01, 2.95200000e-01, 2.95200000e-01],
                    [4.66500000e-01, 4.66500000e-01, 4.66500000e-01],
                    [7.37100000e-01, 7.37100000e-01, 7.37100000e-01],
                    [1.16500000e00, 1.16500000e00, 1.16500000e00],
                    [1.84000000e00, 1.84000000e00, 1.84000000e00],
                    [2.90800000e00, 2.90800000e00, 2.90800000e00],
                    [4.59500000e00, 4.59500000e00, 4.59500000e00],
                    [7.26000000e00, 7.26000000e00, 7.26000000e00],
                    [1.14700000e01, 1.14700000e01, 1.14700000e01],
                    [1.81300000e01, 1.81300000e01, 1.81300000e01],
                    [2.86400000e01, 2.86400000e01, 2.86400000e01],
                    [4.52500000e01, 4.52500000e01, 4.52500000e01],
                    [7.15100000e01, 7.15100000e01, 7.15100000e01],
                    [1.13000000e02, 1.13000000e02, 1.13000000e02],
                    [1.78500000e02, 1.78500000e02, 1.78500000e02],
                    [2.82100000e02, 2.82100000e02, 2.82100000e02],
                    [4.45700000e02, 4.45700000e02, 4.45700000e02],
                    [7.04300000e02, 7.04300000e02, 7.04300000e02],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        assert LUT_1.name == "ACES Proxy 10 to ACES"
        assert LUT_1.dimensions == 2
        np.testing.assert_array_equal(LUT_1.domain, np.array([[0, 0, 0], [1, 1, 1]]))
        assert LUT_1.size == 32
        assert LUT_1.comments == []

        LUT_2 = read_LUT_Cinespace(os.path.join(ROOT_LUTS, "Demo.csp"))
        assert LUT_2.comments == ["Comments are ignored by most parsers"]
        np.testing.assert_array_equal(LUT_2.domain, np.array([[0, 0, 0], [1, 2, 3]]))

        LUT_3 = read_LUT_Cinespace(
            os.path.join(ROOT_LUTS, "Three_Dimensional_Table.csp")
        )
        assert LUT_3.dimensions == 3
        assert LUT_3.size == 2

        LUT_4 = read_LUT_Cinespace(os.path.join(ROOT_LUTS, "Explicit_Domain.csp"))
        assert LUT_4[0].is_domain_explicit() is True
        assert LUT_4[1].table.shape == (2, 3, 4, 3)

        LUT_5 = read_LUT_Cinespace(
            os.path.join(ROOT_LUTS, "Uncommon_3x1D_With_Pre_Lut.csp")
        )
        assert isinstance(LUT_5[0], LUT3x1D)
        assert isinstance(LUT_5[1], LUT3x1D)


class TestWriteLUTCinespace:
    """
    Define :func:`colour.io.luts.cinespace_csp.write_LUT_Cinespace` definition
    unit tests methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def teardown_method(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_write_LUT_Cinespace(self):
        """
        Test :func:`colour.io.luts.cinespace_csp.write_LUT_Cinespace`
        definition.
        """

        LUT_1_r = read_LUT_Cinespace(
            os.path.join(ROOT_LUTS, "ACES_Proxy_10_to_ACES.csp")
        )
        write_LUT_Cinespace(
            LUT_1_r,
            os.path.join(self._temporary_directory, "ACES_Proxy_10_to_ACES.csp"),
        )
        LUT_1_t = read_LUT_Cinespace(
            os.path.join(self._temporary_directory, "ACES_Proxy_10_to_ACES.csp")
        )
        assert LUT_1_r == LUT_1_t
        assert LUT_1_r == LUT_1_t

        LUT_2_r = read_LUT_Cinespace(os.path.join(ROOT_LUTS, "Demo.csp"))
        write_LUT_Cinespace(
            LUT_2_r, os.path.join(self._temporary_directory, "Demo.csp")
        )
        LUT_2_t = read_LUT_Cinespace(
            os.path.join(self._temporary_directory, "Demo.csp")
        )
        assert LUT_2_r == LUT_2_t
        assert LUT_2_r.comments == LUT_2_t.comments

        LUT_3_r = read_LUT_Cinespace(
            os.path.join(ROOT_LUTS, "Three_Dimensional_Table.csp")
        )
        write_LUT_Cinespace(
            LUT_3_r,
            os.path.join(self._temporary_directory, "Three_Dimensional_Table.csp"),
        )
        LUT_3_t = read_LUT_Cinespace(
            os.path.join(self._temporary_directory, "Three_Dimensional_Table.csp")
        )
        assert LUT_3_r == LUT_3_t

        domain = tstack(
            (
                np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.2]),
                np.array([-0.1, 0.5, 1.0, np.nan, np.nan, np.nan]),
                np.array([-1.0, -0.5, 0.0, 0.5, 1.0, np.nan]),
            )
        )
        LUT_4_t = LUT3x1D(domain=domain, table=domain * 2, name="Ragged Domain")
        write_LUT_Cinespace(
            LUT_4_t,
            os.path.join(self._temporary_directory, "Ragged_Domain.csp"),
        )
        LUT_4_r = read_LUT_Cinespace(os.path.join(ROOT_LUTS, "Ragged_Domain.csp"))
        np.testing.assert_allclose(
            LUT_4_t.domain, LUT_4_r.domain, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(LUT_4_t.table, LUT_4_r.table, atol=5e-5)

        LUT_5_r = read_LUT_Cinespace(
            os.path.join(ROOT_LUTS, "Three_Dimensional_Table_With_Shaper.csp")
        )
        LUT_5_r.sequence[0] = LUT_5_r.sequence[0].convert(LUT1D, force_conversion=True)
        write_LUT_Cinespace(
            LUT_5_r,
            os.path.join(
                self._temporary_directory,
                "Three_Dimensional_Table_With_Shaper.csp",
            ),
        )
        LUT_5_r = read_LUT_Cinespace(
            os.path.join(ROOT_LUTS, "Three_Dimensional_Table_With_Shaper.csp")
        )
        LUT_5_t = read_LUT_Cinespace(
            os.path.join(
                self._temporary_directory,
                "Three_Dimensional_Table_With_Shaper.csp",
            )
        )
        assert LUT_5_r == LUT_5_t

        LUT_6_r = read_LUT_Cinespace(
            os.path.join(ROOT_LUTS, "Three_Dimensional_Table_With_Shaper.csp")
        )
        LUT_6_r.sequence[0] = LUT_6_r.sequence[0].convert(
            LUT3x1D, force_conversion=True
        )
        write_LUT_Cinespace(
            LUT_6_r,
            os.path.join(
                self._temporary_directory,
                "Three_Dimensional_Table_With_Shaper.csp",
            ),
        )
        LUT_6_r = read_LUT_Cinespace(
            os.path.join(ROOT_LUTS, "Three_Dimensional_Table_With_Shaper.csp")
        )
        LUT_6_t = read_LUT_Cinespace(
            os.path.join(
                self._temporary_directory,
                "Three_Dimensional_Table_With_Shaper.csp",
            )
        )
        assert LUT_6_r == LUT_6_t

        LUT_7_r = read_LUT_Cinespace(
            os.path.join(ROOT_LUTS, "ACES_Proxy_10_to_ACES.csp")
        )
        write_LUT_Cinespace(
            LUT_7_r.convert(LUT1D, force_conversion=True),
            os.path.join(self._temporary_directory, "ACES_Proxy_10_to_ACES.csp"),
        )
        LUT_7_t = read_LUT_Cinespace(
            os.path.join(self._temporary_directory, "ACES_Proxy_10_to_ACES.csp")
        )
        assert LUT_7_r == LUT_7_t

    def test_raise_exception_write_LUT_Cinespace(self):
        """
        Test :func:`colour.io.luts.cinespace_csp.write_LUT_Cinespace`
        definition raised exception.
        """

        pytest.raises(TypeError, write_LUT_Cinespace, object(), "")
