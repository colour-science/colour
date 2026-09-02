"""Define the unit tests for the :mod:`colour.continuous.multi_signals` module."""

from __future__ import annotations

import pickle
import textwrap
import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np
import pytest

from colour.algebra import (
    CubicSplineInterpolator,
    Extrapolator,
    KernelInterpolator,
    LinearInterpolator,
)
from colour.constants import DTYPE_FLOAT_DEFAULT, TOLERANCE_ABSOLUTE_TESTS
from colour.continuous import MultiSignals, Signal
from colour.utilities import (
    ColourRuntimeWarning,
    as_ndarray,
    attest,
    is_pandas_installed,
    tsplit,
    tstack,
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
    "TestMultiSignals",
]


class TestMultiSignals:
    """
    Define :class:`colour.continuous.multi_signals.MultiSignals` class unit
    tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._range_1 = np.linspace(10, 100, 10)
        self._range_2 = tstack([self._range_1] * 3) + np.array([0, 10, 20])
        self._domain_1 = np.arange(0, 10, 1)
        self._domain_2 = np.arange(100, 1100, 100)

        self._multi_signals = MultiSignals(self._range_2)

    def test_required_attributes(self) -> None:
        """Test the presence of required attributes."""

        required_attributes = (
            "dtype",
            "domain",
            "range",
            "interpolator",
            "interpolator_kwargs",
            "extrapolator",
            "extrapolator_kwargs",
            "function",
            "signals",
            "labels",
            "signal_type",
        )

        for attribute in required_attributes:
            assert attribute in dir(MultiSignals)

    def test_required_methods(self) -> None:
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "__str__",
            "__repr__",
            "__hash__",
            "__getitem__",
            "__setitem__",
            "__contains__",
            "__iter__",
            "__eq__",
            "__ne__",
            "arithmetical_operation",
            "multi_signals_unpack_data",
            "fill_nan",
            "domain_distance",
            "to_dataframe",
        )

        for method in required_methods:
            assert method in dir(MultiSignals)

    def test_pickling(self) -> None:
        """
        Test whether the :class:``colour.continuous.signal.MultiSignals` class
        can be pickled.
        """

        data = pickle.dumps(self._multi_signals)
        data = pickle.loads(data)  # noqa: S301
        assert self._multi_signals == data

    def test_dtype(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.dtype`
        property.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        assert multi_signals.dtype == DTYPE_FLOAT_DEFAULT

        multi_signals = multi_signals.copy()
        multi_signals.dtype = np.float32
        assert multi_signals.dtype == np.float32

    def test_domain(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.domain`
        property.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))

        xp_assert_close(
            multi_signals[[0, 1, 2]],
            [[10.0, 20.0, 30.0], [20.0, 30.0, 40.0], [30.0, 40.0, 50.0]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals.domain = xp_as_array(self._domain_1 * 10, xp=xp)

        xp_assert_equal(multi_signals.domain, self._domain_1 * 10)

        xp_assert_close(
            multi_signals[xp_as_array([0, 1, 2], xp=xp) * 10],
            [[10.0, 20.0, 30.0], [20.0, 30.0, 40.0], [30.0, 40.0, 50.0]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        domain = np.linspace(0, 1, 10)
        domain[0] = -np.inf

        def assert_warns() -> None:
            """Help to test the runtime warning."""

            multi_signals.domain = domain

        with pytest.warns(ColourRuntimeWarning):
            assert_warns()

    def test_range(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.range`
        property.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))

        xp_assert_close(
            multi_signals[[0, 1, 2]],
            [[10.0, 20.0, 30.0], [20.0, 30.0, 40.0], [30.0, 40.0, 50.0]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals.range = xp_as_array(self._range_1 * 10, xp=xp)

        xp_assert_equal(multi_signals.range, tstack([self._range_1] * 3) * 10)

        xp_assert_close(
            multi_signals[[0, 1, 2]],
            xp_as_array(
                [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]], xp=xp
            )
            * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals.range = xp_as_array(self._range_2 * 10, xp=xp)

        xp_assert_equal(multi_signals.range, self._range_2 * 10)

        xp_assert_close(
            multi_signals[[0, 1, 2]],
            xp_as_array(
                [[10.0, 20.0, 30.0], [20.0, 30.0, 40.0], [30.0, 40.0, 50.0]], xp=xp
            )
            * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_interpolator(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.interpolator`
        property.
        """

        # NOTE: A non-linear range is used as a linear one is reproduced
        # exactly by both interpolators.
        multi_signals = MultiSignals(
            xp_as_array(
                tstack([np.sin(np.linspace(0, 1, 10)) * 100] * 3)
                + np.array([0, 10, 20]),
                xp=xp,
            )
        )

        xp_assert_close(
            multi_signals[np.linspace(0, 5, 5)],
            [
                [0.00000000, 10.00000000, 20.00000000],
                [13.82614072, 23.82614072, 33.82614072],
                [27.37962201, 37.37962201, 47.37962201],
                [40.42659468, 50.42659468, 60.42659468],
                [52.74153858, 62.74153858, 72.74153858],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals.interpolator = CubicSplineInterpolator

        xp_assert_close(
            multi_signals[np.linspace(0, 5, 5)],
            [
                [0.00000000, 10.00000000, 20.00000000],
                [13.84426192, 23.84426192, 33.84426192],
                [27.42192158, 37.42192158, 47.42192158],
                [40.47144761, 50.47144761, 60.47144761],
                [52.74153858, 62.74153858, 72.74153858],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_interpolator_kwargs(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.\
interpolator_kwargs` property.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        multi_signals.interpolator = KernelInterpolator

        xp_assert_close(
            multi_signals[np.linspace(0, 5, 5)],
            [
                [10.00000000, 20.00000000, 30.00000000],
                [22.83489024, 32.80460562, 42.77432100],
                [34.80044921, 44.74343470, 54.68642018],
                [47.55353925, 57.52325463, 67.49297001],
                [60.00000000, 70.00000000, 80.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals.interpolator_kwargs = {
            "window": 1,
            "kernel_kwargs": {"a": 1},
        }

        xp_assert_close(
            multi_signals[np.linspace(0, 5, 5)],
            [
                [10.00000000, 20.00000000, 30.00000000],
                [18.91328761, 27.91961505, 36.92594248],
                [28.36993142, 36.47562611, 44.58132080],
                [44.13100443, 53.13733187, 62.14365930],
                [60.00000000, 70.00000000, 80.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_extrapolator(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.extrapolator`
        property.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        assert isinstance(multi_signals.extrapolator(), Extrapolator)

    def test_extrapolator_kwargs(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.\
extrapolator_kwargs` property.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))

        attest(np.all(np.isnan(as_ndarray(multi_signals[np.array([-1000, 1000])]))))

        multi_signals.extrapolator_kwargs = {
            "method": "Linear",
        }

        xp_assert_close(
            multi_signals[[-1000, 1000]],
            [[-9990.0, -9980.0, -9970.0], [10010.0, 10020.0, 10030.0]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_function(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.function`
        property.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        attest(callable(multi_signals.function))

    def test_raise_exception_function(self) -> None:
        """
        Test :func:`colour.continuous.signal.multi_signals.MultiSignals.\
function` property raised exception.
        """

        with pytest.raises((ValueError, TypeError)):
            MultiSignals().function(0)

    def test_signals(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.signals`
        property.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))

        multi_signals.signals = xp_as_array(self._range_1, xp=xp)
        xp_assert_equal(multi_signals.domain, self._domain_1)
        xp_assert_equal(multi_signals.range, self._range_1[:, None])

    def test_labels(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.labels`
        property.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        assert multi_signals.labels == ["0", "1", "2"]

        multi_signals = multi_signals.copy()

        multi_signals.labels = ["a", "b", "c"]

        assert multi_signals.labels == ["a", "b", "c"]

    def test_signal_type(self) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.signal_type`
        property.
        """

        multi_signals = MultiSignals(signal_type=Signal)

        assert multi_signals.signal_type == Signal

    def test__init__(self, xp: ModuleType) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.__init__`
        method.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_1, xp=xp))
        xp_assert_equal(multi_signals.domain, self._domain_1)
        xp_assert_equal(multi_signals.range, self._range_1[:, None])

        multi_signals = MultiSignals(
            xp_as_array(self._range_1, xp=xp), xp_as_array(self._domain_2, xp=xp)
        )
        xp_assert_equal(multi_signals.domain, self._domain_2)
        xp_assert_equal(multi_signals.range, self._range_1[:, None])

        multi_signals = MultiSignals(
            xp_as_array(self._range_2, xp=xp), xp_as_array(self._domain_2, xp=xp)
        )
        xp_assert_equal(multi_signals.domain, self._domain_2)
        xp_assert_equal(multi_signals.range, self._range_2)

        multi_signals = MultiSignals(
            dict(zip(self._domain_2, self._range_2, strict=True))
        )
        xp_assert_equal(multi_signals.domain, self._domain_2)
        xp_assert_equal(multi_signals.range, self._range_2)

        multi_signals = MultiSignals(multi_signals)
        xp_assert_equal(multi_signals.domain, self._domain_2)
        xp_assert_equal(multi_signals.range, self._range_2)

        class NotSignal(Signal):
            """Not :class:`Signal` class."""

        multi_signals = MultiSignals(
            xp_as_array(self._range_1, xp=xp), signal_type=NotSignal
        )
        assert isinstance(multi_signals.signals["0"], NotSignal)
        xp_assert_equal(multi_signals.domain, self._domain_1)
        xp_assert_equal(multi_signals.range, self._range_1[:, None])

        if is_pandas_installed():
            from pandas import DataFrame, Series  # noqa: PLC0415

            multi_signals = MultiSignals(
                Series(dict(zip(self._domain_2, self._range_1, strict=True)))
            )
            xp_assert_equal(multi_signals.domain, self._domain_2)
            xp_assert_equal(multi_signals.range, self._range_1[:, None])

            data = dict(zip(["a", "b", "c"], tsplit(self._range_2), strict=True))
            multi_signals = MultiSignals(DataFrame(data, self._domain_2))
            xp_assert_equal(multi_signals.domain, self._domain_2)
            xp_assert_equal(multi_signals.range, self._range_2)

    def test__hash__(self, xp: ModuleType) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.__hash__`
        method.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        assert isinstance(hash(multi_signals), int)

    def test__str__(self) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.__str__`
        method.
        """

        assert (
            str(self._multi_signals)
            == (
                textwrap.dedent(
                    """
                [[  0.  10.  20.  30.]
                 [  1.  20.  30.  40.]
                 [  2.  30.  40.  50.]
                 [  3.  40.  50.  60.]
                 [  4.  50.  60.  70.]
                 [  5.  60.  70.  80.]
                 [  6.  70.  80.  90.]
                 [  7.  80.  90. 100.]
                 [  8.  90. 100. 110.]
                 [  9. 100. 110. 120.]]"""
                )[1:]
            )
        )

        assert isinstance(str(MultiSignals()), str)

    def test__repr__(self) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.__repr__`
        method.
        """

        assert repr(self._multi_signals) == (
            textwrap.dedent(
                """
                MultiSignals([[  0.,  10.,  20.,  30.],
                              [  1.,  20.,  30.,  40.],
                              [  2.,  30.,  40.,  50.],
                              [  3.,  40.,  50.,  60.],
                              [  4.,  50.,  60.,  70.],
                              [  5.,  60.,  70.,  80.],
                              [  6.,  70.,  80.,  90.],
                              [  7.,  80.,  90., 100.],
                              [  8.,  90., 100., 110.],
                              [  9., 100., 110., 120.]],
                             ['0', '1', '2'],
                             LinearInterpolator,
                             {},
                             Extrapolator,
                             {'method': 'Constant', 'left': nan, 'right': nan})
                """
            ).strip()
        )

        assert isinstance(repr(MultiSignals()), str)

    def test__getitem__(self, xp: ModuleType) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.__getitem__`
        method.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))

        xp_assert_close(
            multi_signals[0],
            [10.0, 20.0, 30.0],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals[[0, 1, 2]],
            [
                [10.0, 20.0, 30.0],
                [20.0, 30.0, 40.0],
                [30.0, 40.0, 50.0],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals[np.linspace(0, 5, 5)],
            [
                [10.00000000, 20.00000000, 30.00000000],
                [22.50000000, 32.50000000, 42.50000000],
                [35.00000000, 45.00000000, 55.00000000],
                [47.50000000, 57.50000000, 67.50000000],
                [60.00000000, 70.00000000, 80.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        attest(np.all(np.isnan(as_ndarray(multi_signals[np.array([-1000, 1000])]))))

        xp_assert_close(
            multi_signals[:],
            multi_signals.range,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals[:, :],
            multi_signals.range,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals[0:3],
            [
                [10.0, 20.0, 30.0],
                [20.0, 30.0, 40.0],
                [30.0, 40.0, 50.0],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals[:, 0:2],
            [
                [10.0, 20.0],
                [20.0, 30.0],
                [30.0, 40.0],
                [40.0, 50.0],
                [50.0, 60.0],
                [60.0, 70.0],
                [70.0, 80.0],
                [80.0, 90.0],
                [90.0, 100.0],
                [100.0, 110.0],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals = multi_signals.copy()
        multi_signals.extrapolator_kwargs = {
            "method": "Linear",
        }
        xp_assert_equal(
            multi_signals[[-1000, 1000]],
            [
                [-9990.0, -9980.0, -9970.0],
                [10010.0, 10020.0, 10030.0],
            ],
        )

        multi_signals.extrapolator_kwargs = {
            "method": "Constant",
            "left": 0,
            "right": 1,
        }
        xp_assert_equal(
            multi_signals[[-1000, 1000]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        )

    def test__setitem__(self, xp: ModuleType) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.__setitem__`
        method.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))

        multi_signals[0] = 20
        xp_assert_close(
            multi_signals[0],
            [20.0, 20.0, 20.0],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals[np.array([0, 1, 2])] = 30
        xp_assert_close(
            multi_signals[[0, 1, 2]],
            [
                [30.0, 30.0, 30.0],
                [30.0, 30.0, 30.0],
                [30.0, 30.0, 30.0],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals[np.linspace(0, 5, 5)] = 50
        xp_assert_close(
            multi_signals.domain,
            [
                0.00,
                1.00,
                1.25,
                2.00,
                2.50,
                3.00,
                3.75,
                4.00,
                5.00,
                6.00,
                7.00,
                8.00,
                9.00,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            multi_signals.range,
            [
                [50.0, 50.0, 50.0],
                [30.0, 30.0, 30.0],
                [50.0, 50.0, 50.0],
                [30.0, 30.0, 30.0],
                [50.0, 50.0, 50.0],
                [40.0, 50.0, 60.0],
                [50.0, 50.0, 50.0],
                [50.0, 60.0, 70.0],
                [50.0, 50.0, 50.0],
                [70.0, 80.0, 90.0],
                [80.0, 90.0, 100.0],
                [90.0, 100.0, 110.0],
                [100.0, 110.0, 120.0],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals[np.array([0, 1, 2])] = np.array([10, 20, 30])
        xp_assert_close(
            multi_signals.range,
            [
                [10.0, 20.0, 30.0],
                [10.0, 20.0, 30.0],
                [50.0, 50.0, 50.0],
                [10.0, 20.0, 30.0],
                [50.0, 50.0, 50.0],
                [40.0, 50.0, 60.0],
                [50.0, 50.0, 50.0],
                [50.0, 60.0, 70.0],
                [50.0, 50.0, 50.0],
                [70.0, 80.0, 90.0],
                [80.0, 90.0, 100.0],
                [90.0, 100.0, 110.0],
                [100.0, 110.0, 120.0],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals[np.array([0, 1, 2])] = xp_reshape(
            xp.arange(1, 10, 1),
            (3, 3),
            xp=xp,
        )
        xp_assert_close(
            multi_signals.range,
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [50.0, 50.0, 50.0],
                [7.0, 8.0, 9.0],
                [50.0, 50.0, 50.0],
                [40.0, 50.0, 60.0],
                [50.0, 50.0, 50.0],
                [50.0, 60.0, 70.0],
                [50.0, 50.0, 50.0],
                [70.0, 80.0, 90.0],
                [80.0, 90.0, 100.0],
                [90.0, 100.0, 110.0],
                [100.0, 110.0, 120.0],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals[:] = 40
        xp_assert_close(multi_signals.range, 40, atol=TOLERANCE_ABSOLUTE_TESTS)

        multi_signals[:, :] = 50  # pyright: ignore
        xp_assert_close(multi_signals.range, 50, atol=TOLERANCE_ABSOLUTE_TESTS)

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        multi_signals[0:3] = 40
        xp_assert_close(
            multi_signals[0:3],
            [
                [40.0, 40.0, 40.0],
                [40.0, 40.0, 40.0],
                [40.0, 40.0, 40.0],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals[:, 0:2] = 50  # pyright: ignore
        xp_assert_close(
            multi_signals.range,
            [
                [50.0, 50.0, 40.0],
                [50.0, 50.0, 40.0],
                [50.0, 50.0, 40.0],
                [50.0, 50.0, 60.0],
                [50.0, 50.0, 70.0],
                [50.0, 50.0, 80.0],
                [50.0, 50.0, 90.0],
                [50.0, 50.0, 100.0],
                [50.0, 50.0, 110.0],
                [50.0, 50.0, 120.0],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test__contains__(self, xp: ModuleType) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.__contains__`
        method.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        assert 0 in multi_signals
        assert 0.5 in multi_signals
        assert 1000 not in multi_signals

    def test__iter__(self, xp: ModuleType) -> None:
        """Test :func:`colour.continuous.signal.Signal.__iter__` method."""

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        domain = np.arange(0, 10)
        for i, domain_range_value in enumerate(multi_signals):
            xp_assert_equal(domain_range_value[0], domain[i])
            xp_assert_equal(domain_range_value[1:], self._range_2[i])

    def test__len__(self, xp: ModuleType) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.__len__`
        method.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        assert len(multi_signals) == 10

    def test__eq__(self, xp: ModuleType) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.__eq__`
        method.
        """

        signal_1 = MultiSignals(xp_as_array(self._range_2, xp=xp))
        signal_2 = MultiSignals(xp_as_array(self._range_2, xp=xp))

        assert signal_1 == signal_2

        assert signal_1 != ()

    def test__ne__(self, xp: ModuleType) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.__ne__`
        method.
        """

        multi_signals_1 = MultiSignals(xp_as_array(self._range_2, xp=xp))
        multi_signals_2 = MultiSignals(xp_as_array(self._range_2, xp=xp))

        multi_signals_2[0] = 20
        assert multi_signals_1 != multi_signals_2

        multi_signals_2[0] = np.array([10, 20, 30])
        assert multi_signals_1 == multi_signals_2

        multi_signals_2.interpolator = CubicSplineInterpolator
        assert multi_signals_1 != multi_signals_2

        multi_signals_2.interpolator = LinearInterpolator
        assert multi_signals_1 == multi_signals_2

        multi_signals_2.interpolator_kwargs = {"window": 1}
        assert multi_signals_1 != multi_signals_2

        multi_signals_2.interpolator_kwargs = {}
        assert multi_signals_1 == multi_signals_2

        class NotExtrapolator(Extrapolator):
            """Not :class:`Extrapolator` class."""

        multi_signals_2.extrapolator = NotExtrapolator
        assert multi_signals_1 != multi_signals_2

        multi_signals_2.extrapolator = Extrapolator
        assert multi_signals_1 == multi_signals_2

        multi_signals_2.extrapolator_kwargs = {}
        assert multi_signals_1 != multi_signals_2

        multi_signals_2.extrapolator_kwargs = {
            "method": "Constant",
            "left": np.nan,
            "right": np.nan,
        }
        assert multi_signals_1 == multi_signals_2

    def test_arithmetical_operation(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.\
arithmetical_operation` method.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))

        xp_assert_close(
            multi_signals.arithmetical_operation(10, "+", False).range,
            self._range_2 + 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals.arithmetical_operation(10, "-", False).range,
            self._range_2 - 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals.arithmetical_operation(10, "*", False).range,
            self._range_2 * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals.arithmetical_operation(10, "/", False).range,
            self._range_2 / 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals.arithmetical_operation(10, "**", False).range,
            self._range_2**10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            (multi_signals + 10).range,
            self._range_2 + 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            (multi_signals - 10).range,
            self._range_2 - 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            (multi_signals * 10).range,
            self._range_2 * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            (multi_signals / 10).range,
            self._range_2 / 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            (multi_signals**10).range,
            self._range_2**10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))

        xp_assert_close(
            multi_signals.arithmetical_operation(10, "+", True).range,
            self._range_2 + 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals.arithmetical_operation(10, "-", True).range,
            self._range_2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals.arithmetical_operation(10, "*", True).range,
            self._range_2 * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals.arithmetical_operation(10, "/", True).range,
            self._range_2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals.arithmetical_operation(10, "**", True).range,
            self._range_2**10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        xp_assert_close(
            multi_signals.arithmetical_operation(self._range_2, "+", False).range,
            self._range_2 + self._range_2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals.arithmetical_operation(multi_signals, "+", False).range,
            self._range_2 + self._range_2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_is_uniform(self, xp: ModuleType) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.is_uniform`
        method.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        assert multi_signals.is_uniform()

        multi_signals = multi_signals.copy()
        multi_signals[0.5] = 1.0
        assert not multi_signals.is_uniform()

    def test_copy(self, xp: ModuleType) -> None:
        """Test :func:`colour.continuous.multi_signals.MultiSignals.copy` method."""

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))
        assert multi_signals is not multi_signals.copy()
        assert multi_signals == multi_signals.copy()

    def test_multi_signals_unpack_data(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.\
multi_signals_unpack_data` method.
        """

        domain, range_, labels = MultiSignals.multi_signals_unpack_data(
            xp_as_array(self._range_1, xp=xp)
        )
        assert labels == ["0"]
        xp_assert_equal(domain, self._domain_1)
        xp_assert_equal(range_, self._range_1.reshape(-1, 1))

        domain, range_, labels = MultiSignals.multi_signals_unpack_data(
            xp_as_array(self._range_1, xp=xp), xp_as_array(self._domain_2, xp=xp)
        )
        assert labels == ["0"]
        xp_assert_equal(domain, self._domain_2)
        xp_assert_equal(range_, self._range_1.reshape(-1, 1))

        domain, range_, labels = MultiSignals.multi_signals_unpack_data(
            self._range_1, dict(zip(self._domain_2, self._range_1, strict=True)).keys()
        )
        xp_assert_equal(domain, self._domain_2)

        domain, range_, labels = MultiSignals.multi_signals_unpack_data(
            xp_as_array(self._range_2, xp=xp), xp_as_array(self._domain_2, xp=xp)
        )
        assert labels == ["0", "1", "2"]
        xp_assert_equal(range_[:, 0], self._range_1)
        xp_assert_equal(range_[:, 1], self._range_1 + 10)
        xp_assert_equal(range_[:, 2], self._range_1 + 20)

        domain, range_, labels = MultiSignals.multi_signals_unpack_data(
            dict(zip(self._domain_2, self._range_2, strict=True))
        )
        assert labels == ["0", "1", "2"]
        xp_assert_equal(range_[:, 0], self._range_1)
        xp_assert_equal(range_[:, 1], self._range_1 + 10)
        xp_assert_equal(range_[:, 2], self._range_1 + 20)

        domain, range_, labels = MultiSignals.multi_signals_unpack_data(
            dict(zip(self._domain_2, self._range_2, strict=True)),
            labels=["0", "0", "0"],
        )
        assert labels == ["0 - 0", "0 - 1", "0 - 2"]

        if is_pandas_installed():
            from pandas import DataFrame, Series  # noqa: PLC0415

            domain, range_, labels = MultiSignals.multi_signals_unpack_data(
                Series(dict(zip(self._domain_1, self._range_1, strict=True)))
            )
            assert labels == ["0"]
            xp_assert_equal(domain, self._domain_1)
            xp_assert_equal(range_, self._range_1.reshape(-1, 1))

            data = dict(zip(["a", "b", "c"], tsplit(self._range_2), strict=True))
            domain, range_, labels = MultiSignals.multi_signals_unpack_data(
                DataFrame(data, self._domain_1)
            )
            assert labels == ["a", "b", "c"]
            xp_assert_equal(range_[:, 0], self._range_1)
            xp_assert_equal(range_[:, 1], self._range_1 + 10)
            xp_assert_equal(range_[:, 2], self._range_1 + 20)

    def test_fill_nan(self, xp: ModuleType) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.fill_nan`
        method.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))

        multi_signals[3:7] = np.nan

        xp_assert_close(
            multi_signals.fill_nan().range,
            [
                [10.0, 20.0, 30.0],
                [20.0, 30.0, 40.0],
                [30.0, 40.0, 50.0],
                [40.0, 50.0, 60.0],
                [50.0, 60.0, 70.0],
                [60.0, 70.0, 80.0],
                [70.0, 80.0, 90.0],
                [80.0, 90.0, 100.0],
                [90.0, 100.0, 110.0],
                [100.0, 110.0, 120.0],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        multi_signals[3:7] = np.nan

        xp_assert_close(
            multi_signals.fill_nan(method="Constant").range,
            [
                [10.0, 20.0, 30.0],
                [20.0, 30.0, 40.0],
                [30.0, 40.0, 50.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [80.0, 90.0, 100.0],
                [90.0, 100.0, 110.0],
                [100.0, 110.0, 120.0],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_distance(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.continuous.multi_signals.MultiSignals.\
domain_distance` method.
        """

        multi_signals = MultiSignals(xp_as_array(self._range_2, xp=xp))

        xp_assert_close(
            multi_signals.domain_distance(0.5),
            0.5,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            multi_signals.domain_distance(np.linspace(0, 9, 10) + 0.5),
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_to_dataframe(self) -> None:
        """
        Test :meth:`colour.continuous.multi_signals.MultiSignals.to_dataframe`
        method.
        """

        if is_pandas_installed():
            from pandas import DataFrame  # noqa: PLC0415

            data = dict(zip(["a", "b", "c"], tsplit(self._range_2), strict=True))

            attest(
                MultiSignals(self._range_2, self._domain_2, labels=["a", "b", "c"])
                .to_dataframe()
                .equals(DataFrame(data, self._domain_2))
            )
