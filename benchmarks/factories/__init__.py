"""
Suite Factories
===============
"""


import colour.ndarray as np
import sys
from functools import partial

from colour.hints import Callable, List, Optional, Sequence

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Suite",
    "suites_factory",
    "I_suites_factory",
    "IJ_suites_factory",
    "IJK_suites_factory",
]


class Suite:
    """
    Define a benchmark suite for *Standard Definition*, *High Definition* and
    *Ultra High Definition* resolution data.
    """

    def time_sd(self):
        """
        Execute underlying callable on *Standard Definition* resolution data.
        """

        if hasattr(self, "_data_sd"):
            self._callable(self._data_sd)

    def time_hd(self):
        """
        Execute underlying callable on *High Definition* resolution data.
        """

        if hasattr(self, "_data_hd"):
            self._callable(self._data_hd)

    def time_uhd(self):
        """
        Execute underlying callable on *Ultra High Definition* resolution data.
        """

        if hasattr(self, "_data_uhd"):
            self._callable(self._data_uhd)


def suites_factory(
    callables: Sequence,
    module: str = __name__,
    data: Optional[Sequence[Callable]] = None,
) -> List[Suite]:
    """
    Produce a benchmark suites for given callables.

    Parameters
    ----------
    callables
        Callables to produce benchmark suites for.
    module
        Module to set the benchmark suites into.
    data
        *Standard Definition*, *High Definition* and *Ultra High Definition*
        resolution data

    Returns
    -------
    :class:`list`
        Benchmark suites
    """

    classes = []

    for callable_ in callables:
        class_ = type(callable_.__name__, (Suite,), {})

        class_._callable = staticmethod(callable_)

        class_._data_sd = data[0]
        class_._data_hd = data[1]
        class_._data_uhd = data[2]

        classes.append(classes)

        setattr(sys.modules[module], f"{callable_.__name__}_Suite", class_)

    return classes


np.random.seed(16)


I_suites_factory = partial(
    suites_factory,
    data=(
        np.random.random([720, 1280]),
        np.random.random([1080, 1920]),
        np.random.random([2160, 3840]),
    ),
)

IJ_suites_factory = partial(
    suites_factory,
    data=(
        np.random.random([720, 1280, 2]),
        np.random.random([1080, 1920, 2]),
        np.random.random([2160, 3840, 2]),
    ),
)

IJK_suites_factory = partial(
    suites_factory,
    data=(
        np.random.random([720, 1280, 3]),
        np.random.random([1080, 1920, 3]),
        np.random.random([2160, 3840, 3]),
    ),
)
