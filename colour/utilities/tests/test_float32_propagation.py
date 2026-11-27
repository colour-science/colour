"""
Define the float32 propagation tests.

With :attr:`colour.constants.DTYPE_FLOAT_DEFAULT` set to float32, definitions
must not compute float64 intermediates. The definitions are traced with
*JAX* and the float64 bindings of the resulting *jaxpr* are counted.
"""

from __future__ import annotations

import dataclasses
import math
import typing

import numpy as np
import pytest

import colour
from colour.graph.conversion import CONVERSION_SPECIFICATIONS
from colour.utilities import (
    array_api_enable,
    in_array,
    is_networkx_installed,
    row_as_diagonal,
    set_default_float_dtype,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from colour.hints import Any

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax_extend = pytest.importorskip("jax.extend")

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "NODES",
    "DEFINITIONS",
    "count_float64_bindings",
    "TestFloat32Propagation",
]


def _nodes_from_XYZ() -> list[str]:
    if not is_networkx_installed():
        return []

    import networkx as nx  # noqa: PLC0415

    graph = nx.DiGraph(
        (specification.source, specification.target)
        for specification in CONVERSION_SPECIFICATIONS
    )

    return sorted(nx.descendants(graph, "cie xyz"))


NODES: list[str] = _nodes_from_XYZ()
"""Conversion graph nodes reachable from *CIE XYZ*."""


def _f32(*values: float) -> Any:
    return jnp.asarray(np.array(values, dtype=np.float32))


DEFINITIONS: dict[str, Callable[[], Any]] = {
    "row_as_diagonal": lambda: row_as_diagonal(_f32(1, 2, 3)),
    "in_array": lambda: in_array(_f32(0.5, 0.6), _f32(0.5, 0.7)),
    "chromatic_adaptation_VonKries": (
        lambda: colour.adaptation.chromatic_adaptation_VonKries(
            _f32(0.2, 0.12, 0.05), _f32(0.95, 1, 1.09), _f32(0.96, 1, 0.82)
        )
    ),
    "matrix_chromatic_adaptation_vk20": (
        lambda: colour.adaptation.matrix_chromatic_adaptation_vk20(
            _f32(0.95, 1, 1.09), _f32(0.96, 1, 0.82), _f32(0.95, 1, 1.09)
        )
    ),
    "RGB_to_ICtCp HLG": lambda: colour.RGB_to_ICtCp(
        _f32(0.45, 0.3, 0.2), method="ITU-R BT.2100-2 HLG"
    ),
    "ICtCp_to_RGB HLG": lambda: colour.ICtCp_to_RGB(
        _f32(0.07, 0, 0.04), method="ITU-R BT.2100-2 HLG"
    ),
    "delta_E_DIN99": lambda: colour.difference.delta_E_DIN99(
        _f32(60.25, 2.68, 7.92), _f32(59.17, 2.82, 6.73)
    ),
}


def _arrays(result: Any) -> Any:
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return [value for value in dataclasses.astuple(result) if value is not None]

    return result


def count_float64_bindings(function: Callable[[], Any]) -> int:
    """
    Count the float64 bindings in the *jaxpr* of ``function``, or its float64
    outputs when it cannot be traced, e.g. host-bound *SciPy* paths.
    """

    def count(jaxpr: Any) -> int:
        total = 0
        for equation in jaxpr.eqns:
            for sub_jaxpr in jax_extend.core.jaxprs_in_params(equation.params):
                total += count(sub_jaxpr)
            # Size-0 arrays, e.g. the namespace probe of *array-api-compat*,
            # carry no computation.
            total += sum(
                str(getattr(variable.aval, "dtype", "")) == "float64"
                and math.prod(variable.aval.shape) > 0
                for variable in equation.outvars
            )
        return total

    with array_api_enable(True):
        try:
            return count(jax.make_jaxpr(lambda: _arrays(function()))().jaxpr)
        except Exception:  # noqa: BLE001
            outputs = jax.tree_util.tree_leaves(_arrays(function()))

    return sum(str(getattr(output, "dtype", "")) == "float64" for output in outputs)


@pytest.fixture
def float32() -> Generator[None, None, None]:
    set_default_float_dtype(np.float32)
    try:
        yield
    finally:
        set_default_float_dtype(np.float64)


@pytest.mark.usefixtures("float32")
class TestFloat32Propagation:
    """Define float32 propagation unit tests methods."""

    @pytest.mark.skipif(not is_networkx_installed(), reason="Requires NetworkX.")
    @pytest.mark.parametrize("node", NODES)
    def test_conversion_graph(self, node: str) -> None:
        """Test the conversions from *CIE XYZ* along the conversion graph."""

        XYZ = _f32(0.20654008, 0.12197225, 0.05136952)

        assert count_float64_bindings(lambda: colour.convert(XYZ, "CIE XYZ", node)) == 0

    @pytest.mark.parametrize("name", list(DEFINITIONS))
    def test_definitions(self, name: str) -> None:
        """Test definitions outside the conversion graph."""

        assert count_float64_bindings(DEFINITIONS[name]) == 0
