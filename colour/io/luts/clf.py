"""
Define functionality to execute and run CLF workflows.
"""
import colour_clf_io as clf
import numpy as np

from colour.algebra import (
    table_interpolation_tetrahedral,
    table_interpolation_trilinear,
)
from colour.hints import ArrayLike
from colour.io.luts import (
    LUT1D, LUT3D
)

__all__ = ["apply"]

import numpy.typing as npt


def from_uint16_to_f16(array: npt.NDArray[np.uint16]) -> npt.NDArray[np.float16]:
    values = list(map(int, array))
    array = np.array(values, dtype=np.uint16)
    array.dtype = np.float16  # type: ignore
    return array  # type: ignore


def from_f16_to_uint16(array: npt.NDArray[np.float16]) -> npt.NDArray[np.uint16]:
    array = np.array(array, dtype=np.float16)
    array.dtype = np.uint16  # type: ignore
    return array  # type: ignore

def get_interpolator_for_LUT3D(node: clf.LUT3D):
    if node.interpolation == node.interpolation.TRILINEAR:
        return table_interpolation_trilinear
    elif node.interpolation == node.interpolation.TETRAHEDRAL:
        return table_interpolation_tetrahedral
    else:
        raise NotImplementedError

def apply_LUT3D(node: clf.LUT3D, value: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    table = node.array.as_array()
    size = node.array.dim[0]
    if node.raw_halfs:
        table = from_uint16_to_f16(table)
    if node.half_domain:
        value = from_f16_to_uint16(value) / (size - 1)
    domain = np.min(table, axis=3, keepdims=True), np.max(table, axis=3, keepdims=True)
    # We need to map to indices, where 1 indicates the last element in the LUT array.
    value_scaled = value * (size - 1)
    extrapolator_kwargs = {"method": "Constant"}
    interpolator = get_interpolator_for_LUT3D(node)
    lut = LUT3D(table, size=size)
    return lut.apply(value_scaled, extrapolator_kwargs=extrapolator_kwargs, interpolator=interpolator)

def apply_LUT1D(
    node: clf.LUT1D, value: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    table = node.array.as_array()
    size = node.array.dim[0]
    if node.raw_halfs:
        table = from_uint16_to_f16(table)
    if node.half_domain:
        value = from_f16_to_uint16(value) / (size - 1)
    domain = np.min(table), np.max(table)
    # We need to map to indices, where 1 indicates the last element in the LUT array.
    value_scaled = value * (size - 1)
    lut = LUT1D(table, size=size, domain=domain)
    extrapolator_kwargs = {"method": "Constant"}
    return lut.apply(value_scaled, extrapolator_kwargs=extrapolator_kwargs)


def apply_proces_node(
    node: clf.ProcessNode, value: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    if isinstance(node, clf.LUT1D):
        return apply_LUT1D(node, value)
    if isinstance(node, clf.LUT3D):
        return apply_LUT3D(node, value)
    raise RuntimeError("No matching process node found")  # TODO: Better error handling


def apply_next_node(
    process_list: clf.ProcessList,
    value: npt.NDArray[np.float_],
    use_normalised_values: bool,
) -> npt.NDArray[np.float_]:
    next_node = process_list.process_nodes.pop(0)
    if not use_normalised_values:
        value = value / next_node.in_bit_depth.scale_factor()
    result = apply_proces_node(next_node, value)
    if use_normalised_values:
        result = result / next_node.out_bit_depth.scale_factor()
    return result


def apply(
    process_list: clf.ProcessList,
    value: npt.NDArray[np.float_],
    use_normalised_values=False,
) -> ArrayLike:
    """Apply the transformation described by the given ProcessList to the given
    value.
    """
    result = value
    while process_list.process_nodes:
        result = apply_next_node(process_list, result, use_normalised_values)
        use_normalised_values = False
    return result
