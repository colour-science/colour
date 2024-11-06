"""
Define functionality to execute and run CLF workflows.
"""
import colour_clf_io as clf
import numpy as np
import numpy.lib.scimath as emath

from colour.algebra import (
    table_interpolation_tetrahedral,
    table_interpolation_trilinear,
)
from colour.hints import (
    NDArrayFloat,
)
from colour.io.luts import LUT1D, LUT3D
from colour.utilities import tsplit, tstack

__all__ = ["apply"]

import numpy.typing as npt
from colour_clf_io import ExponentStyle
from colour_clf_io.values import Channel


def from_uint16_to_f16(array: npt.NDArray[np.uint16]) -> npt.NDArray[np.float16]:
    values = list(map(int, array))
    array = np.array(values, dtype=np.uint16)
    array.dtype = np.float16  # type: ignore
    return array  # type: ignore


def from_f16_to_uint16(array: npt.NDArray[np.float16]) -> npt.NDArray[np.uint16]:
    array = np.array(array, dtype=np.float16)
    array.dtype = np.uint16  # type: ignore
    return array  # type: ignore


def apply_by_channel(value, f, params, extra_args=None):
    if params is None or len(params) == 0:
        return f(value, params, extra_args)
    elif len(params) == 1 and params[0].channel is None:
        return f(value, params[0], extra_args)
    else:
        R, G, B = tsplit(value)
        for param in params:
            match param.channel:
                case Channel.R:
                    R = f(R, param, extra_args)
                case Channel.G:
                    G = f(G, param, extra_args)
                case Channel.B:
                    B = f(B, param, extra_args)
        return tstack([R, G, B])


def get_interpolator_for_LUT3D(node: clf.LUT3D):
    if node.interpolation == node.interpolation.TRILINEAR:
        return table_interpolation_trilinear
    elif node.interpolation == node.interpolation.TETRAHEDRAL:
        return table_interpolation_tetrahedral
    else:
        raise NotImplementedError


def apply_LUT3D(node: clf.LUT3D, normalised_value: NDArrayFloat) -> NDArrayFloat:
    table = node.array.as_array()
    size = node.array.dim[0]
    if node.raw_halfs:
        table = from_uint16_to_f16(table)
    if node.half_domain:
        normalised_value = np.array(normalised_value, dtype=np.float16)
        normalised_value = from_f16_to_uint16(normalised_value) / (size - 1)
    # We need to map to indices, where 1 indicates the last element in the LUT array.
    value_scaled = normalised_value * (size - 1)
    extrapolator_kwargs = {"method": "Constant"}
    interpolator = get_interpolator_for_LUT3D(node)
    lut = LUT3D(table, size=size)
    return lut.apply(
        value_scaled, extrapolator_kwargs=extrapolator_kwargs, interpolator=interpolator
    )


def apply_LUT1D(node: clf.LUT1D, normalised_value: NDArrayFloat) -> NDArrayFloat:
    table = node.array.as_array()
    size = node.array.dim[0]
    if node.raw_halfs:
        table = from_uint16_to_f16(table)
    if node.half_domain:
        normalised_value = np.array(normalised_value, dtype=np.float16)
        normalised_value = from_f16_to_uint16(normalised_value) / (size - 1)
    domain = np.min(table), np.max(table)
    # We need to map to indices, where 1 indicates the last element in the LUT array.
    value_scaled = normalised_value * (size - 1)
    lut = LUT1D(table, size=size, domain=domain)
    extrapolator_kwargs = {"method": "Constant"}
    return lut.apply(value_scaled, extrapolator_kwargs=extrapolator_kwargs)


def apply_matrix(node: clf.Matrix, value: NDArrayFloat) -> NDArrayFloat:
    matrix = node.array.as_array()
    return matrix.dot(value)


def assert_range_correct(in_out, bit_depth_scale):
    if None not in in_out:
        expected_out_value = in_out[0] * bit_depth_scale
        if in_out[1] != expected_out_value:
            raise ValueError(
                f"Inconsistent settings in range node. "
                f"Input value was {in_out[1]}. "
                f"Expected output value to be {expected_out_value}, but got {in_out[1]}"
            )


def apply_range(node: clf.Range, normalised_value: NDArrayFloat):
    value = normalised_value * node.in_bit_depth.scale_factor()
    max_in = node.max_in_value
    max_out = node.max_out_value
    max_in_out = node.max_in_value, node.max_out_value
    min_in = node.min_in_value
    min_out = node.min_out_value
    min_in_out = node.min_in_value, node.min_out_value
    do_clamping = node.style is None or node.style == node.style.CLAMP

    if None in max_in_out or None in min_in_out:
        if not do_clamping:
            raise ValueError(
                "Inconsistent settings in range node. "
                "Clamping was not set, but not all values to calculate a "
                "range are supplied. "
            )
        bit_depth_scale = (
            node.out_bit_depth.scale_factor() / node.in_bit_depth.scale_factor()
        )
        assert_range_correct(min_in_out, bit_depth_scale)
        assert_range_correct(max_in_out, bit_depth_scale)
        scaled_value = value * bit_depth_scale
        return np.clip(scaled_value, min_out, max_out)
    else:
        scale = (max_out - min_out) / (max_in - min_in)
        result = value * scale + min_out - min_in * scale
        if do_clamping:
            result = np.clip(result, min_out, max_out)
        return result


FLT_MIN = 1.175494e-38


def apply_log_internal(value: NDArrayFloat, params, extra_args) -> NDArrayFloat:
    style, in_bit_depth, out_bit_depth = extra_args
    match style:
        case clf.LogStyle.LOG_10:
            return np.log10(np.maximum(value, FLT_MIN)) / out_bit_depth.scale_factor()
        case clf.LogStyle.ANTI_LOG_10:
            return np.power(10, value)
        case clf.LogStyle.LOG_2:
            return np.log2(np.maximum(value, FLT_MIN))
        case clf.LogStyle.ANTI_LOG_2:
            return np.power(2, value)
        case clf.LogStyle.LIN_TO_LOG:
            log_side_slope = params.log_side_slope
            lin_side_slope = params.lin_side_slope
            log_side_offset = params.log_side_offset
            lin_side_offset = params.lin_side_offset
            base = params.base
            log_side_value = emath.logn(
                base,
                np.maximum(lin_side_slope * value + lin_side_offset, FLT_MIN),
            )
            return log_side_slope * log_side_value + log_side_offset
        case clf.LogStyle.LOG_TO_LIN:
            log_side_slope = params.log_side_slope
            lin_side_slope = params.lin_side_slope
            log_side_offset = params.log_side_offset
            lin_side_offset = params.lin_side_offset
            base = params.base
            lin_side_value = base ** ((value - log_side_offset) / log_side_slope)
            return (lin_side_value - lin_side_offset) / lin_side_slope
        case clf.LogStyle.CAMERA_LIN_TO_LOG:
            log_side_slope = params.log_side_slope
            lin_side_slope = params.lin_side_slope
            log_side_offset = params.log_side_offset
            lin_side_offset = params.lin_side_offset
            lin_side_break = params.lin_side_break
            base = params.base
            linear_slope = params.linear_slope
            if lin_side_slope is None:
                linear_slope = (
                    log_side_slope
                    * lin_side_slope
                    / (
                        (lin_side_slope * lin_side_break + lin_side_offset)
                        * np.log(base)
                    )
                )
            log_side_break = (
                log_side_slope
                * emath.logn(base, lin_side_slope * lin_side_break + lin_side_offset)
                + log_side_offset
            )
            linear_offset = log_side_break - linear_slope * lin_side_break
            return np.where(
                value < lin_side_break,
                linear_slope * value + linear_offset,
                log_side_slope
                * emath.logn(
                    base,
                    np.maximum(lin_side_slope * value + lin_side_offset, FLT_MIN),
                )
                + log_side_offset,
            )
        case clf.LogStyle.CAMERA_LOG_TO_LIN:
            log_side_slope = params.log_side_slope
            lin_side_slope = params.lin_side_slope
            log_side_offset = params.log_side_offset
            lin_side_offset = params.lin_side_offset
            lin_side_break = params.lin_side_break
            base = params.base
            linear_slope = params.linear_slope
            if lin_side_slope is None:
                linear_slope = (
                    log_side_slope
                    * lin_side_slope
                    / (
                        (lin_side_slope * lin_side_break + lin_side_offset)
                        * np.log(base)
                    )
                )
            log_side_break = (
                log_side_slope
                * emath.logn(base, lin_side_slope * lin_side_break + lin_side_offset)
                + log_side_offset
            )
            linear_offset = log_side_break - linear_slope * lin_side_break
            return np.where(
                value <= log_side_break,
                (value - linear_offset) / linear_slope,
                (
                    (base ** ((value - log_side_offset) / log_side_slope))
                    - lin_side_offset
                )
                / lin_side_slope,
            )
        case _:
            raise ValueError(f"Invalid Log Style: {style}")


def apply_log(node: clf.Log, normalised_value: NDArrayFloat) -> NDArrayFloat:
    style = node.style
    params = node.log_params
    extra_args = style, node.in_bit_depth, node.out_bit_depth
    return apply_by_channel(
        normalised_value,
        apply_log_internal,
        params,
        extra_args,
    )


def mon_curve_forward(x, exponent, offset):
    x_break = offset / (exponent - 1)
    s = ((exponent - 1) / offset) * (
        (offset * exponent) / ((exponent - 1) * (1 + offset))
    ) ** exponent
    return np.where(x >= x_break, ((x + offset) / (1 + offset)) ** exponent, x * s)


def mon_curve_reverse(x, exponent, offset):
    y_break = ((offset * exponent) / ((exponent - 1) * (1 + offset))) ** exponent
    s = ((exponent - 1) / offset) * (
        (offset * exponent) / ((exponent - 1) * (1 + offset))
    ) ** exponent
    return np.where(x >= y_break, (1 + offset) * x ** (1 / exponent) - offset, x / s)


def apply_exponent_internal(
    value: NDArrayFloat, params: clf.ExponentParams, extra_args
) -> NDArrayFloat:
    exponent = params.exponent
    offset = params.offset
    style = extra_args
    match style:
        case ExponentStyle.BASIC_FWD:
            return np.maximum(0.0, value) ** exponent
        case ExponentStyle.BASIC_REV:
            return np.maximum(0.0, value) ** (1 / exponent)
        case ExponentStyle.BASIC_MIRROR_FWD:
            return np.where(
                value >= 0,
                value**exponent,
                -((-value) ** exponent),  # TODO check for type in reference
            )
        case ExponentStyle.BASIC_MIRROR_REV:
            return np.where(
                value >= 0,
                value ** (1 / exponent),
                -((-value) ** (1 / exponent)),
            )
        case ExponentStyle.BASIC_PASS_THRU_FWD:
            return np.where(value >= 0, value**exponent, value)
        case ExponentStyle.BASIC_PASS_THRU_REV:
            return np.where(
                value >= 0,
                value ** (1 / exponent),
                value,
            )
        case ExponentStyle.MON_CURVE_FWD:
            return mon_curve_forward(value, exponent, offset)
        case ExponentStyle.MON_CURVE_REV:
            return mon_curve_reverse(value, exponent, offset)
        case ExponentStyle.MON_CURVE_MIRROR_FWD:
            return np.where(
                value >= 0,
                mon_curve_forward(value, exponent, offset),
                -mon_curve_forward(-value, exponent, offset),
            )
        case ExponentStyle.MON_CURVE_MIRROR_REV:
            return np.where(
                value >= 0,
                mon_curve_reverse(value, exponent, offset),
                -mon_curve_reverse(-value, exponent, offset),
            )
        case _:
            raise ValueError(f"Invalid Exponent Style: {style}")


def apply_exponent(node: clf.Exponent, normalised_value: NDArrayFloat) -> NDArrayFloat:
    style = node.style
    params = node.exponent_params
    return apply_by_channel(
        normalised_value, apply_exponent_internal, params, extra_args=style
    )


def asc_cdl_luma(value):
    R, G, B = tsplit(value)
    luma = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return luma


def apply_asc_cdl(node: clf.ASC_CDL, normalised_value: NDArrayFloat):
    sop = node.sopnode
    if sop is None:
        slope = np.array([1.0, 1.0, 1.0])
        offset = np.array([0.0, 0.0, 0.0])
        power = np.array([1.0, 1.0, 1.0])
    else:
        slope = np.array(sop.slope)
        offset = np.array(sop.offset)
        power = np.array(sop.power)
    saturation = 1.0 if node.sat_node is None else node.sat_node.saturation

    def clamp(x):
        return np.clip(x, 0.0, 1.0)

    match node.style:
        case clf.ASC_CDL_Style.FWD:
            out_sop = (
                clamp(
                    normalised_value * slope + offset,
                )
                ** power
            )
            R, G, B = tsplit(out_sop)
            luma = asc_cdl_luma(out_sop)
            return clamp(luma + saturation * (out_sop - luma))
        case clf.ASC_CDL_Style.FWD_NO_CLAMP:
            lin = normalised_value * slope + offset
            out_sop = np.where(lin >= 0, lin**power, lin)
            luma = asc_cdl_luma(out_sop)
            return luma + saturation * (out_sop - luma)
        case clf.ASC_CDL_Style.REV:
            in_clamp = clamp(normalised_value)
            luma = asc_cdl_luma(in_clamp)
            out_sat = luma + (in_clamp - luma) / saturation
            return clamp((clamp(out_sat) ** (1.0 / power) - offset) / slope)
        case clf.ASC_CDL_Style.REV_NO_CLAMP:
            luma = asc_cdl_luma(normalised_value)
            out_sat = luma + (normalised_value - luma) / saturation
            out_pw = np.where(out_sat >= 0, (out_sat) ** (1 / power), out_sat)
            return (out_pw - offset) / slope
        case _:
            raise ValueError(f"Invalid ASC_CDL Style: {node.style}")


def apply_proces_node(
    node: clf.ProcessNode, normalised_value: NDArrayFloat
) -> NDArrayFloat:
    if isinstance(node, clf.LUT1D):
        return apply_LUT1D(node, normalised_value)
    if isinstance(node, clf.LUT3D):
        return apply_LUT3D(node, normalised_value)
    if isinstance(node, clf.Matrix):
        return apply_matrix(node, normalised_value)
    if isinstance(node, clf.Range):
        return apply_range(node, normalised_value)
    if isinstance(node, clf.Log):
        return apply_log(node, normalised_value)
    if isinstance(node, clf.Exponent):
        return apply_exponent(node, normalised_value)
    if isinstance(node, clf.ASC_CDL):
        return apply_asc_cdl(node, normalised_value)

    raise RuntimeError("No matching process node found")  # TODO: Better error handling


def apply_next_node(
    process_list: clf.ProcessList,
    value: NDArrayFloat,
    use_normalised_values: bool,
) -> NDArrayFloat:
    next_node = process_list.process_nodes.pop(0)
    if not use_normalised_values:
        value = value / next_node.in_bit_depth.scale_factor()
    result = apply_proces_node(next_node, value)
    if use_normalised_values:
        result = result / next_node.out_bit_depth.scale_factor()
    return result


def apply(
    process_list: clf.ProcessList,
    value: NDArrayFloat,
    use_normalised_values=False,
) -> NDArrayFloat:
    """Apply the transformation described by the given ProcessList to the given
    value.
    """
    result = value
    while process_list.process_nodes:
        result = apply_next_node(process_list, result, use_normalised_values)
        use_normalised_values = False
    return result
