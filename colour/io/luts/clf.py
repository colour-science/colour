"""
Define functionality to execute and run CLF workflows.
"""
import colour_clf_io as clf
import numpy as np
from numpy.typing import ArrayLike, NDArray

from colour import LUTSequence
from colour.algebra import (
    table_interpolation_tetrahedral,
    table_interpolation_trilinear,
)
from colour.hints import (
    Any,
    NDArrayFloat,
    ProtocolLUTSequenceItem,
)
from colour.io import AbstractLUTSequenceOperator, luts
from colour.utilities import as_float_array, tsplit, tstack

__all__ = ["apply"]

import numpy.typing as npt
from colour_clf_io import ExponentStyle
from colour_clf_io.values import Channel

from colour.models.rgb.transfer_functions import (
    exponent_function_basic,
    exponent_function_monitor_curve,
    logarithmic_function_basic,
    logarithmic_function_camera,
    logarithmic_function_quasilog,
)


def from_uint16_to_f16(array: npt.NDArray[np.uint16]) -> npt.NDArray[np.float16]:
    values = list(map(int, array))
    array = np.array(values, dtype=np.uint16)
    array.dtype = np.float16  # type: ignore
    return array  # type: ignore


def from_f16_to_uint16(array: npt.NDArray[np.float16]) -> npt.NDArray[np.uint16]:
    array = np.array(array, dtype=np.float16)
    array.dtype = np.uint16  # type: ignore
    return array  # type: ignore


def apply_by_channel(value, f, params, extra_args=None) -> NDArray:
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


class CLFNode(AbstractLUTSequenceOperator):
    node: clf.ProcessNode

    def __init__(self, node: clf.ProcessNode):
        super().__init__(node.name, [node.description])
        self.node = node

    def from_input_range(self, value):
        return value

    def to_output_range(self, value):
        return value / self.node.out_bit_depth.scale_factor()


class LUT3D(CLFNode):
    node: clf.LUT3D

    def __init__(self, node: clf.LUT3D):
        super().__init__(node)
        self.node = node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        RGB = self.from_input_range(RGB)
        node = self.node
        table = node.array.as_array()
        size = node.array.dim[0]
        if node.raw_halfs:
            table = from_uint16_to_f16(table)
        if node.half_domain:
            RGB = np.array(RGB, dtype=np.float16)
            RGB = from_f16_to_uint16(RGB) / (size - 1)
        # We need to map to indices, where 1 indicates the last element in the
        # LUT array.
        value_scaled = RGB * (size - 1)
        extrapolator_kwargs = {"method": "Constant"}
        interpolator = get_interpolator_for_LUT3D(node)
        lut = luts.LUT3D(table, size=size)
        out = lut.apply(
            value_scaled,
            extrapolator_kwargs=extrapolator_kwargs,
            interpolator=interpolator,
        )
        out = self.to_output_range(out)
        return out


class LUT1D(CLFNode):
    node: clf.LUT1D

    def __init__(self, node: clf.LUT1D):
        super().__init__(node)
        self.node = node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        RGB = self.from_input_range(RGB)
        table = self.node.array.as_array()
        size = self.node.array.dim[0]
        if self.node.raw_halfs:
            table = from_uint16_to_f16(table)
        if self.node.half_domain:
            RGB = np.array(RGB, dtype=np.float16)
            RGB = from_f16_to_uint16(RGB) / (size - 1)
        domain = np.min(table), np.max(table)
        # We need to map to indices, where 1 indicates the last element in the
        # LUT array.
        value_scaled = RGB * (size - 1)
        lut = luts.LUT1D(table, size=size, domain=domain)
        extrapolator_kwargs = {"method": "Constant"}
        out = lut.apply(value_scaled, extrapolator_kwargs=extrapolator_kwargs)
        out = self.to_output_range(out)
        return out


class Matrix(CLFNode):
    node: clf.Matrix

    def __init__(self, node: clf.Matrix):
        super().__init__(node)
        self.node = node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        RGB = self.from_input_range(RGB)
        matrix = self.node.array.as_array()
        return matrix.dot(RGB)


def assert_range_correct(in_out, bit_depth_scale):
    if None not in in_out:
        expected_out_value = in_out[0] * bit_depth_scale
        if in_out[1] != expected_out_value:
            raise ValueError(
                f"Inconsistent settings in range node. "
                f"Input value was {in_out[1]}. "
                f"Expected output value to be {expected_out_value}, but got {in_out[1]}"
            )


class Range(CLFNode):
    node: clf.LUT1D

    def __init__(self, node: clf.LUT1D):
        super().__init__(node)
        self.node = node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        node = self.node
        value = RGB * self.node.in_bit_depth.scale_factor()
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
            out = np.clip(scaled_value, min_out, max_out)
        else:
            scale = (max_out - min_out) / (max_in - min_in)
            result = value * scale + min_out - min_in * scale
            if do_clamping:
                result = np.clip(result, min_out, max_out)
            out = result
        out = self.to_output_range(out)
        return out


FLT_MIN = 1.175494e-38


def apply_log_internal(value: NDArrayFloat, params, extra_args) -> NDArrayFloat:
    style, in_bit_depth, out_bit_depth = extra_args
    match style:
        case clf.LogStyle.LOG_10:
            return (
                logarithmic_function_basic(np.maximum(value, FLT_MIN), "log10")
                / out_bit_depth.scale_factor()
            )
        case clf.LogStyle.ANTI_LOG_10:
            return logarithmic_function_basic(np.maximum(value, FLT_MIN), "antiLog10")
        case clf.LogStyle.LOG_2:
            return logarithmic_function_basic(np.maximum(value, FLT_MIN), "log2")
        case clf.LogStyle.ANTI_LOG_2:
            return logarithmic_function_basic(np.maximum(value, FLT_MIN), "antiLog2")
        case clf.LogStyle.LIN_TO_LOG:
            log_side_slope = params.log_side_slope
            lin_side_slope = params.lin_side_slope
            log_side_offset = params.log_side_offset
            lin_side_offset = params.lin_side_offset
            base = params.base
            return logarithmic_function_quasilog(
                value,
                "linToLog",
                base,
                log_side_slope,
                lin_side_slope,
                log_side_offset,
                lin_side_offset,
            )
        case clf.LogStyle.LOG_TO_LIN:
            log_side_slope = params.log_side_slope
            lin_side_slope = params.lin_side_slope
            log_side_offset = params.log_side_offset
            lin_side_offset = params.lin_side_offset
            base = params.base
            return logarithmic_function_quasilog(
                value,
                "logToLin",
                base,
                log_side_slope,
                lin_side_slope,
                log_side_offset,
                lin_side_offset,
            )
        case clf.LogStyle.CAMERA_LIN_TO_LOG:
            log_side_slope = params.log_side_slope
            lin_side_slope = params.lin_side_slope
            log_side_offset = params.log_side_offset
            lin_side_offset = params.lin_side_offset
            lin_side_break = params.lin_side_break
            base = params.base
            linear_slope = params.linear_slope
            return logarithmic_function_camera(
                value,
                "cameraLinToLog",
                base,
                log_side_slope,
                lin_side_slope,
                log_side_offset,
                lin_side_offset,
                lin_side_break,
                linear_slope,
            )
        case clf.LogStyle.CAMERA_LOG_TO_LIN:
            log_side_slope = params.log_side_slope
            lin_side_slope = params.lin_side_slope
            log_side_offset = params.log_side_offset
            lin_side_offset = params.lin_side_offset
            lin_side_break = params.lin_side_break
            base = params.base
            linear_slope = params.linear_slope
            return logarithmic_function_camera(
                value,
                "cameraLogToLin",
                base,
                log_side_slope,
                lin_side_slope,
                log_side_offset,
                lin_side_offset,
                lin_side_break,
                linear_slope,
            )
        case _:
            raise ValueError(f"Invalid Log Style: {style}")


class Log(CLFNode):
    node: clf.Log

    def __init__(self, node: clf.Log):
        super().__init__(node)
        self.node = node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        RGB = self.from_input_range(RGB)
        node = self.node
        style = node.style
        params = node.log_params
        extra_args = style, node.in_bit_depth, node.out_bit_depth
        out = apply_by_channel(
            RGB,
            apply_log_internal,
            params,
            extra_args,
        )
        out = self.to_output_range(out)
        return out


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
            return exponent_function_basic(value, exponent, "basicFwd")
        case ExponentStyle.BASIC_REV:
            return exponent_function_basic(value, exponent, "basicRev")
        case ExponentStyle.BASIC_MIRROR_FWD:
            return exponent_function_basic(value, exponent, "basicMirrorFwd")
        case ExponentStyle.BASIC_MIRROR_REV:
            return exponent_function_basic(value, exponent, "basicMirrorRev")
        case ExponentStyle.BASIC_PASS_THRU_FWD:
            return exponent_function_basic(value, exponent, "basicPassThruFwd")
        case ExponentStyle.BASIC_PASS_THRU_REV:
            return exponent_function_basic(value, exponent, "basicPassThruRev")
        case ExponentStyle.MON_CURVE_FWD:
            return exponent_function_monitor_curve(
                value, exponent, offset, "monCurveFwd"
            )
        case ExponentStyle.MON_CURVE_REV:
            return exponent_function_monitor_curve(
                value, exponent, offset, "monCurveRev"
            )
        case ExponentStyle.MON_CURVE_MIRROR_FWD:
            return exponent_function_monitor_curve(
                value, exponent, offset, "monCurveMirrorFwd"
            )
        case ExponentStyle.MON_CURVE_MIRROR_REV:
            return exponent_function_monitor_curve(
                value, exponent, offset, "monCurveMirrorRev"
            )
        case _:
            raise ValueError(f"Invalid Exponent Style: {style}")


class Exponent(CLFNode):
    node: clf.Exponent

    def __init__(self, node: clf.Exponent):
        super().__init__(node)
        self.node = node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        node = self.node
        RGB = self.from_input_range(RGB)
        style = node.style
        params = node.exponent_params
        out = apply_by_channel(RGB, apply_exponent_internal, params, extra_args=style)
        out = self.to_output_range(out)
        return out


def asc_cdl_luma(value):
    # R, G, B = tsplit(value)
    # luma = 0.2126 * R + 0.7152 * G + 0.0722 * B
    weights = [0.2126, 0.7152, 0.0722]
    luma = np.sum(weights * value, axis=-1)
    return luma


class ASC_CDL(CLFNode):
    node: clf.ASC_CDL

    def __init__(self, node: clf.ASC_CDL):
        super().__init__(node)
        self.node = node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        node = self.node
        RGB = self.from_input_range(RGB)
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
                        RGB * slope + offset,
                    )
                    ** power
                )
                R, G, B = tsplit(out_sop)
                luma = asc_cdl_luma(out_sop)
                out = clamp(luma + saturation * (out_sop - luma))
            case clf.ASC_CDL_Style.FWD_NO_CLAMP:
                lin = as_float_array(RGB * slope + offset)
                out_sop = np.where(lin >= 0, lin**power, lin)
                luma = asc_cdl_luma(out_sop)
                out = luma + saturation * (out_sop - luma)
            case clf.ASC_CDL_Style.REV:
                in_clamp = clamp(RGB)
                luma = asc_cdl_luma(in_clamp)
                out_sat = luma + (in_clamp - luma) / saturation
                out = clamp((clamp(out_sat) ** (1.0 / power) - offset) / slope)
            case clf.ASC_CDL_Style.REV_NO_CLAMP:
                luma = asc_cdl_luma(RGB)
                out_sat = luma + (RGB - luma) / saturation
                out_pw = np.where(out_sat >= 0, (out_sat) ** (1 / power), out_sat)
                out = (out_pw - offset) / slope
            case _:
                raise ValueError(f"Invalid ASC_CDL Style: {node.style}")
        out = self.to_output_range(out)
        return out


def as_LUT_sequence_item(node: clf.ProcessNode) -> ProtocolLUTSequenceItem:
    if isinstance(node, clf.LUT1D):
        return LUT1D(node)
    if isinstance(node, clf.LUT3D):
        return LUT3D(node)
    if isinstance(node, clf.Matrix):
        return Matrix(node)
    if isinstance(node, clf.Range):
        return Range(node)
    if isinstance(node, clf.Log):
        return Log(node)
    if isinstance(node, clf.Exponent):
        return Exponent(node)
    if isinstance(node, clf.ASC_CDL):
        return ASC_CDL(node)
    raise RuntimeError(f"No matching process node found for {node}.")


def apply(
    process_list: clf.ProcessList,
    value: NDArrayFloat,
    normalised_values=False,
) -> NDArrayFloat:
    """Apply the transformation described by the given ProcessList to the given
    value.
    """
    if not normalised_values:
        value = value / process_list.process_nodes[0].in_bit_depth.scale_factor()

    lut_sequence_items = [
        as_LUT_sequence_item(node) for node in process_list.process_nodes
    ]
    sequence = LUTSequence(*lut_sequence_items)
    result = sequence.apply(value)

    if not normalised_values:
        result = result * process_list.process_nodes[-1].out_bit_depth.scale_factor()

    return result
