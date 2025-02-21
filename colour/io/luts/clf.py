"""
Define functionality to execute and run CLF workflows.
"""

from collections.abc import Callable
from typing import cast

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
from colour.utilities import required


def from_uint16_to_f16(array: npt.NDArray[np.uint16]) -> npt.NDArray[np.float16]:
    values = list(map(int, array))
    array = np.array(values, dtype=np.uint16)
    array.dtype = np.float16  # type: ignore
    return array  # type: ignore


def from_f16_to_uint16(array: npt.NDArray[np.float16]) -> npt.NDArray[np.uint16]:
    array = np.array(array, dtype=np.float16)
    array.dtype = np.uint16  # type: ignore
    return array  # type: ignore


def apply_by_channel(
    value: ArrayLike, f: Callable, params: Any, extra_args: Any = None
) -> NDArray:
    if params is None or len(params) == 0:
        return f(value, None, extra_args)
    if len(params) == 1 and params[0].channel is None:
        return f(value, params[0], extra_args)
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


def get_interpolator_for_LUT3D(
    node: clf.LUT3D,
) -> Callable:
    if node.interpolation is None:
        return lambda x: x
    if node.interpolation == node.interpolation.TRILINEAR:
        return table_interpolation_trilinear
    if node.interpolation == node.interpolation.TETRAHEDRAL:
        return table_interpolation_tetrahedral
    raise NotImplementedError


class CLFNode(AbstractLUTSequenceOperator):
    node: clf.ProcessNode

    def __init__(self, node: clf.ProcessNode) -> None:
        super().__init__(node.name, node.description)
        self.node = node

    def from_input_range(self, value: ArrayLike) -> NDArrayFloat:
        return cast(NDArrayFloat, value)

    def to_output_range(self, value: NDArrayFloat) -> NDArrayFloat:
        return value / self.node.out_bit_depth.scale_factor()


class LUT3D(CLFNode):
    node: clf.LUT3D

    def __init__(self, node: clf.LUT3D) -> None:
        super().__init__(node)
        self.node = node  # type: ignore

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
        return self.to_output_range(out)


class LUT1D(CLFNode):
    node: clf.LUT1D

    def __init__(self, node: clf.LUT1D) -> None:
        super().__init__(node)
        self.node = node  # type: ignore

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
        return self.to_output_range(out)


class Matrix(CLFNode):
    node: clf.Matrix

    def __init__(self, node: clf.Matrix) -> None:
        super().__init__(node)
        self.node = node  # type: ignore

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        RGB = self.from_input_range(RGB)
        matrix = self.node.array.as_array()
        return matrix.dot(RGB)


def assert_range_correct(
    in_out: tuple[float | None, float | None], bit_depth_scale: float
) -> None:
    if None not in in_out:
        in_out = cast(tuple[float, float], in_out)
        expected_out_value = in_out[0] * bit_depth_scale
        if in_out[1] != expected_out_value:
            message = (
                f"Inconsistent settings in range node. "
                f"Input value was {in_out[1]}. "
                f"Expected output value to be {expected_out_value}, but got {in_out[1]}"
            )
            raise ValueError(message)


class Range(CLFNode):
    node: clf.Range

    def __init__(self, node: clf.Range) -> None:
        super().__init__(node)
        self.node = node  # type: ignore

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        node = self.node
        RGB = self.from_input_range(RGB)
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
                message = (
                    "Inconsistent settings in range node. "
                    "Clamping was not set, but not all values to calculate a "
                    "range are supplied. "
                )
                raise ValueError(message)
            bit_depth_scale = (
                node.out_bit_depth.scale_factor() / node.in_bit_depth.scale_factor()
            )
            assert_range_correct(min_in_out, bit_depth_scale)
            assert_range_correct(max_in_out, bit_depth_scale)
            scaled_value = value * bit_depth_scale
            out = np.clip(scaled_value, min_out, max_out)
        else:
            assert max_out is not None  # noqa: S101
            assert min_out is not None  # noqa: S101
            assert max_in is not None  # noqa: S101
            assert min_in is not None  # noqa: S101
            scale = (max_out - min_out) / (max_in - min_in)
            result = value * scale + min_out - min_in * scale
            if do_clamping:
                result = np.clip(result, min_out, max_out)
            out = result
        return self.to_output_range(out)


FLT_MIN = 1.175494e-38


def apply_log_internal(  # noqa: PLR0911
    value: NDArrayFloat, params: clf.LogParams, extra_args: Any
) -> NDArrayFloat:
    style, in_bit_depth, out_bit_depth = extra_args

    params = params if params is not None else clf.LogParams.default()
    base = params.base if params.base is not None else clf.LogParams.default().base
    assert base is not None  # noqa: S101
    base = int(base)
    log_side_slope = (
        params.log_side_slope
        if params.log_side_slope is not None
        else clf.LogParams.default().log_side_slope
    )
    assert log_side_slope is not None  # noqa: S101
    lin_side_slope = (
        params.lin_side_slope
        if params.lin_side_slope is not None
        else clf.LogParams.default().lin_side_slope
    )
    assert lin_side_slope is not None  # noqa: S101
    log_side_offset = (
        params.log_side_offset
        if params.log_side_offset is not None
        else clf.LogParams.default().log_side_offset
    )
    assert log_side_offset is not None  # noqa: S101
    lin_side_offset = (
        params.lin_side_offset
        if params.lin_side_offset is not None
        else clf.LogParams.default().lin_side_offset
    )
    assert lin_side_offset is not None  # noqa: S101
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
            lin_side_break = params.lin_side_break
            if lin_side_break is None:
                err = """"The `linSideBreak` This is required if
                style="cameraLinToLog"."""
                raise ValueError(err)
            linear_slope = params.linear_slope
            if linear_slope is None:
                err = (
                    """"The `linearSlope` This is required if style="cameraLinToLog"."""
                )
                raise ValueError(err)
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
            lin_side_break = params.lin_side_break
            if lin_side_break is None:
                err = """"The `linSideBreak` This is required if "cameraLogToLin"""
                raise ValueError(err)
            linear_slope = params.linear_slope
            if linear_slope is None:
                err = """"The `linearSlope` This is required if "cameraLogToLin"""
                raise ValueError(err)
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
            message = f"Invalid Log Style: {style}"
            raise ValueError(message)


class Log(CLFNode):
    node: clf.Log

    def __init__(self, node: clf.Log) -> None:
        super().__init__(node)
        self.node = node  # type: ignore

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
        return self.to_output_range(out)


def mon_curve_forward(x: NDArrayFloat, exponent: float, offset: float) -> NDArrayFloat:
    x_break = offset / (exponent - 1)
    s = ((exponent - 1) / offset) * (
        (offset * exponent) / ((exponent - 1) * (1 + offset))
    ) ** exponent
    return np.where(x >= x_break, ((x + offset) / (1 + offset)) ** exponent, x * s)


def mon_curve_reverse(x: NDArrayFloat, exponent: float, offset: float) -> NDArrayFloat:
    y_break = ((offset * exponent) / ((exponent - 1) * (1 + offset))) ** exponent
    s = ((exponent - 1) / offset) * (
        (offset * exponent) / ((exponent - 1) * (1 + offset))
    ) ** exponent
    return np.where(x >= y_break, (1 + offset) * x ** (1 / exponent) - offset, x / s)


def apply_exponent_internal(  # noqa: PLR0911
    value: NDArrayFloat, params: clf.ExponentParams, extra_args: Any
) -> NDArrayFloat:
    exponent = (
        params.exponent
        if params.exponent is not None
        else clf.ExponentParams.default().exponent
    )
    assert exponent is not None  # noqa: S101
    offset = (
        params.offset
        if params.offset is not None
        else clf.ExponentParams.default().offset
    )
    assert offset is not None  # noqa: S101
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
            message = f"Invalid Exponent Style: {style}"
            raise ValueError(message)


class Exponent(CLFNode):
    node: clf.Exponent

    def __init__(self, node: clf.Exponent) -> None:
        super().__init__(node)
        self.node = node  # type: ignore

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        node = self.node
        RGB = self.from_input_range(RGB)
        style = node.style
        params = node.exponent_params
        out = apply_by_channel(RGB, apply_exponent_internal, params, extra_args=style)
        return self.to_output_range(out)


def asc_cdl_luma(value: NDArrayFloat) -> NDArrayFloat:
    weights = [0.2126, 0.7152, 0.0722]
    return np.sum(weights * value, axis=-1)


class ASC_CDL(CLFNode):
    node: clf.ASC_CDL

    def __init__(self, node: clf.ASC_CDL) -> None:
        super().__init__(node)
        self.node = node  # type: ignore

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

        def clamp(x: NDArrayFloat) -> NDArrayFloat:
            return np.clip(x, 0.0, 1.0)

        match node.style:
            case clf.ASC_CDLStyle.FWD:
                value: NDArrayFloat = RGB  # Needed to satisfy pywright,
                out_sop = clamp(value * slope + offset) ** power
                luma = asc_cdl_luma(out_sop)
                out = clamp(luma + saturation * (out_sop - luma))
            case clf.ASC_CDLStyle.FWD_NO_CLAMP:
                lin = as_float_array(RGB * slope + offset)
                out_sop = np.where(lin >= 0, lin**power, lin)
                luma = asc_cdl_luma(out_sop)
                out = luma + saturation * (out_sop - luma)
            case clf.ASC_CDLStyle.REV:
                in_clamp = clamp(RGB)
                luma = asc_cdl_luma(in_clamp)
                out_sat = luma + (in_clamp - luma) / saturation
                out = clamp((clamp(out_sat) ** (1.0 / power) - offset) / slope)
            case clf.ASC_CDLStyle.REV_NO_CLAMP:
                luma = asc_cdl_luma(RGB)
                out_sat = luma + (RGB - luma) / saturation
                out_pw = np.where(out_sat >= 0, (out_sat) ** (1 / power), out_sat)
                out = (out_pw - offset) / slope
            case _:
                message = f"Invalid ASC_CDL Style: {node.style}"
                raise ValueError(message)
        return self.to_output_range(out)


def as_LUT_sequence_item(  # noqa: PLR0911
    node: clf.ProcessNode,
) -> ProtocolLUTSequenceItem:
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
    message = f"No matching process node found for {node}."
    raise RuntimeError(message)


@required("colour_clf_io")
def apply(
    process_list: clf.ProcessList,
    value: NDArrayFloat,
    normalised_values: bool = False,
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
