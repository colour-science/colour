"""
Define functionality to execute and run CLF workflows.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import cast

import colour_clf_io as clf
import numpy as np
from numpy.typing import ArrayLike, NDArray

from colour.algebra import (
    table_interpolation_tetrahedral,
    table_interpolation_trilinear,
)
from colour.hints import (
    Any,
    NDArrayFloat,
    ProtocolLUTSequenceItem,
)
from colour.io import luts
from colour.io.luts.operator import AbstractLUTSequenceOperator
from colour.io.luts.sequence import LUTSequence
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


class CLFExecutionError(Exception):
    """
    Exception raised when there is an issue with the execution of a CLF workflow.
    """


def from_uint16_to_f16(array: npt.NDArray[np.uint16]) -> npt.NDArray[np.float16]:
    """
    Convert a :class:`numpy.ndarray` with values of type :class:`numpy.uint16`  to
    values of :class:`numpy.float16` by reinterpreting the bit pattern of the stored
    numbers.

    Parameters
    ----------
    array
        Array to re-interpret.

    Returns
    -------
        :class:`npt.NDArray[np.float16]`
    """
    values = list(map(int, array))
    array = np.array(values, dtype=np.uint16)
    array.dtype = np.float16  # type: ignore
    return array  # type: ignore


def from_f16_to_uint16(array: npt.NDArray[np.float16]) -> npt.NDArray[np.uint16]:
    """
    Convert a :class:`numpy.ndarray` with values of type :class:`numpy.float16`  to
    values of :class:`numpy.uint16` by reinterpreting the bit pattern of the stored
    numbers.

    Parameters
    ----------
    array
        Array to re-interpret.

    Returns
    -------
        :class:`npt.NDArray[np.uint16]`
    """
    array = np.array(array, dtype=np.float16)
    array.dtype = np.uint16  # type: ignore
    return array  # type: ignore


def apply_by_channel(
    value: ArrayLike, f: Callable, params: None | list[Any], extra_args: Any = None
) -> NDArray:
    """
    Apply a callable to the separate colour channels in a numpy array.

    How the callable is applied is determined by the number of parameters supplied:
        (1) if *params* is empty or *None*, the callable is called once on the input
            array,
        (2) if  *params* contains exactly one element, the callable is called once
            on input array,
        (3) if *params* contain more elements, the callable is called for each element.
            In this last case it is expected that the elements in params contain a
            *channel* attribute of type :class:`colour_clf_io.Channel` that indicates
            which channel the parameters should be applied to. The input value is then
            split along the axis, and R,G and B and the callable is called on each
            channel with the respective parameter item.
     In addition to *extra_args* are always supplied to the callable as third argument.

    Parameters
    ----------
    value
        Array that the callable is applied to.
    f
        Callable that is applied to the input array. Should take three arguments:
        the input array, the parameters item and the extra arguments.
    params
        Optional list of parameters to pass to the callable.
    extra_args
        Additional arguments to pass to the callable.

    Returns
    -------
        :class:`npt.NDArray`
            Result of applying the callable to the input array.
    """
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
    """
    Return the interpolator for the given LUT3D instance. Translates from the *CLF IO*
    data to the Colour LUT interpolator.

    Parameters
    ----------
    node
        LUT 3D instance to check for its interpolation.

    Returns
    -------
    :class:`Callable`
        Callable compatible with the API for :func:`colour.io.luts.apply`.
    """
    if node.interpolation is None:
        return lambda x: x
    if node.interpolation == node.interpolation.TRILINEAR:
        return table_interpolation_trilinear
    if node.interpolation == node.interpolation.TETRAHEDRAL:
        return table_interpolation_tetrahedral
    raise NotImplementedError


class CLFNode(AbstractLUTSequenceOperator):
    """
    Define the base class for *CLF IO* LUTs.

    This is the class that will be inherited by the actual CLF node implementations.

    Parameters
    ----------
    node
        The :class:`colour_clf_io.ProcessNode` instance from which this LUT node is
        built. `

    Attributes
    ----------
    -   :attr:`~colour.io.luts.clf.CLFNode.node`
    """

    def __init__(self, node: clf.ProcessNode) -> None:
        super().__init__(node.name, node.description)

    @property
    @abstractmethod
    def node(self) -> clf.ProcessNode:
        """
        Getter property for the wrapped process node.

        Returns
        -------
        :class:`colour_clf_io.ProcessNode`
            Process node.
        """

    def _from_input_range(self, value: ArrayLike) -> NDArrayFloat:
        """
        Convert the input array to the appropriate format for the internal computations.

        Parameters
        ----------
        value
            Array to convert.

        Returns
        -------
            :class:`NDArrayFloat`

        """
        return cast(NDArrayFloat, value)

    def _to_output_range(self, value: NDArrayFloat) -> NDArrayFloat:
        """
        Convert the array from our internal computation format to the result format.

        Parameters
        ----------
        value
            Array to convert.

        Returns
        -------
            :class:`NDArrayFloat`

        """
        return value / self.node.out_bit_depth.scale_factor()


class LUT3D(CLFNode):
    """
    Define define the *LUT* operator based on the clf.LUT3D node.

    Parameters
    ----------
    node
        The :class:`colour_clf_io.LUT3D` instance from which this LUT node is built. `

    Attributes
    ----------
    -   :attr:`~colour_clf_io.LUT3D`

    Methods
    -------
    -   :meth:`~colour.io.luts.LUT3D.apply`
    """

    def __init__(self, node: clf.LUT3D) -> None:
        super().__init__(node)
        self._node = node

    @property
    def node(self) -> clf.LUT3D:
        return self._node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        RGB = self._from_input_range(RGB)
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
        return self._to_output_range(out)


def apply_lut1D_internal(node: clf.LUT1D, table: NDArray, RGB: NDArray) -> NDArray:
    size = node.array.dim[0]
    if node.raw_halfs:
        table = from_uint16_to_f16(table)
    if node.half_domain:
        RGB = np.array(RGB, dtype=np.float16)
        RGB = from_f16_to_uint16(RGB) / (size - 1)
    domain = np.min(table), np.max(table)
    # We need to map to indices, where 1 indicates the last element in the
    # LUT array.
    value_scaled = RGB * (size - 1)
    lut = luts.LUT1D(table, size=size, domain=domain)
    extrapolator_kwargs = {"method": "Constant"}
    return lut.apply(value_scaled, extrapolator_kwargs=extrapolator_kwargs)


class LUT1D(CLFNode):
    """
    Define define the *LUT* operator based on the clf.LUT1D node.

    Parameters
    ----------
    node
        The :class:`colour_clf_io.LUT1D` instance from which this LUT node is built. `

    Attributes
    ----------
    -   :attr:`~colour.io.luts.clf.CLFNode.node`

    Methods
    -------
    -   :meth:`~colour.io.luts.LUT1D.apply`
    """

    def __init__(self, node: clf.LUT1D) -> None:
        super().__init__(node)
        self._node = node

    @property
    def node(self) -> clf.LUT1D:
        return self._node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        RGB = self._from_input_range(RGB)
        table = self.node.array.as_array()
        if len(table.shape) > 1:
            table_r, table_g, table_b = tsplit(table)
            R, G, B = tsplit(RGB)
            out_r = apply_lut1D_internal(self.node, table_r, R)
            out_g = apply_lut1D_internal(self.node, table_g, G)
            out_b = apply_lut1D_internal(self.node, table_b, B)
            result = tstack([out_r, out_g, out_b])
        else:
            result = apply_lut1D_internal(self.node, table, RGB)

        return self._to_output_range(result)


class Matrix(CLFNode):
    """
    Define define the *LUT* operator based on the clf.Matrix node.

    Parameters
    ----------
    node
        The :class:`colour_clf_io.Matrix` instance from which this LUT node is built. `

    Attributes
    ----------
    -   :attr:`~colour_clf_io.Matrix`

    Methods
    -------
    -   :meth:`~colour.io.luts.Matrix.apply`
    """

    def __init__(self, node: clf.Matrix) -> None:
        super().__init__(node)
        self._node = node

    @property
    def node(self) -> clf.Matrix:
        return self._node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        RGB = self._from_input_range(RGB)
        matrix = self.node.array.as_array()
        return matrix.dot(RGB)


def assert_range_correct(
    in_out: tuple[float | None, float | None], bit_depth_scale: float
) -> None:
    """
    Assert the input and output ranges are consistent.
    """
    if None not in in_out:
        in_out = cast(tuple[float, float], in_out)
        expected_out_value = in_out[0] * bit_depth_scale
        if in_out[1] != expected_out_value:
            message = (
                f"Inconsistent settings in range node. "
                f"Input value was {in_out[1]}. "
                f"Expected output value to be {expected_out_value}, but got {in_out[1]}"
            )
            raise CLFExecutionError(message)


class Range(CLFNode):
    """
    Define define the *LUT* operator based on the clf.Range node.

    Parameters
    ----------
    node
        The :class:`colour_clf_io.Range` instance from which this LUT node is built. `

    Attributes
    ----------
    -   :attr:`~colour_clf_io.Range`

    Methods
    -------
    -   :meth:`~colour.io.luts.Range.apply`
    """

    def __init__(self, node: clf.Range) -> None:
        super().__init__(node)
        self._node = node

    @property
    def node(self) -> clf.Range:
        return self._node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        node = self.node
        RGB = self._from_input_range(RGB)
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
                raise CLFExecutionError(message)
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
        return self._to_output_range(out)


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
                raise CLFExecutionError(err)
            linear_slope = params.linear_slope
            if linear_slope is None:
                err = (
                    """"The `linearSlope` This is required if style="cameraLinToLog"."""
                )
                raise CLFExecutionError(err)
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
                raise CLFExecutionError(err)
            linear_slope = params.linear_slope
            if linear_slope is None:
                err = """"The `linearSlope` This is required if "cameraLogToLin"""
                raise CLFExecutionError(err)
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
            raise CLFExecutionError(message)


class Log(CLFNode):
    """
    Define define the *LUT* operator based on the clf.Log node.

    Parameters
    ----------
    node
        The :class:`colour_clf_io.Log` instance from which this LUT node is built. `

    Attributes
    ----------
    -   :attr:`~colour_clf_io.Log`

    Methods
    -------
    -   :meth:`~colour.io.luts.Log.apply`
    """

    def __init__(self, node: clf.Log) -> None:
        super().__init__(node)
        self._node = node

    @property
    def node(self) -> clf.Log:
        return self._node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        RGB = self._from_input_range(RGB)
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
        return self._to_output_range(out)


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
            raise CLFExecutionError(message)


class Exponent(CLFNode):
    """
    Define define the *LUT* operator based on the clf.Exponent node.

    Parameters
    ----------
    node
        The :class:`colour_clf_io.Exponent` instance from which this LUT node is built.

    Attributes
    ----------
    -   :attr:`~colour_clf_io.Exponent`

    Methods
    -------
    -   :meth:`~colour.io.luts.Exponent.apply`
    """

    def __init__(self, node: clf.Exponent) -> None:
        super().__init__(node)
        self._node = node

    @property
    def node(self) -> clf.Exponent:
        return self._node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        node = self.node
        RGB = self._from_input_range(RGB)
        style = node.style
        params = node.exponent_params
        out = apply_by_channel(RGB, apply_exponent_internal, params, extra_args=style)
        return self._to_output_range(out)


def asc_cdl_luma(value: NDArrayFloat) -> NDArrayFloat:
    weights = [0.2126, 0.7152, 0.0722]
    return np.sum(weights * value, axis=-1)


class ASC_CDL(CLFNode):
    """
    Define define the *LUT* operator based on the clf.ASC_CDL node.

    Parameters
    ----------
    node
        The :class:`colour_clf_io.ASC_CDL` instance from which this LUT node is built. `

    Attributes
    ----------
    -   :attr:`~colour_clf_io.ASC_CDL`

    Methods
    -------
    -   :meth:`~colour.io.luts.ASC_CDL.apply`
    """

    def __init__(self, node: clf.ASC_CDL) -> None:
        super().__init__(node)
        self._node = node

    @property
    def node(self) -> clf.ASC_CDL:
        return self._node

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: ARG002
        node = self.node
        RGB = self._from_input_range(RGB)
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
                raise CLFExecutionError(message)
        return self._to_output_range(out)


def as_LUT_sequence_item(  # noqa: PLR0911
    node: clf.ProcessNode,
) -> ProtocolLUTSequenceItem:
    """
    Return the corresponding LUT sequence item for the given CLF node.

    Parameters
    ----------
    node
        Node to convert.

    Returns
    -------
    :class:`ProtocolLUTSequenceItem`

    Raises
    ------
    :class:`RuntimeError`
        If there exists no corresponding LUT sequence item for the given CLF node.

    """
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
    """
    Apply the transformation described by the given :class:`colour_clf_io.ProcessList`
    to the given value.

    Parameters
    ----------
        process_list
            The :class:`colour_clf_io.ProcessList` instance to apply.
        value
            Input value in the form of an array. Shape and data format need to be
            compatible with the given `process_list`.
        normalised_values
            Indicates whether the input values are normalised to the range 0..1. If
            this is the case, the range will be expanded to the input range expected
            by the given `process_list`.

    Returns
    -------
    :class:`NDArrayFloat`
        Result of applying the given :class:`colour_clf_io.ProcessList`.

    Raises
    ------
    :class:`CLFExecutionError`
        If the given *process_list* is invalid according to the CLF specification
        (e.g., missing or invalid parameters).
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
