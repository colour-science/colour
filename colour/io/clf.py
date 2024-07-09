"""
CLF Parsing
===========

Defines the functionality and data structures to  parse CLF documents.

The main functionality is exposed through the following two methods:
-   :func:`colour.io.clf.read_clf`: Read a file in the CLF format and return the
    corresponding :class: ProcessList.
-   :func:`colour.io.clf.parse_clf`: Read a string that contains a CLF document and
    return the corresponding :class: ProcessList.

References
----------
-   :cite:`CLFv3` : Common LUT Format (CLF) - A Common File Format for Look-Up Tables.
    Retrieved May 1st, 2024, from https://docs.acescentral.com/specifications/clf
"""

import collections
import enum
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import Callable, Optional

import lxml.etree

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Array",
    "ASC_CDL",
    "ASC_CDL_Style",
    "BitDepth",
    "CalibrationInfo",
    "Channel",
    "CLFValidationError",
    "Exponent",
    "ExponentParams",
    "ExponentStyle",
    "Info",
    "Interpolation1D",
    "Interpolation3D",
    "Log",
    "LogParams",
    "LogStyle",
    "LUT1D",
    "LUT3D",
    "Matrix",
    "parse_clf",
    "ProcessList",
    "ProcessList",
    "Range",
    "RangeStyle",
    "read_clf",
    "SatNode",
    "SOPNode",
]

NAMESPACE_NAME = "urn:NATAS:AMPAS:LUT:v2.0"
CLF_NAMESPACE = {"clf": NAMESPACE_NAME}


def fully_qualified_name(name):
    return f"{{{NAMESPACE_NAME}}}{name}"


def map_optional(f: Callable, value):
    if value is not None:
        return f(value)
    return None


def retrieve_attributes(
    xml, attribute_mapping: dict[str, str]
) -> dict[str, Optional[str]]:
    def get_attribute(name):
        return xml.get(name)

    return {
        k: get_attribute(attribute_name)
        for k, attribute_name in attribute_mapping.items()
    }


def retrieve_attributes_as_float(
    xml, attribute_mapping: dict[str, str]
) -> dict[str, Optional[float]]:
    attributes = retrieve_attributes(xml, attribute_mapping)

    def as_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    return {key: as_float(value) for key, value in attributes.items()}


@dataclass
class Array:
    """
    Represents an Array element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#array
    """

    values: list[float]
    dim: tuple[int, ...]

    @staticmethod
    def from_xml(xml):
        """
        Parse and return the Array from the given XML node. Returns None if the given
        element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.

        """
        if xml is None:
            return None
        dim = tuple(map(int, xml.get("dim").split()))
        values = list(map(float, xml.text.split()))
        return Array(values=values, dim=dim)

    def as_array(self):
        """
        Convert the CLF element into a numpy array.

        Returns
        -------
        :class:`numpy.ndarray`
            Array of shape `dim` with the data from `values`.
        """
        import numpy as np

        return np.array(self.values).reshape(self.dim)


class Interpolation1D(Enum):
    """
    Represents the valid interpolation values of a LUT1D element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#lut1d
    """

    LINEAR = "linear"


class Interpolation3D(Enum):
    """
    Represents the valid interpolation values of a LUT3D element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#lut3d
    """

    TRILINEAR = "trilinear"
    TETRAHEDRAL = "tetrahedral"


class BitDepth(Enum):
    """
    Represents the valid bit depth values of the CLF specification.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#processNode
    """

    i8 = "8i"
    i10 = "10i"
    i12 = "12i"
    i16 = "16i"
    f16 = "16f"
    f32 = "32f"


@dataclass
class CalibrationInfo:
    """
    Represents a Calibration Info element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#processlist
    """

    display_device_serial_num: Optional[str]
    display_device_host_name: Optional[str]
    operator_name: Optional[str]
    calibration_date_time: Optional[str]
    measurement_probe: Optional[str]
    calibration_software_name: Optional[str]
    calibration_software_version: Optional[str]

    @staticmethod
    def from_xml(xml):
        """
        Parse and return the Calibration Info from the given XML node. Returns None
        if the given element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.

        """
        if xml is None:
            return None
        attributes = retrieve_attributes(
            xml,
            {
                "display_device_serial_num": "DisplayDeviceSerialNum",
                "display_device_host_name": "DisplayDeviceHostName",
                "operator_name": "OperatorName",
                "calibration_date_time": "CalibrationDateTime",
                "measurement_probe": "MeasurementProbe",
                "calibration_software_name": "CalibrationSoftwareName",
                "calibration_software_version": "CalibrationSoftwareVersion",
            },
        )
        return CalibrationInfo(**attributes)


@dataclass
class Info:
    """
    Represents a Info element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#processList
    """

    app_release: Optional[str]
    copyright: Optional[str]
    revision: Optional[str]
    aces_transform_id: Optional[str]
    aces_user_name: Optional[str]
    calibration_info: Optional[CalibrationInfo]

    @staticmethod
    def from_xml(xml):
        """
        Parse and return the Info from the given XML node. Returns None if the given
        element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.

        """
        if xml is None:
            return None
        attributes = retrieve_attributes(
            xml,
            {
                "app_release": "AppRelease",
                "copyright": "Copyright",
                "revision": "Revision",
                "aces_transform_id": "ACEStransformID",
                "aces_user_name": "ACESuserName",
            },
        )
        calibration_info = CalibrationInfo.from_xml(
            child_element(xml, "CalibrationInfo")
        )
        return Info(calibration_info=calibration_info, **attributes)


class CLFValidationError(Exception):
    """
    Indicates an error with parsing a CLF document..
    """


def must_have(value, message):
    if value is None:
        raise CLFValidationError(message)


def child_element(xml, name, xpath_function=""):
    elements = xml.xpath(f"clf:{name}{xpath_function}", namespaces=CLF_NAMESPACE)
    element_count = len(elements)
    match element_count:
        case 0:
            return None
        case 1:
            return elements[0]
        case _:
            raise CLFValidationError(
                f"Found multiple elements of type {name} in "
                f"element {xml}, but only expected exactly one."
            )


def element_as_text(xml, name):
    return child_element(xml, name, xpath_function="/text()")


def elements_as_text_list(xml, name):
    return xml.xpath(f"clf:{name}/text()", namespaces=CLF_NAMESPACE)


processing_node_constructors = {}


def register_process_node_xml_constructor(name):
    def register(constructor):
        processing_node_constructors[name] = constructor
        return constructor

    return register


def valid_processing_node_tags():
    return processing_node_constructors.values()


@dataclass
class ProcessNode:
    """
    Represents the common data of all Process Node elements.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#processNode
    """

    id: Optional[str]
    name: Optional[str]
    in_bit_depth: Optional[BitDepth]
    out_bit_depth: Optional[BitDepth]
    description: Optional[str]

    @staticmethod
    def parse_attributes(xml):
        attributes = retrieve_attributes(
            xml,
            {
                "id": "id",
                "name": "name",
            },
        )
        in_bit_depth = BitDepth(xml.get("inBitDepth"))
        out_bit_depth = BitDepth(xml.get("outBitDepth"))
        description = element_as_text(xml, "Description")
        args = {
            "in_bit_depth": in_bit_depth,
            "out_bit_depth": out_bit_depth,
            "description": description,
            **attributes,
        }
        return args


def parse_process_node(xml):
    """
    Return the correct process node that corresponds to this XML element.

    Returns
    -------
    :class: colour.clf.ProcessNode
        A subclass of `ProcessNode` that represents the given Process Node.

    Raises
    ------
    :class: CLFValidationError
        If the given element does not match any valid process node, or the node does not
        correctly correspond to the specification..

    """
    tag = lxml.etree.QName(xml).localname
    constructor = processing_node_constructors.get(tag)
    if constructor is not None:
        return processing_node_constructors[tag](xml)
    raise CLFValidationError(
        f"Encountered invalid processing node with tag '{xml.tag}'"
    )


@dataclass
class ProcessList:
    """
    Represents a Profess List element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#processList
    """

    id: str
    compatible_CLF_version: str
    process_nodes: list[ProcessNode]

    name: Optional[str]
    inverse_of: Optional[str]

    description: list[str]
    input_descriptor: Optional[str]
    output_descriptor: Optional[str]

    info: Optional[Info]

    @staticmethod
    def from_xml(xml):
        """
        Parse and return the Process List from the given XML node. Returns None if the
        given element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.

        """
        if xml is None:
            return None
        id = xml.get("id")  # noqa: A001
        must_have(id, "ProcessList must contain an `id` attribute")
        compatible_clf_version = xml.get("compCLFversion")
        must_have(
            compatible_clf_version,
            "ProcessList must contain an `compCLFversion` attribute",
        )

        name = xml.get("name")
        inverse_of = xml.get("inverseOf")
        info = Info.from_xml(xml)

        description = elements_as_text_list(xml, "Description")
        input_descriptor = element_as_text(xml, "InputDescriptor")
        output_descriptor = element_as_text(xml, "OutputDescriptor")

        ignore_nodes = ["Description", "InputDescriptor", "OutputDescriptor", "Info"]
        process_nodes = filter(
            lambda node: lxml.etree.QName(node).localname not in ignore_nodes, xml
        )
        process_nodes = [parse_process_node(xml_node) for xml_node in process_nodes]
        check_bit_depth_compatibility(process_nodes)

        return ProcessList(
            id=id,
            compatible_CLF_version=compatible_clf_version,
            process_nodes=process_nodes,
            name=name,
            inverse_of=inverse_of,
            input_descriptor=input_descriptor,
            output_descriptor=output_descriptor,
            info=info,
            description=description,
        )


def sliding_window(iterable, n):
    """
    Collect data into overlapping fixed-length chunks or blocks.
    Source: https://docs.python.org/3/library/itertools.html
    """
    it = iter(iterable)
    window = collections.deque(islice(it, n - 1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)


def check_bit_depth_compatibility(process_nodes: list[ProcessNode]):
    for node_a, node_b in sliding_window(process_nodes, 2):
        is_compatible = node_a.out_bit_depth == node_b.in_bit_depth
        if not is_compatible:
            raise CLFValidationError(
                f"Encountered incompatible bit depth between two processing nodes: "
                f"{node_a} and {node_b}"
            )


@dataclass
class LUT1D(ProcessNode):
    """
    Represents a LUT1D element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#lut1d
    """

    array: Array
    half_domain: bool
    raw_halfs: bool
    interpolation: Optional[Interpolation1D]

    @staticmethod
    @register_process_node_xml_constructor("LUT1D")
    def from_xml(xml):
        """
        Parse and return the LUT1D from the given XML node. Returns None if the given
        element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.
        """
        if xml is None:
            return None
        super_args = ProcessNode.parse_attributes(xml)
        array = Array.from_xml(child_element(xml, "Array"))
        half_domain = xml.get("halfDomain") == "true"
        raw_halfs = xml.get("rawHalfs") == "true"
        interpolation = map_optional(Interpolation1D, xml.get("interpolation"))
        return LUT1D(
            array=array,
            half_domain=half_domain,
            raw_halfs=raw_halfs,
            interpolation=interpolation,
            **super_args,
        )


@dataclass
class LUT3D(ProcessNode):
    """
    Represents a LUT3D element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#lut3d
    """

    array: Array
    half_domain: bool
    raw_halfs: bool
    interpolation: Optional[Interpolation3D]

    @staticmethod
    @register_process_node_xml_constructor("LUT3D")
    def from_xml(xml):
        """
        Parse and return the LUT3D from the given XML node. Returns None if the given
        element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.
        """
        if xml is None:
            return None
        super_args = ProcessNode.parse_attributes(xml)
        array = Array.from_xml(child_element(xml, "Array"))

        half_domain = xml.get("halfDomain") == "true"
        raw_halfs = xml.get("rawHalfs") == "true"
        interpolation = Interpolation3D(xml.get("interpolation"))
        return LUT3D(
            array=array,
            half_domain=half_domain,
            raw_halfs=raw_halfs,
            interpolation=interpolation,
            **super_args,
        )


@dataclass
class Matrix(ProcessNode):
    """
    Represents a Matrix element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#matrix
    """

    array: Array

    @staticmethod
    @register_process_node_xml_constructor("Matrix")
    def from_xml(xml):
        """
        Parse and return the Matrix from the given XML node. Returns None if the given
        element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.
        """
        if xml is None:
            return None
        super_args = ProcessNode.parse_attributes(xml)
        array = Array.from_xml(child_element(xml, "Array"))
        return Matrix(array=array, **super_args)


class RangeStyle(enum.Enum):
    """
    Represents the valid values of the style attribute within a Range element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#range
    """

    CLAMP = "Clamp"
    NO_CLAMP = "noClamp"


@dataclass
class Range(ProcessNode):
    """
    Represents a Range element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#range
    """

    min_in_value: Optional[float]
    max_in_value: Optional[float]
    min_out_value: Optional[float]
    max_out_value: Optional[float]

    style: Optional[RangeStyle]

    @staticmethod
    @register_process_node_xml_constructor("Range")
    def from_xml(xml):
        """
        Parse and return the Range from the given XML node. Returns None if the given
        element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.
        """
        if xml is None:
            return None

        super_args = ProcessNode.parse_attributes(xml)

        min_in_value = float(element_as_text(xml, "minInValue"))
        max_in_value = float(element_as_text(xml, "maxInValue"))
        min_out_value = float(element_as_text(xml, "minOutValue"))
        max_out_value = float(element_as_text(xml, "maxOutValue"))

        style = map_optional(RangeStyle, xml.get("style"))

        return Range(
            min_in_value=min_in_value,
            max_in_value=max_in_value,
            min_out_value=min_out_value,
            max_out_value=max_out_value,
            style=style,
            **super_args,
        )


class LogStyle(enum.Enum):
    """
    Represents the valid values of the style attribute in a Log element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#processList
    """

    LOG_10 = "log10"
    ANTI_LOG_10 = "antiLog10"
    LOG_2 = "log2"
    ANTI_LOG_2 = "antiLog2"
    LIN_TO_LOG = "linToLog"
    LOG_TO_LIN = "logToLin"
    CAMERA_LIN_TO_LOG = "cameraLinToLog"
    CAMERA_LOG_TO_LIN = "cameraLogToLin"


class Channel(enum.Enum):
    """
    Represents the valid values of the channel attribute in the Range element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#ranges
    """

    R = "R"
    G = "G"
    B = "B"


@dataclass
class LogParams:
    """
    Represents a Log Param List element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#log
    """

    base: Optional[float]
    log_side_slope: Optional[float]
    log_side_offset: Optional[float]
    lin_side_slope: Optional[float]
    lin_side_offset: Optional[float]
    lin_side_break: Optional[float]
    linear_slope: Optional[float]
    channel: Optional[Channel]

    @staticmethod
    def from_xml(xml):
        """
        Parse and return the Log Param from the given XML node. Returns None if the
        given element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.
        """
        if xml is None:
            return None
        attributes = retrieve_attributes_as_float(
            xml,
            {
                "base": "base",
                "log_side_slope": "logSideSlope",
                "log_side_offset": "logSideOffset",
                "lin_side_slope": "linSideSlope",
                "lin_side_offset": "linSideOffset",
                "lin_side_break": "linSideBreak",
                "linear_slope": "linearSlope",
            },
        )

        channel = map_optional(Channel, xml.get("channel"))

        return LogParams(channel=channel, **attributes)


@dataclass
class Log(ProcessNode):
    """
    Represents a Log element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#log
    """

    style: LogStyle
    log_params: Optional[LogParams]

    @staticmethod
    @register_process_node_xml_constructor("Log")
    def from_xml(xml):
        """
        Parse and return the Log from the given XML node. Returns None if the given
        element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.
        """
        if xml is None:
            return None
        super_args = ProcessNode.parse_attributes(xml)
        style = LogStyle(xml.get("style"))
        log_params = LogParams.from_xml(child_element(xml, "LogParams"))

        return Log(style=style, log_params=log_params, **super_args)


class ExponentStyle(enum.Enum):
    """
    Represents the valid values of the style attribute of an Exponent element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#exponent
    """

    BASIC_FWD = "basicFwd"
    BASIC_REV = "basicRev"
    BASIC_MIRROR_FWD = "basicMirrorFwd"
    BASIC_MIRROR_REV = "basicMirrorRev"
    BASIC_PASS_THRU_FWD = "basicPassThruFwd"  # noqa: S105
    BASIC_PASS_THRU_REV = "basicPassThruRev"  # noqa: S105
    MON_CURVE_FWD = "monCurveFwd"
    MON_CURVE_REV = "monCurveRev"
    MON_CURVE_MIRROR_FWD = "monCurveMirrorFwd"
    MON_CURVE_MIRROR_REV = "monCurveMirrorRev"


@dataclass
class ExponentParams:
    """
    Represents a Exponent Params element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#exponent
    """

    exponent: float
    offset: Optional[float]
    channel: Optional[Channel]

    @staticmethod
    def from_xml(xml):
        """
        Parse and return the Exponent Params from the given XML node. Returns None if
        the given element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.
        """
        if xml is None:
            return None
        attributes = retrieve_attributes_as_float(
            xml,
            {
                "exponent": "exponent",
                "offset": "offset",
            },
        )
        must_have(
            attributes["exponent"], "Exponent process node has no `exponent' value."
        )
        channel = map_optional(Channel, xml.get("channel"))

        return ExponentParams(channel=channel, **attributes)


@dataclass
class Exponent(ProcessNode):
    """
    Represents a Exponent element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#exponent
    """

    style: ExponentStyle
    exponent_params: Optional[ExponentParams]

    @staticmethod
    @register_process_node_xml_constructor("Exponent")
    def from_xml(xml):
        """
        Parse and return the Exponent from the given XML node. Returns None if the given
        element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.
        """
        if xml is None:
            return None
        super_args = ProcessNode.parse_attributes(xml)
        style = map_optional(ExponentStyle, xml.get("style"))
        log_params = ExponentParams.from_xml(child_element(xml, "ExponentParams"))
        return Exponent(style=style, exponent_params=log_params, **super_args)


class ASC_CDL_Style(enum.Enum):
    """
    Represents the valid values of the style attribute of an ASC_CDL element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#asc_cdl
    """

    FWD = "Fwd"
    REV = "Rev"
    FWD_NO_CLAMP = "FwdNoClamp"
    REV_NO_CLAMP = "RevNoClamp"


def three_floats(s) -> tuple[float, float, float]:
    if s is None:
        raise CLFValidationError(f"Failed to parse three float values from {s}")
    parts = s.split()
    if len(parts) != 3:
        raise CLFValidationError(f"Failed to parse three float values from {s}")
    values = tuple(map(float, parts))
    # Repacking here to satisfy type check.
    return values[0], values[1], values[2]


@dataclass
class SOPNode:
    """
    Represents a SOPNode element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#asc_cdl
    """

    slope: tuple[float, float, float]
    offset: tuple[float, float, float]
    power: tuple[float, float, float]

    @staticmethod
    def from_xml(xml):
        """
        Parse and return the SOPNode from the given XML node. Returns None if the given
        element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.
        """
        if xml is None:
            return None
        slope = three_floats(child_element(xml, "Slope").text)
        offset = three_floats(child_element(xml, "Offset").text)
        power = three_floats(child_element(xml, "Power").text)
        return SOPNode(slope=slope, offset=offset, power=power)


@dataclass
class SatNode:
    """
    Represents a SatNode element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#asc_cdl
    """

    saturation: float

    @staticmethod
    def from_xml(xml):
        """
        Parse and return the SatNode from the given XML node. Returns None if the given
        element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.
        """
        if xml is None:
            return None
        saturation = float(child_element(xml, "Saturation").text)
        return SatNode(saturation=saturation)


@dataclass
class ASC_CDL(ProcessNode):
    """
    Represents a ASC_CDL element.

    References
    ----------
    https://docs.acescentral.com/specifications/clf/#asc_cdl
    """

    style: ASC_CDL_Style
    sopnode: Optional[SOPNode]
    sat_node: Optional[SatNode]

    @staticmethod
    @register_process_node_xml_constructor("ASC_CDL")
    def from_xml(xml):
        """
        Parse and return the ASC_CDL from the given XML node. Returns None if the given
        element is None.

        Expects the xml element to be a valid element according to the CLF
        specification.

        Raises
        ------
        :class: CLFValidationError
            If the node does not conform to the specification, a `CLFValidationError`
            will be raised. The error message will indicate the details of the issue
            that was encountered.
        """
        if xml is None:
            return None
        super_args = ProcessNode.parse_attributes(xml)
        style = ASC_CDL_Style(xml.get("style"))
        sopnode = SOPNode.from_xml(child_element(xml, "SOPNode"))
        sat_node = SatNode.from_xml(child_element(xml, "SatNode"))
        return ASC_CDL(style=style, sopnode=sopnode, sat_node=sat_node, **super_args)


def read_clf(path) -> ProcessList:
    """
    Read given *CLF* file and return the resulting `ProcessList`.

    Parameters
    ----------
    path
        Path to the *CLF* file.

    Returns
    -------
    :class: colour.clf.ProcessList

    Raises
    ------
    :class: CLFValidationError
        If the given file does not contain a valid CLF document.

    """
    # Security issues in lxml should be addressed and no longer be a concern:
    # https://discuss.python.org/t/status-of-defusedxml-and-recommendation-in-docs/34762/6
    xml = lxml.etree.parse(path)  # noqa: S320
    xml_process_list = xml.getroot()
    root = ProcessList.from_xml(xml_process_list)
    return root


def parse_clf(text):
    """
    Read given string as a *CLF* document and return the resulting `ProcessList`.

    Parameters
    ----------
    text
        String that contains the *CLF* document.

    Returns
    -------
    :class: colour.clf.ProcessList.

    Raises
    ------
    :class: CLFValidationError
        If the given string does not contain a valid CLF document.

    """
    xml = lxml.etree.fromstring(text)  # noqa: S320
    root = ProcessList.from_xml(xml)
    return root
