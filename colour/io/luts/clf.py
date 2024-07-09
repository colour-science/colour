from __future__ import division, unicode_literals

import re
from xml.dom import minidom
from xml.etree import ElementTree

import numpy as np
from six import StringIO

from colour import LinearInterpolator
from colour.constants import DTYPE_FLOAT_DEFAULT, DTYPE_INT_DEFAULT
from colour.io.luts import (
    ASC_CDL,
    LUT1D,
    LUT3D,
    AbstractLUTSequenceOperator,
    Exponent,
    LUT3x1D,
    LUTSequence,
    Matrix,
    Range,
)
from colour.utilities import as_float, as_float_array, tsplit, tstack

__all__ = [
    "half_to_uint16",
    "uint16_to_half",
    "half_domain_lookup",
    "HalfDomain1D",
    "HalfDomain3x1D",
    "read_index_map",
    "string_to_array",
    "read_clf",
    "write_clf",
]


def half_to_uint16(x):
    x_half = np.copy(np.asarray(x)).astype(np.float16)

    return np.frombuffer(x_half.tobytes(), dtype=np.uint16).reshape(x_half.shape)


def uint16_to_half(y):
    y_int = np.copy(np.asarray(y)).astype(np.uint16)

    return as_float(
        np.frombuffer(y_int.tobytes(), dtype=np.float16)
        .reshape(y_int.shape)
        .astype(DTYPE_FLOAT_DEFAULT)
    )


def half_domain_lookup(x, LUT, raw_halfs=True):
    h0 = half_to_uint16(x)  # nearest integer which codes for x
    f0 = uint16_to_half(h0)  # convert back to float
    f1 = uint16_to_half(h0 + 1)  # float value for next integer
    # find h1 such that h0 and h1 code floats either side of x
    h1 = np.where((x - f0) * (x - f1) > 0, h0 - 1, h0 + 1)
    # ratio of position of x in the interval
    ratio = np.where(f0 == x, 1.0, (x - uint16_to_half(h1)) / (f0 - uint16_to_half(h1)))
    # get table entries either side of x
    out0 = LUT[h0]
    out1 = LUT[h1]

    if raw_halfs:  # convert table entries to float if necessary
        out0 = uint16_to_half(out0)
        out1 = uint16_to_half(out1)

    # calculate linear interpolated value between table entries
    f = LinearInterpolator(out1, out0)
    return f(ratio)


class HalfDomain1D(AbstractLUTSequenceOperator):
    def __init__(self, table=None, name="", comments=None, raw_halfs=False):
        if table is not None:
            self.table = table
        else:
            self.table = self.linear_table(raw_halfs=raw_halfs)
        self.name = name
        self.comments = comments
        self.raw_halfs = raw_halfs

    # TODO: Add properties.

    def _validate_table(self, table):
        table = np.asarray(table)

        assert table.shape == (65536,), "The table must be 65536 lines!"

        return table

    @staticmethod
    def linear_table(raw_halfs=False):
        if raw_halfs:
            return np.arange(65536, dtype="uint16")
        else:
            return uint16_to_half(np.arange(65536, dtype="uint16"))

    def apply(self, RGB):
        return half_domain_lookup(RGB, self.table, raw_halfs=self.raw_halfs)


class HalfDomain3x1D(AbstractLUTSequenceOperator):
    def __init__(self, table=None, name="", comments=None, raw_halfs=False):
        if table is not None:
            self.table = table
        else:
            self.table = self.linear_table(raw_halfs=raw_halfs)
        self.name = name
        self.comments = comments
        self.raw_halfs = raw_halfs

    # TODO: Add properties.

    def _validate_table(self, table):
        table = np.asarray(table)

        assert table.shape == (65536, 3), "The table must be 65536 x 3!"

        return table

    @staticmethod
    def linear_table(raw_halfs=False):
        if raw_halfs:
            samples = np.arange(65536, dtype="uint16")

            return tstack([samples, samples, samples]).astype(np.uint16)
        else:
            samples = uint16_to_half(np.arange(65536, dtype="uint16"))

            return tstack([samples, samples, samples])

    def apply(self, RGB):
        R, G, B = tsplit(as_float_array(RGB))
        table_R, table_G, table_B = tsplit(self.table)

        R = half_domain_lookup(R, table_R, raw_halfs=self.raw_halfs)
        G = half_domain_lookup(G, table_G, raw_halfs=self.raw_halfs)
        B = half_domain_lookup(B, table_B, raw_halfs=self.raw_halfs)

        return tstack([R, G, B])


def read_index_map(index_map):
    index_map = index_map.strip().split()
    index_map = np.core.defchararray.partition(index_map, "@")
    domain, at, table = np.hsplit(index_map, 3)
    domain = domain.reshape(-1).astype(np.float_)
    table = table.reshape(-1).astype(np.float_)

    return LUT1D(table=table, domain=domain)


def string_to_array(string, dimensions=None):
    assert dimensions is not None, "Dimensions must be set to dimensions of array!"

    return np.fromstring(string, count=dimensions[0] * dimensions[1], sep=" ").reshape(
        dimensions
    )


def parse_array(array):
    return np.array(array.strip().split()).astype(DTYPE_FLOAT_DEFAULT)


def ns_strip(s):
    return re.sub("{.+}", "", s)


def add_LUT1D(LUT, node):
    shaper, is_half_domain, is_raw_halfs = None, None, None
    if "halfDomain" in node:
        is_half_domain = bool(node.attrib["halfDomain"])

        if "rawHalfs" in node:
            is_raw_halfs = bool(node.attrib["rawHalfs"])

    for child in node:
        if child.tag.lower().endswith("indexmap"):
            shaper = read_index_map(child.text)

        if child.tag.lower().endswith("array"):
            dimensions = child.attrib["dim"]
            rows = DTYPE_INT_DEFAULT(dimensions.split()[0])
            columns = DTYPE_INT_DEFAULT(dimensions.split()[1])
            array = string_to_array(child.text, dimensions=(rows, columns))

            if columns == 1:
                if is_half_domain:
                    LUT_1 = HalfDomain1D(
                        table=array.reshape(-1), raw_halfs=is_raw_halfs
                    )
                else:
                    LUT_1 = LUT1D(table=array.reshape(-1))
            else:
                if is_half_domain:
                    LUT_1 = HalfDomain3x1D(table=array, raw_halfs=is_raw_halfs)
                else:
                    LUT_1 = LUT3x1D(table=array)
    if shaper:
        # Fully enumerated.
        if np.all(shaper.table == np.arange(shaper.size)):
            LUT_1.domain = shaper.domain
        else:
            if shaper.domain.shape == (2,):
                if columns == 1:
                    LUT_1.domain = shaper.domain
                else:
                    LUT_1.domain = tstack(
                        [
                            shaper.domain,
                            shaper.domain,
                            shaper.domain,
                        ]
                    )
            else:
                shaper.table /= np.max(shaper.table)

                if "name" in node:
                    shaper.name = f"{node.attrib['name']}_shaper"

                LUT.append(shaper)

    if "name" in node:
        LUT_1.name = node.attrib["name"]

    LUT.append(LUT_1)

    return LUT


def add_LUT3D(LUT, node):
    shaper = None
    for child in node:
        tag = child.tag.lower()
        if tag.endswith("indexmap"):
            shaper = read_index_map(child.text)
            shaper.table /= np.max(shaper.table)
        elif child.tag.lower().endswith("array"):
            dimensions = child.attrib["dim"]
            size_R = DTYPE_INT_DEFAULT(dimensions.split()[0])
            size_G = DTYPE_INT_DEFAULT(dimensions.split()[1])
            size_B = DTYPE_INT_DEFAULT(dimensions.split()[2])
            length = size_R * size_G * size_B
            channels = DTYPE_INT_DEFAULT(dimensions.split()[3])
            array = string_to_array(child.text, dimensions=(length, channels))
            array = array.reshape([size_R, size_G, size_B, 3], order="C")
            LUT_1 = LUT3D(table=array)

    if shaper:
        if shaper.domain.shape[0] == 2:
            LUT_1.domain = tstack(
                [
                    shaper.domain,
                    shaper.domain,
                    shaper.domain,
                ]
            )
        else:
            if "name" in node:
                shaper.name = f"{node.attrib['name']}_shaper"

            LUT.append(shaper)

    if "name" in node:
        LUT_1.name = node.attrib["name"]

    LUT.append(LUT_1)

    return LUT


def add_Range(LUT, node):
    operator = Range()

    if "name" in node:
        operator.name = node.attrib["name"]

    if "style" in node:
        style = node.attrib["style"].lower()

        if style == "noclamp":
            operator.no_clamp = True
        elif style == "clamp":
            operator.no_clamp = False

    for child in node:
        tag = child.tag.lower()

        if tag.endswith("mininvalue"):
            operator.min_in_value = float(child.text)
        elif tag.endswith("maxinvalue"):
            operator.max_in_value = float(child.text)
        elif tag.endswith("minoutvalue"):
            operator.min_out_value = float(child.text)
        elif tag.endswith("maxoutvalue"):
            operator.max_out_value = float(child.text)

    LUT.append(operator)

    return LUT


def add_Matrix(LUT, node):
    operator = Matrix()

    if "name" in node:
        operator.name = node.attrib["name"]

    for child in node:
        if child.tag.endswith("Array"):
            dimensions = child.attrib["dim"]
            rows = DTYPE_INT_DEFAULT(dimensions.split()[0])
            columns = DTYPE_INT_DEFAULT(dimensions.split()[1])
            operator.array = string_to_array(child.text, dimensions=(rows, columns))

    LUT.append(operator)

    return LUT


def add_ASC_CDL(LUT, node):
    operator = ASC_CDL()

    if "name" in node:
        operator.name = node.attrib["name"]

    if "style" in node:
        style = node.attrib["style"].lower()

        if style == "fwdnoclamp":
            operator.clamp, operator.reverse = False, False
        elif style == "revnoclamp":
            operator.clamp, operator.reverse = False, True
        elif style == "fwd":
            operator.clamp, operator.reverse = True, False
        elif style == "reverse":
            operator.clamp, operator.reverse = True, True

    for child in node:
        if child.tag.lower().endswith("sopnode"):
            for grand_child in child:
                tag = grand_child.tag.lower()

                if tag.endswith("slope"):
                    operator.slope = parse_array(grand_child.text)
                elif tag.endswith("offset"):
                    operator.offset = parse_array(grand_child.text)
                elif tag.endswith("power"):
                    operator.power = parse_array(grand_child.text)

        if child.tag.lower().endswith("satnode"):
            for grand_child in child:
                if grand_child.tag.lower().endswith("saturation"):
                    operator.saturation = float(grand_child.text)

    LUT.append(operator)

    return LUT


def add_Exponent(LUT, node):
    operator = Exponent(exponent=[1, 1, 1], offset=[0, 0, 0])

    if "name" in node:
        operator.name = node.attrib["name"]

    if "style" in node:
        operator.style = node.attrib["style"]

    for child in node:
        if child.tag.lower().endswith("exponentparams"):
            if "channel" in child:
                if child.attrib["channel"].lower() == "r":
                    operator.exponent[0] = child.attrib["exponent"]
                    if "offset" in child:
                        operator.offset[0] = child.attrib["offset"]
                elif child.attrib["channel"].lower() == "g":
                    operator.exponent[1] = child.attrib["exponent"]
                    if "offset" in child:
                        operator.offset[1] = child.attrib["offset"]
                elif child.attrib["channel"].lower() == "b":
                    operator.exponent[2] = child.attrib["exponent"]
                    if "offset" in child:
                        operator.offset[2] = child.attrib["offset"]
            else:
                operator.exponent = [
                    child.attrib["exponent"],
                    child.attrib["exponent"],
                    child.attrib["exponent"],
                ]
                if "offset" in child:
                    operator.offset = [
                        child.attrib["offset"],
                        child.attrib["offset"],
                        child.attrib["offset"],
                    ]

    LUT.append(operator)

    return LUT


def read_clf(path):
    LUT = LUTSequence()
    tree = ElementTree.parse(path)
    process_list = tree.getroot()
    # if 'name' in process_list.keys():
    # LUT.name = process_list.attrib['name']
    for node in process_list:
        tag = node.tag.lower()
        # if node.tag.endswith('Description'):
        # LUT.comments.append(node.text)
        # if node.tag.endswith('InputDescriptor'):
        # LUT.InputDescriptor = node.text
        # if node.tag.endswith('OutputDescriptor'):
        # LUT.OutputDescriptor = node.text
        if tag.endswith("lut1d"):
            LUT = add_LUT1D(LUT, node)
        elif tag.endswith("lut3d"):
            LUT = add_LUT3D(LUT, node)
        elif tag.endswith("range"):
            LUT = add_Range(LUT, node)
        elif tag.endswith("matrix"):
            LUT = add_Matrix(LUT, node)
        elif tag.endswith("asc_cdl"):
            LUT = add_ASC_CDL(LUT, node)
        elif tag.endswith("exponent"):
            LUT = add_Exponent(LUT, node)

    return LUT


def write_clf(LUT, path, name="", id="", decimals=10):
    def _format_array(array, decimals=10):
        buffer = StringIO()
        if array.dtype != np.uint16:
            np.savetxt(buffer, array, fmt=str(f"%.{decimals}f"))
        else:
            np.savetxt(buffer, array, fmt="%d")
        return ("\n" + buffer.getvalue()).replace("\n", "\n\t\t\t")[:-1]

    def _format_row(array, decimals=10):
        return "{1:0.{0}f} {2:0.{0}f} {3:0.{0}f}".format(decimals, *array)

    def _add_comments(node, comments):
        for comment in comments:
            d = ElementTree.SubElement(node, "Description")
            d.text = comment

    process_list = ElementTree.Element("ProcessList")
    process_list.set("xmlns", "urn:NATAS:AMPAS:LUT:v2.0")
    process_list.set("id", id)
    process_list.set("name", name)
    process_list.set("compCLFversion", "2.0")
    for node in LUT:
        if isinstance(node, ASC_CDL):
            process_node = ElementTree.Element("ASC_CDL")

            if node.comments:
                _add_comments(process_node, node.comments)

            if node.reverse and not node.clamp:
                process_node.set("style", "RevNoClamp")

            if node.reverse and node.clamp:
                process_node.set("style", "Rev")

            if not node.reverse and node.clamp:
                process_node.set("style", "FwdClamp")

            if not node.reverse and not node.clamp:
                process_node.set("style", "FwdNoClamp")

            sop_node = ElementTree.SubElement(process_node, "SOPNode")
            slope = ElementTree.SubElement(sop_node, "Slope")
            slope.text = _format_row(node.slope, decimals=decimals)
            offset = ElementTree.SubElement(sop_node, "Offset")
            offset.text = _format_row(node.offset, decimals=decimals)
            power = ElementTree.SubElement(sop_node, "power")
            power.text = _format_row(node.power, decimals=decimals)
            sat_node = ElementTree.SubElement(process_node, "SatNode")
            saturation = ElementTree.SubElement(sat_node, "Saturation")
            saturation.text = "{1:0.{0}f}".format(decimals, node.saturation)

        if isinstance(node, HalfDomain1D):
            process_node = ElementTree.Element("LUT1D")
            process_node.set("halfDomain", "True")
            array = ElementTree.SubElement(process_node, "Array")
            array.set("dim", "65536 1")

            if node.raw_halfs:
                process_node.set("rawHalfs", "True")
                array.text = _format_array(node.table)
            else:
                array.text = _format_array(node.table, decimals=decimals)

        if isinstance(node, HalfDomain3x1D):
            process_node = ElementTree.Element("LUT1D")
            process_node.set("halfDomain", "True")
            array = ElementTree.SubElement(process_node, "Array")
            array.set("dim", "65536 3")

            if node.raw_halfs:
                process_node.set("rawHalfs", "True")
                array.text = _format_array(node.table)
            else:
                array.text = _format_array(node.table, decimals=decimals)

        if isinstance(node, LUT1D):
            process_node = ElementTree.Element("LUT1D")
            if not np.all(node.domain == np.array([0, 1])):
                index_map = ElementTree.SubElement(process_node, "IndexMap")

                if node.domain.shape == (2,):
                    index_map.set("dim", "2")
                    index_map.text = (
                        f"{node.domain[0]}@0 {node.domain[1]}@{node.size - 1}"
                    )
                else:
                    index_text = ""

                    for i in range(len(node.domain)):
                        index_text = "{1} {2:0.{0}f}@{3}".format(
                            decimals,
                            index_text,
                            node.domain[i],
                            int((node.size - 1) * node.table[i] / np.max(node.table)),
                        )

                    index_map.text = index_text[1:]

            if node.comments:
                _add_comments(process_node, node.comments)

            array = ElementTree.SubElement(process_node, "Array")
            array.set("dim", f"{node.size} 1")
            array.text = _format_array(node.table, decimals=decimals)

        if isinstance(node, LUT3x1D):
            process_node = ElementTree.Element("LUT1D")

            if not np.all(node.domain == np.array([[0, 0, 0], [1, 1, 1]])):
                index_map = ElementTree.SubElement(process_node, "IndexMap")
                index_map.set("dim", "2")
                index_map.text = "{1:0.{0}f}@0 {2:0.{0}f}@{3}".format(
                    decimals, node.domain[0][0], node.domain[1][0], node.size - 1
                )  # assuming consistent domain

            if node.comments:
                _add_comments(process_node, node.comments)

            array = ElementTree.SubElement(process_node, "Array")
            array.set("dim", f"{node.size} 3")
            array.text = _format_array(node.table, decimals=decimals)

        if isinstance(node, LUT3D):
            process_node = ElementTree.Element("LUT3D")

            if not np.all(node.domain == np.array([[0, 0, 0], [1, 1, 1]])):
                index_map = ElementTree.SubElement(process_node, "IndexMap")
                index_map.set("dim", "2")
                index_map.text = "{1:0.{0}f}@0 {2:0.{0}f}@{3}".format(
                    decimals, node.domain[0][0], node.domain[1][0], node.size - 1
                )  # assuming consistent domain

            if node.comments:
                _add_comments(process_node, node.comments)

            array = ElementTree.SubElement(process_node, "Array")
            array.set("dim", f"{node.size} {node.size} {node.size} 3")
            array.text = _format_array(
                node.table.reshape(-1, 3, order="C"), decimals=decimals
            )

        if isinstance(node, Matrix):
            process_node = ElementTree.Element("Matrix")

            if node.comments:
                _add_comments(process_node, node.comments)

            array = ElementTree.SubElement(process_node, "Array")
            array.set("dim", f"{node.array.shape[0]} {node.array.shape[1]} 3")
            array.text = _format_array(node.array, decimals=decimals)

        if isinstance(node, Range):
            process_node = ElementTree.Element("Range")

            if node.comments:
                _add_comments(process_node, node.comments)

            min_in = ElementTree.SubElement(process_node, "minInValue")
            min_in.text = "{1:0.{0}f}".format(decimals, node.min_in_value)
            max_in = ElementTree.SubElement(process_node, "maxInValue")
            max_in.text = "{1:0.{0}f}".format(decimals, node.max_in_value)
            min_out = ElementTree.SubElement(process_node, "minOutValue")
            min_out.text = "{1:0.{0}f}".format(decimals, node.min_out_value)
            max_out = ElementTree.SubElement(process_node, "maxOutValue")
            max_out.text = "{1:0.{0}f}".format(decimals, node.max_out_value)

            if not node.no_clamp:
                process_node.set("noClamp", "False")
            else:
                process_node.set("noClamp", "True")

        if isinstance(node, Exponent):
            process_node = ElementTree.Element("Exponent")
            process_node.set("style", node.style)

            if node.comments:
                _add_comments(process_node, node.comments)

            if (node.offset[0] == node.offset[1] == node.offset[2]) and (
                node.exponent[0] == node.exponent[2] == node.exponent[2]
            ):
                ep = ElementTree.SubElement(process_node, "ExponentParams")
                ep.set("exponent", f"{node.exponent[0]}")
                if node.style.lower()[:8] == "moncurve":
                    ep.set("offset", f"{node.offset[0]}")
            else:
                ep_red = ElementTree.SubElement(process_node, "ExponentParams")
                ep_red.set("channel", "R")
                ep_red.set("exponent", f"{node.exponent[0]}")
                if node.style.lower()[:8] == "moncurve":
                    ep_red.set("offset", f"{node.offset[0]}")
                ep_green = ElementTree.SubElement(process_node, "ExponentParams")
                ep_green.set("channel", "G")
                ep_green.set("exponent", f"{node.exponent[1]}")
                if node.style.lower()[:8] == "moncurve":
                    ep_green.set("offset", f"{node.offset[1]}")
                ep_blue = ElementTree.SubElement(process_node, "ExponentParams")
                ep_blue.set("channel", "B")
                ep_blue.set("exponent", f"{node.exponent[2]}")
                if node.style.lower()[:8] == "moncurve":
                    ep_blue.set("offset", f"{node.offset[2]}")

        process_node.set("inBitDepth", "32f")
        process_node.set("outBitDepth", "32f")
        process_node.set("name", node.name)
        process_list.append(process_node)

    xml_string = ElementTree.tostring(process_list, encoding="utf-8")

    with open(path, "w") as clf_file:
        clf_file.write(minidom.parseString(xml_string).toprettyxml())
