"""
ASC CDL Input Utilities
=======================

Defines *ASC CDL* correction operator related objects.

-   :class:`colour.io.ASC_CDL`
-   :func:`colour.io.read_LUT_cdl_xml`
-   :func:`colour.io.read_LUT_cdl_edl`
-   :func:`colour.io.read_LUT_cdl_ale`
"""
import os
import re
from xml.dom import minidom

import numpy as np

from colour.constants import DTYPE_FLOAT_DEFAULT, DTYPE_INT_DEFAULT
from colour.io.luts import AbstractLUTSequenceOperator, LUTSequence
from colour.models import gamma_function
from colour.utilities import as_float_array, tsplit, tstack

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013-2018 - Colour Developers"
__license__ = "New BSD License - http://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["ASC_CDL", "read_LUT_cdl_xml", "read_LUT_cdl_edl", "read_LUT_cdl_ale"]


class ASC_CDL(AbstractLUTSequenceOperator):
    """
    Defines an *ASC CDL* correction operator.

    Parameters
    ----------
    slope : array_like, optional
        Multipliers for the *R*, *G* and *B* channels.
    offset : array_like, optional
        Offsets added to the *R*, *G* and *B* channels.
    power : array_like, optional
        Exponents to which the *R*, *G* and *B* channels are raised.
    saturation : array_like, optional
        Saturation (Rec.709 weighted) applied to the *RGB* values.
    name : unicode, optional
        *ASC CDL* correction operator name.
    comments : array_like, optional
        Comments to add to the *ASC CDL* correction operator.
    clamp : boolean, optional
        Whether the output is clamped to range [0, 1].
    reverse : boolean, optional
        Whether to reverse/invert the correction.
    id : unicode, optional
        ID of the correction.

    Methods
    -------
    apply

    Examples
    --------
    Instantiating an identity *ASC CDL* correction operator:

    >>> print(ASC_CDL(name="Identity"))
    ASC CDL - Identity
    ------------------
    <BLANKLINE>
    <ColorCorrection id="">
        <SOPNode>
            <Slope> 1.0 1.0 1.0 </Slope>
             <Offset> 0.0 0.0 0.0 </Offset>
            <Power> 1.0 1.0 1.0 </Power>
        </SOPNode>
        <SATNode>
            <Saturation> 1.0 </Saturation>
        </SATNode>
    </ColorCorrection>
    <BLANKLINE>
    Clamping   : Yes
    Reverse    : No

    Instantiating an *ASC CDL* correction operator with comments:

    >>> print(
    ...     ASC_CDL(
    ...         slope=[1.1, 1.0, 0.9],
    ...         offset=[-0.1, 0.0, 0.1],
    ...         power=[0.9, 1.0, 1.1],
    ...         saturation=0.9,
    ...         name="Correction_01",
    ...         comments=["A first comment.", "A second comment."],
    ...     )
    ... )
    ASC CDL - Correction_01
    -----------------------

    <ColorCorrection id="">
        <SOPNode>
            <Slope> 1.1 1.0 0.9 </Slope>
            <Offset> -0.1 0.0 0.1 </Offset>
            <Power> 0.9 1.0 1.1 </Power>
        </SOPNode>
        <SATNode>
            <Saturation> 0.9 </Saturation>
        </SATNode>
    </ColorCorrection>

    Clamping   : Yes
    Reverse    : No

    Comment 01 : A first comment.
    Comment 02 : A second comment.
    """

    def __init__(
        self,
        slope=(1, 1, 1),
        offset=(0, 0, 0),
        power=(1, 1, 1),
        saturation=1.0,
        name="",
        comments=None,
        clamp=True,
        reverse=False,
        id="",
    ):
        self.slope = np.asarray(slope)
        self.offset = np.asarray(offset)
        self.power = np.asarray(power)
        self.saturation = saturation
        self.name = name
        self.comments = comments
        self.clamp = clamp
        self.reverse = reverse
        self.id = id

    # TODO: Add properties.

    def __str__(self):
        """
        Returns a formatted string representation of the *ASC CDL* correction operator.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        def _format_array(array):
            array = np.asarray(array)
            if array.shape == (3,):
                return f"{array[0]} {array[1]} {array[2]}"
            else:
                return f"{array} {array} {array}"

        if self.comments:
            comments = [
                f"Comment {str(i + 1).zfill(2)} : {comment}"
                for i, comment in enumerate(self.comments)
            ]

        return (
            "ASC CDL - {}\n"
            "{}\n\n"
            '<ColorCorrection id="{}">\n'
            "    <SOPNode>\n"
            "        <Slope> {} </Slope>\n"
            "        <Offset> {} </Offset>\n"
            "        <Power> {} </Power>\n"
            "    </SOPNode>\n"
            "    <SATNode>\n"
            "        <Saturation> {} </Saturation>\n"
            "    </SATNode>\n"
            "</ColorCorrection>\n\n"
            "Clamping   : {}\n"
            "Reverse    : {}"
            "{}".format(
                self.name,
                "-" * (10 + len(self.name)),
                self.id,
                _format_array(self.slope),
                _format_array(self.offset),
                _format_array(self.power),
                self.saturation,
                "Yes" if self.clamp else "No",
                "Yes" if self.reverse else "No",
                "\n\n{}".format("\n".join(comments)) if self.comments else "",
            )
        )

    def apply(self, RGB):
        """
        Applies the *ASC CDL* correction operator to given *RGB* array.

        Parameters
        ----------
        RGB : array_like
            *RGB* array to apply the *ASC CDL* correction operator to.

        Returns
        -------
        ndarray
            Corrected *RGB* array.

        Examples
        --------
        >>> cdl = ASC_CDL(
        ...     slope=[1.1, 1.0, 0.9],
        ...     offset=[-0.1, 0.0, 0.1],
        ...     power=[0.9, 1.0, 1.1],
        ...     saturation=0.9,
        ... )
        >>> RGB = [0.18, 0.18, 0.18]
        >>> cdl.apply(RGB)
        array([ 0.12841813,  0.17915636,  0.22339685])
        """

        RGB_out = as_float_array(np.copy(RGB))

        if self.reverse:
            if self.clamp:
                RGB_out = np.clip(RGB_out, 0, 1)

            if self.saturation != 1.0:
                R, G, B = tsplit(RGB_out)
                luma = 0.2126 * R + 0.7152 * G + 0.0722 * B
                luma = tstack([luma, luma, luma])
                RGB_out = luma + (1 / self.saturation) * (RGB_out - luma)

                if self.clamp:
                    RGB_out = np.clip(RGB_out, 0, 1)

            RGB_out = gamma_function(RGB_out, 1 / self.power, "preserve")
            RGB_out -= self.offset
            RGB_out /= self.slope

            if self.clamp:
                RGB_out = np.clip(RGB_out, 0, 1)
        else:
            RGB_out *= self.slope
            RGB_out += self.offset
            RGB_out = gamma_function(RGB_out, self.power, "preserve")

            if self.clamp:
                RGB_out = np.clip(RGB_out, 0, 1)

            if self.saturation != 1.0:
                R, G, B = tsplit(RGB_out)
                luma = 0.2126 * R + 0.7152 * G + 0.0722 * B
                luma = tstack([luma, luma, luma])
                RGB_out = luma + self.saturation * (RGB_out - luma)

                if self.clamp:
                    RGB_out = np.clip(RGB_out, 0, 1)

        return RGB_out


def read_LUT_cdl_xml(path):
    def _parse_array(array):
        return np.array(list(map(DTYPE_FLOAT_DEFAULT, array.split())))

    title = re.sub("_|-|\\.", " ", os.path.splitext(os.path.basename(path))[0])
    data = minidom.parse(path)
    LUT = LUTSequence()
    corrections = data.getElementsByTagName("ColorCorrection")

    for idx, correction in enumerate(corrections):
        event = ASC_CDL()
        slope = correction.getElementsByTagName("Slope")
        slope = "1 1 1" if not slope else slope[0].firstChild.data
        offset = correction.getElementsByTagName("Offset")
        offset = "0 0 0" if not offset else offset[0].firstChild.data
        power = correction.getElementsByTagName("Power")
        power = "1 1 1" if not power else power[0].firstChild.data
        saturation = correction.getElementsByTagName("Saturation")
        saturation = "1" if not saturation else saturation[0].firstChild.data

        if "id" in correction.attributes:
            event.id = correction.attributes["id"].value

        event.slope = _parse_array(slope)
        event.offset = _parse_array(offset)
        event.power = _parse_array(power)
        event.saturation = _parse_array(saturation)
        event.name = f"{title} ({idx + 1})"
        LUT.append(event)

    if len(LUT) == 1:
        LUT[0].name = title

        return LUT[0]
    else:
        return LUT


def read_LUT_cdl_edl(path):
    with open(path) as edl_file:
        edl_lines = edl_file.readlines()

    if "TITLE" in edl_lines[0]:
        title = edl_lines[0].split()[1]
    else:
        title = re.sub("_|-|\\.", " ", os.path.splitext(os.path.basename(path))[0])
    event_cdl = None
    has_cdl = False
    LUT = LUTSequence()
    for line in edl_lines:
        if len(line.split()) == 0:
            continue

        if line.split()[0].isdigit():
            if has_cdl:
                LUT.append(event_cdl)
                event_cdl = None
                has_cdl = False
            event_number = DTYPE_INT_DEFAULT(line.split()[0])
            event_cdl = ASC_CDL(name=f"{title} EV{event_number:04d}")
            event_cdl.comments = []
            continue

        if event_cdl and line[0] == "*":
            trimmed = line[1:].lstrip()

            if trimmed.startswith("ASC_SOP"):
                sop = re.sub(r"\)\s*\(|\s*\(|\s*\)", " ", trimmed).split()
                event_cdl.slope = np.array(sop[1:4]).astype(np.float)
                event_cdl.offset = np.array(sop[4:7]).astype(np.float)
                event_cdl.power = np.array(sop[7:]).astype(np.float)
                has_cdl = True
            elif trimmed.startswith("ASC_SAT"):
                event_cdl.saturation = float(trimmed.split()[1])
                has_cdl = True
            else:
                event_cdl.comments.append(trimmed)
    if event_cdl:
        LUT.append(event_cdl)

    return LUT


def read_LUT_cdl_ale(path):
    with open(path, "rU") as ale_file:
        ale_lines = ale_file.readlines()

    title = re.sub("_|-|\\.", " ", os.path.splitext(os.path.basename(path))[0])
    event_cdl = None
    LUT = LUTSequence()

    # TODO: Implement proper exception catching.
    try:
        header_line = ale_lines.index("Column\n") + 1
    except:
        raise ValueError("ALE format error")

    headers = ale_lines[header_line].split("\t")

    try:
        sop_index = headers.index("ASC_SOP")
        sat_index = headers.index("ASC_SAT")
        name_index = headers.index("Name")
    except:
        raise ValueError("No ASC CDL data")

    try:
        first_data = ale_lines.index("Data\n") + 1
    except:
        raise ValueError("ALE format error")

    for line in ale_lines[first_data:]:
        line_data = line.split("\t")
        sop = re.sub(r"\)\s*\(|\s*\(|\s*\)", " ", line_data[sop_index]).split()
        sat = line_data[sat_index]
        name = line_data[name_index]
        event_cdl = ASC_CDL(name=name)
        event_cdl.slope = np.array(sop[0:3]).astype(np.float)
        event_cdl.offset = np.array(sop[3:6]).astype(np.float)
        event_cdl.power = np.array(sop[6:]).astype(np.float)
        event_cdl.saturation = float(sat)
        LUT.append(event_cdl)

    return LUT
